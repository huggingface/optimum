# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import operator
from collections import defaultdict
from typing import TYPE_CHECKING, List

import torch


if TYPE_CHECKING:
    from torch.fx import GraphModule, Node


def _get_bias(linear: torch.nn.Linear) -> torch.Tensor:
    if hasattr(linear, "bias"):
        return linear.bias
    return torch.zeros(shape=(linear.out_features), dtype=linear.weight.dtype).to(linear.weight.device)


def _get_linear_module_name(linear_node):
    return linear_node.target.split(".")[-1]


def _find_parent(model: torch.nn.Module, module: torch.nn.Module) -> torch.nn.Module:
    """Finds the parent module of module in model"""
    parent = None
    for mod in model.children():
        if mod is module:
            return model
        parent = _find_parent(mod, module)
        if parent is not None:
            break

    return parent


def _merge_linears(
    graph_module: "GraphModule", input_node: "Node", linear_nodes: List["Node"], linears: List[torch.nn.Linear]
):
    in_features = linears[0].in_features
    out_features = [linear.out_features for linear in linears]
    total_out_features = sum(out_features)
    use_bias = any(hasattr(linear, "bias") for linear in linears)
    merged_linear = torch.nn.Linear(in_features, total_out_features, bias=use_bias)

    with torch.no_grad():
        new_weight = torch.cat([linear.weight for linear in linears], dim=0)
        merged_linear.weight = torch.nn.Parameter(new_weight)
        if use_bias:
            new_bias = torch.cat([_get_bias(linear) for linear in linears], dim=0)
            merged_linear.bias = torch.nn.Parameter(new_bias)

    linear_module_names = [_get_linear_module_name(node) for node in linear_nodes]
    merged_linear_name = "_".join(linear_module_names + ["merged"])
    parent_module = _find_parent(graph_module, linears[0])
    parent_module.add_module(merged_linear_name, merged_linear)
    for name in linear_module_names:
        delattr(parent_module, name)

    graph = graph_module.graph
    with graph.inserting_after(input_node):
        fully_qualified_merged_linear_name = ".".join(
            [linear_nodes[0].target.rsplit(".", maxsplit=1)[0], merged_linear_name]
        )
        merged_linear_node = graph.call_module(fully_qualified_merged_linear_name, args=(input_node,))

    accum_out_features = list(itertools.accumulate([0] + out_features))
    for idx, node in enumerate(linear_nodes):
        node.op = "call_function"
        node.target = operator.getitem
        slice_to_get = slice(accum_out_features[idx], accum_out_features[idx + 1])
        node.args = (merged_linear_node, (Ellipsis, slice_to_get))


def merge_linears(graph_module: "GraphModule") -> "GraphModule":
    """
    Transformation that merges linear layers that take the same input into one big linear layer.

    Args:
        graph_module ([`~torch.fx.GraphModule`]):
            The module to transform.

    Returns:
        `torch.fx.GraphModule`: The transformed module.
    """
    candidates = defaultdict(list)
    named_modules = dict(graph_module.named_modules())
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            mod = named_modules[node.target]
            if isinstance(mod, torch.nn.Linear):
                input_node = node.args[0]
                candidates[input_node].append((node, mod))

    # Only keep the candidates with more than one linear and the ones with the same number of
    # output features.
    candidates = {k: v for k, v in candidates.items() if len(v) > 1}

    for input_node, t in candidates.items():
        linear_nodes, linears = list(zip(*t))
        _merge_linears(graph_module, input_node, linear_nodes, linears)

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module
