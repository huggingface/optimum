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
from typing import TYPE_CHECKING, List
from collections import defaultdict

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


def _merge_linears(graph_module: "GraphModule", input_node: "Node", linear_nodes: List["Node"], linears: List[torch.nn.Linear]):
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
        fully_qualified_merged_linear_name = ".".join([linear_nodes[0].target.rsplit(".", maxsplit=1)[0], merged_linear_name])
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


def apply_normalization_factor_to_query(gm):
    """
    Transformation that applies the normalization factor directly to the query weights saving the computation at
    runtime.
    """
    # TODO: add safety checks to make sure that the transformation is applied to attention layers.
    named_modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module" and "query" in node.name:
            p = node
            while p and p.target != operator.truediv:
                p = p.next

            if not isinstance(p.args[0], torch.fx.Node) and p.args[0].target != torch.matmul:
                continue
            if not isinstance(p.args[1], torch.fx.Node):
                query = named_modules[node.target]
                query.weight = torch.nn.Parameter(query.weight / p.args[1])
                if hasattr(query, "bias"):
                    query.bias = torch.nn.Parameter(query.bias / p.args[1])
            p.replace_all_uses_with(p.args[0])

    gm.graph.lint()
    gm.recompile()
    return gm


def optimize_attention(gm):
    """
    Transformation that optimizes attention layers by:
        1. Appliying the normalization factor to the query weights instead of computing it at runtime
        2. Merging the query, key and value linear projections as one big linear projection.
        3. Merging the transpose_for_scores by transposing the output of the merged linear projection instead of doing
           it individually for the query, key and value.
    """
    gm = apply_normalization_factor_to_query(gm)

    graph = gm.graph
    candidates = defaultdict(list)
    named_modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            mod = named_modules[node.target]
            if isinstance(mod, torch.nn.Linear):
                input_node = node.args[0]
                candidates[input_node].append((node, mod))

    # Only keep the candidates with more than one linear and the ones with the same number of
    # output features.
    # TODO: add safety checks to make sure that the transformation is applied to attention layers.
    candidates = _keep_good_candidates(candidates)
    for input_node, t in candidates.items():
        linear_nodes, linears = list(zip(*t))
        _merge_linears(gm, input_node, linear_nodes, linears)

        def find_child_nodes_of_target(node, target_name):
            children = []
            for user in node.users:
                if user.target == target_name:
                    children.append(user)
            return children

        parent_node = linear_nodes[0].args[0]  # output of the linear.
        view_node = find_child_nodes_of_target(linear_nodes[0], "view")
        if len(view_node) != 1:
            continue
        view_node = view_node[0]
        view_shape = list(view_node.args[1:])
        insertion_idx = len(view_shape) - 2
        view_shape.insert(insertion_idx, len(linear_nodes))
        permute_node = find_child_nodes_of_target(view_node, "permute")
        if len(permute_node) != 1:
            continue
        permute_node = permute_node[0]
        permutation = list(permute_node.args[1:])
        permutation = [insertion_idx] + [i if i < insertion_idx else i + 1 for i in permutation]
        with graph.inserting_after(parent_node):
            view_args = tuple([parent_node] + view_shape)
            view_node = graph.call_method("view", args=view_args)
        with graph.inserting_after(view_node):
            permute_args = tuple([view_node] + permutation)
            permute_node = graph.call_method("permute", args=permute_args)
        for idx, node in enumerate(linear_nodes):
            v = find_child_nodes_of_target(node, "view")[0]
            p = find_child_nodes_of_target(v, "permute")[0]
            new_args = [permute_node, (idx,)]
            node.args = tuple(new_args)
            p.replace_all_uses_with(v.args[0])
            graph.erase_node(p)
            graph.erase_node(v)

    gm.graph.lint()
    gm.recompile()
    return gm
