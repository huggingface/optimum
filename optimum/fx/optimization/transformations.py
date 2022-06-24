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
import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List

import torch


if TYPE_CHECKING:
    from torch.fx import GraphModule, Node


class Transformation(ABC):
    """
    A torch.fx graph transformation.

    Attributes:
        preserves_computation (`bool`, defaults to `False`):
            Whether the transformation preserves the graph computation, if `True`, the original and the transformed
            graph should produce the same outputs.
    """

    preserves_computation: bool = False

    @abstractmethod
    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        """
        Applies the transformation to graph_module.

        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.

        Returns:
            `torch.fx.GraphModule`: The transformed module.
        """
        raise NotImplementedError("The transform method needs to be implemented.")

    def __call__(self, graph_module: "GraphModule", lint_and_recompile: bool = True) -> "GraphModule":
        f"""
        {self.__doc__}

        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.
            lint_and_recompile (`bool`, defaults to `True`):
                Whether the transformed module should be linted and recompiled.
                This can be set to `False` when chaining transformations together to perform this operation only once.

        Returns:
            `torch.fx.GraphModule`: The transformed module.

        Example:

        ```python
        >>> from transformers import BertModel
        >>> from transformers.utils.fx import symbolic_trace
        >>> from optimum.fx.optimization import {self.__class__.__name__}

        >>> model = BertModel.from_pretrained("bert-base-uncased")
        >>> traced = symbolic_trace(
        >>>     model,
        >>>     input_names=["input_ids", "attention_mask", "token_type_ids"],
        >>> )
        >>> transformation = {self.__class__.__name__}()
        >>> transformed_model = transformation(traced)
        ```
        """
        graph_module = self.transform(graph_module)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module


class ReversibleTransformation(Transformation):
    """
    A torch.fx graph transformation that is reversible.
    """

    @abstractmethod
    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        """
        Applies the reverse transformation to graph_module.

        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.
            lint_and_recompile (`bool`, defaults to `True`):
                Whether the transformed module should be linted and recompiled.
                This can be set to `False` when chaining transformations together to perform this operation only once.

        Returns:
            `torch.fx.GraphModule`: The transformed module.
        """
        raise NotImplementedError("The reverse transform method needs to be implemented.")

    def __call__(
        self, graph_module: "GraphModule", lint_and_recompile: bool = True, reverse: bool = False
    ) -> "GraphModule":
        f"""
        {self.__doc__}

        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.
            lint_and_recompile (`bool`, defaults to `True`):
                Whether the transformed module should be linted and recompiled.
                This can be set to `False` when chaining transformations together to perform this operation only once.
            reverse (`bool`, defaults to `False`):
                If `True`, the reverse transformation is performed.

        Returns:
            `torch.fx.GraphModule`: The transformed module.

        Example:

        ```python
        >>> from transformers import BertModel
        >>> from transformers.utils.fx import symbolic_trace
        >>> from optimum.fx.optimization import {self.__class__.__name__}

        >>> model = BertModel.from_pretrained("bert-base-uncased")
        >>> traced = symbolic_trace(
        >>>     model,
        >>>     input_names=["input_ids", "attention_mask", "token_type_ids"],
        >>> )
        >>> transformation = {self.__class__.__name__}()
        >>> transformed_model = transformation(traced)
        ```
        """
        func = self.transform if not reverse else self.reverse
        graph_module = func(graph_module)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module


class MergeLinears(ReversibleTransformation):
    """
    Transformation that merges linear layers that take the same input into one big linear layer.
    """

    preserves_computation = True

    @staticmethod
    def _get_bias(linear: torch.nn.Linear) -> torch.Tensor:
        if hasattr(linear, "bias"):
            return linear.bias
        return torch.zeros(shape=(linear.out_features), dtype=linear.weight.dtype).to(linear.weight.device)

    @staticmethod
    def _get_linear_module_name(linear_node):
        return linear_node.target.split(".")[-1]

    @staticmethod
    def _merge_linears(
        graph_module: "GraphModule", input_node: "Node", linear_nodes: List["Node"], linears: List[torch.nn.Linear]
    ):
        in_features = linears[0].in_features
        out_features = [linear.out_features for linear in linears]
        total_out_features = sum(out_features)
        use_bias = any(hasattr(linear, "bias") for linear in linears)
        if use_bias and not all(hasattr(linear, "bias") for linear in linears):
            warnings.warn(
                "Not all the linear layers that are merged contain a bias, but some do. By merging, this is equivalent "
                "to adding a bias to the layers missing one."
            )
        merged_linear = torch.nn.Linear(in_features, total_out_features, bias=use_bias)

        with torch.no_grad():
            new_weight = torch.cat([linear.weight for linear in linears], dim=0)
            merged_linear.weight = torch.nn.Parameter(new_weight)
            if use_bias:
                new_bias = torch.cat([MergeLinears._get_bias(linear) for linear in linears], dim=0)
                merged_linear.bias = torch.nn.Parameter(new_bias)

        linear_module_names = [MergeLinears._get_linear_module_name(node) for node in linear_nodes]
        merged_linear_name = "-".join(linear_module_names + ["merged"])
        fully_qualified_parent_name = linear_nodes[0].target.rsplit(".", maxsplit=1)[0]
        parent_module = graph_module.get_submodule(fully_qualified_parent_name)
        parent_module.add_module(merged_linear_name, merged_linear)
        for name in linear_module_names:
            delattr(parent_module, name)

        graph = graph_module.graph
        with graph.inserting_after(input_node):
            fully_qualified_merged_linear_name = ".".join([fully_qualified_parent_name, merged_linear_name])
            merged_linear_node = graph.call_module(fully_qualified_merged_linear_name, args=(input_node,))
            merged_linear_node.was_transformed = "MergeLinears"

        accum_out_features = list(itertools.accumulate([0] + out_features))
        for idx, node in enumerate(linear_nodes):
            node.op = "call_function"
            node.target = operator.getitem
            slice_to_get = slice(accum_out_features[idx], accum_out_features[idx + 1])
            node.args = (merged_linear_node, (Ellipsis, slice_to_get))

    @staticmethod
    def _unmerge_linears(graph_module: "GraphModule", merged_linear_node: "Node", merged_linear: torch.nn.Linear):
        merged_linear_name = merged_linear_node.target.split(".")[-1]
        # The linear module names and the output nodes need to be in the same order.
        # merge_linear_name gives the order in which the weights were concatenated, and we use the slice start index to
        # sort the output nodes since the start index tells when a weight was concatenated.
        linear_module_names = merged_linear_name.replace("-merged", "").split("-")
        output_nodes = sorted(merged_linear_node.users, key=lambda node: node.args[1][1].start)

        in_features = merged_linear.in_features
        out_features = []
        for node in output_nodes:
            slice_to_get = node.args[1][1]
            out_features.append(slice_to_get.stop - slice_to_get.start)

        linears = [
            torch.nn.Linear(in_features, out_feat, bias=hasattr(merged_linear, "bias")) for out_feat in out_features
        ]

        fully_qualified_parent_name = merged_linear_node.target.rsplit(".", maxsplit=1)[0]
        parent_module = graph_module.get_submodule(fully_qualified_parent_name)
        parent_module_name = merged_linear_node.target.rsplit(".", maxsplit=1)[0]
        for name, node, linear in zip(linear_module_names, output_nodes, linears):
            with torch.no_grad():
                slice_to_get = node.args[1][1]
                linear.weight = torch.nn.Parameter(merged_linear.weight[slice_to_get.start : slice_to_get.stop])
                if hasattr(merged_linear, "bias"):
                    linear.bias = torch.nn.Parameter(merged_linear.bias[slice_to_get.start : slice_to_get.stop])
            parent_module.add_module(name, linear)
            node.op = "call_module"
            node.target = ".".join([parent_module_name, name])
            node.args = (merged_linear_node.args[0],)

        delattr(parent_module, merged_linear_name)
        graph_module.graph.erase_node(merged_linear_node)

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        candidates = collections.defaultdict(list)
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
            self._merge_linears(graph_module, input_node, linear_nodes, linears)

        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        named_modules = dict(graph_module.named_modules())
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                if getattr(node, "was_transformed", "") == "MergeLinears":
                    self._unmerge_linears(graph_module, node, named_modules[node.target])

        return graph_module


class ChangeTrueDivToMulByInverse(ReversibleTransformation):
    """
    Transformation that changes truediv nodes to multiplication by the inverse when the denominator is static.
    For example, that is sometimes the case for the scaling factor in attention layers.
    """

    preserves_computation = True

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op == "call_function" and node.target == operator.truediv:
                x, y = node.args
                if not isinstance(y, torch.fx.Node):
                    node.target = operator.mul
                    node.args = (x, 1 / y)
                    node.was_transformed = "ChangeTrueDivToMulByInverse"

        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        graph = graph_module.graph
        for node in graph.nodes:
            if getattr(node, "was_transformed", "") == "ChangeTrueDivToMulByInverse":
                node.target = operator.truediv
                x, y = node.args
                node.args = (x, 1 / y)

        return graph_module


class DeepCopy(ReversibleTransformation):
    """
    Transformation that does nothing except making a deepcopy of the graph module.
    """

    preserves_computation = True

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        clone = copy.deepcopy(graph_module)
        # This is needed because copy.deepcopy does not take care of it.
        # Without these attributes, the reverse transformation cannot be done.
        for n1, n2 in zip(graph_module.graph.nodes, clone.graph.nodes):
            if hasattr(n1, "was_transformed"):
                n2.was_transformed = n1.was_transformed
        return clone

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        return self.transform(graph_module)


class LintAndRecompile(ReversibleTransformation):
    """
    Transformation that does nothing except linting and recompiling the graph module.
    """

    preserves_computation = True

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module


def compose(*args: Transformation, inplace: bool = True) -> Callable[["GraphModule"], "GraphModule"]:
    """
    Composes a list of transformations together.
    """
    transformations = list(reversed(args))

    composition_preserves_computation = all(t.preserves_computation for t in transformations)
    composition_is_reversible = all((isinstance(t, ReversibleTransformation) for t in transformations))

    if not inplace:
        transformations.append(DeepCopy())

    if not composition_is_reversible:

        def reduce_fn(f, g):
            def composition(graph_module, lint_and_recompile=False):
                return f(g(graph_module, lint_and_recompile=lint_and_recompile))

            return composition

        class ComposeTransformation(Transformation):
            preserves_computation = composition_preserves_computation

            def transform(self, graph_module):
                return functools.reduce(reduce_fn, transformations)(graph_module)

    else:

        def make_reduce_fn(reverse):
            def reduce_fn(f, g):
                def composition(graph_module, lint_and_recompile=False, reverse=reverse):
                    return f(
                        g(graph_module, lint_and_recompile=lint_and_recompile, reverse=reverse),
                        lint_and_recompile=lint_and_recompile,
                        reverse=reverse,
                    )

                return composition

            return reduce_fn

        class ComposeTransformation(ReversibleTransformation):
            preserves_computation = composition_preserves_computation

            def transform(self, graph_module):
                return functools.reduce(make_reduce_fn(False), transformations)(graph_module)

            def reverse(self, graph_module):
                return functools.reduce(make_reduce_fn(True), transformations)(graph_module)

    return ComposeTransformation()
