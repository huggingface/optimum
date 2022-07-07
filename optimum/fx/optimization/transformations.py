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
from typing import TYPE_CHECKING, List

import torch


if TYPE_CHECKING:
    from torch.fx import GraphModule, Node


_ATTRIBUTES_DOCSTRING = r"""
Attributes:
    preserves_computation (`bool`, defaults to `False`):
        Whether the transformation preserves the graph computation or not. If `True`, the original and the
        transformed graph should produce the same outputs.
"""
_EXAMPLE_DOCSTRING = r"""
```python
>>> from transformers import BertModel
>>> from transformers.utils.fx import symbolic_trace
>>> from optimum.fx.optimization import {class_name}

>>> model = BertModel.from_pretrained("bert-base-uncased")
>>> traced = symbolic_trace(
>>>     model,
>>>     input_names=["input_ids", "attention_mask", "token_type_ids"],
>>> )
>>> transformation = {class_name}()
>>> transformed_model = transformation(traced)
```
"""
_REVERSIBLE_EXAMPLE_DOCSTRING = r"""
```python
>>> from transformers import BertModel
>>> from transformers.utils.fx import symbolic_trace
>>> from optimum.fx.optimization import {class_name}

>>> model = BertModel.from_pretrained("bert-base-uncased")
>>> traced = symbolic_trace(
>>>     model,
>>>     input_names=["input_ids", "attention_mask", "token_type_ids"],
>>> )
>>> transformation = {class_name}()
>>> transformed_model = transformation(traced)
>>> restored_model = transformation(transformed_model, reverse=True)
```
"""


def add_docstring(add_example=True):
    def wrapper(class_):
        example_docstring = _EXAMPLE_DOCSTRING
        if "ReversibleTransformation" in map(lambda cls: cls.__name__, class_.mro()):
            example_docstring = _REVERSIBLE_EXAMPLE_DOCSTRING
        new_doc = [f"{class_.__doc__}", f"{_ATTRIBUTES_DOCSTRING}"]
        if add_example:
            new_doc.append("Example:")
            new_doc.append(f"\t{example_docstring.format(class_name=class_.__name__)}")

        class_.__doc__ = "\n".join(new_doc)
        return class_

    return wrapper


@add_docstring(add_example=False)
class Transformation(ABC):
    """
    A torch.fx graph transformation.

    It  must implemement the [`~optimum.fx.optimization.ReversibleTransformation.transform`] method, and be used as a
    callable.
    """

    preserves_computation: bool = False

    @abstractmethod
    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.

        Returns:
            `torch.fx.GraphModule`:
                The transformed module.
        """
        raise NotImplementedError("The transform method needs to be implemented.")

    def __call__(self, graph_module: "GraphModule", lint_and_recompile: bool = True) -> "GraphModule":
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.
            lint_and_recompile (`bool`, defaults to `True`):
                Whether the transformed module should be linted and recompiled.
                This can be set to `False` when chaining transformations together to perform this operation only once.

        Returns:
            `torch.fx.GraphModule`:
                The transformed module.
        """
        graph_module = self.transform(graph_module)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module

    @property
    def signature(self):
        """
        Returns a hash that can be used to identify the transformation.
        """
        attributes_to_use_for_hashing = vars(self)
        attributes_to_use_for_hashing[""] = self.__class__
        hash_str = "_".join(f"k_{hash(v)}" for k, v in attributes_to_use_for_hashing.items())
        return hash(hash_str)

    def mark_as_transformed(self, node: "Node"):
        """
        Marks a node as transformed by this transformation.

        Args:
            node (`torch.fx.Node`):
                The node to mark as transformed.
        """
        node_transformations = getattr(node, "transformations", set())
        node_transformations.add(self.signature)
        node.transformations = node_transformations

    def transformed(self, node: "Node") -> bool:
        """
        Args:
            node (`torch.fx.Node`):
                The node to check.

        Returns:
            `bool`:
                Specifies whether the node was transformed by this transformation or not.
        """
        return self.signature in getattr(node, "transformations", set())

    def get_transformed_nodes(self, graph_module: "GraphModule") -> List["Node"]:
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The graph_module to get the nodes from.

        Returns:
            `List[torch.fx.Node]`:
                Gives the list of nodes that were transformed by the transformation.
        """

        return [node for node in graph_module.graph.nodes if self.transformed(node)]


@add_docstring(add_example=False)
class ReversibleTransformation(Transformation):
    """
    A torch.fx graph transformation that is reversible.

    It must implemement the [`~optimum.fx.optimization.ReversibleTransformation.transform`] and
    [`~optimum.fx.optimization.ReversibleTransformation.reverse`] methods, and be used as a callable.
    """

    @abstractmethod
    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.

        Returns:
            `torch.fx.GraphModule`:
                The reverse transformed module.
        """
        raise NotImplementedError("The reverse transform method needs to be implemented.")

    def __call__(
        self, graph_module: "GraphModule", lint_and_recompile: bool = True, reverse: bool = False
    ) -> "GraphModule":
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.
            lint_and_recompile (`bool`, defaults to `True`):
                Whether the transformed module should be linted and recompiled.
                This can be set to `False` when chaining transformations together to perform this operation only once.
            reverse (`bool`, defaults to `False`):
                If `True`, the reverse transformation is performed.

        Returns:
            `torch.fx.GraphModule`:
                The transformed module.

        """
        func = self.transform if not reverse else self.reverse
        graph_module = func(graph_module)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module

    def mark_as_restored(self, node: "Node"):
        """
        Marks a node as restored back to its original state.

        Args:
            node (`torch.fx.Node`):
                The node to mark as restored.
        """
        node_transformations = getattr(node, "transformations", set())
        if self.signature not in node_transformations:
            raise ValueError("The node was not transformed by this transformation.")
        node_transformations.remove(self.signature)


@add_docstring()
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


@add_docstring()
class ChangeTrueDivToMulByInverse(ReversibleTransformation):
    """
    Transformation that changes truediv nodes to multiplication by the inverse nodes when the denominator is static.
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


def compose(*args: Transformation, inplace: bool = True) -> Transformation:
    """
    Composes a list of transformations together.

    Args:
        args ([`~optimum.fx.optimization.Transformation`]):
            The transformations to compose together.
        inplace (`bool`, defaults to `True`):
            Whether the resulting transformation should be inplace, or create a new graph module.

    Returns:
        The composition transformation object.

    Example:

    ```python
    >>> from transformers import BertModel
    >>> from transformers.utils.fx import symbolic_trace
    >>> from optimum.fx.optimization import ChangeTrueDivToMulByInverse, MergeLinears, compose

    >>> model = BertModel.from_pretrained("bert-base-uncased")
    >>> traced = symbolic_trace(
    >>>     model,
    >>>     input_names=["input_ids", "attention_mask", "token_type_ids"],
    >>> )
    >>> composition = compose(ChangeTrueDivToMulByInverse(), MergeLinears())
    >>> transformed_model = composition(traced)
    ```
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

            _composition = functools.reduce(reduce_fn, transformations)

            def transform(self, graph_module):
                return ComposeTransformation._composition(graph_module)

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

            _composition = functools.reduce(make_reduce_fn(False), transformations)
            _reverse_composition = functools.reduce(make_reduce_fn(True), reversed(transformations))

            def transform(self, graph_module):
                return ComposeTransformation._composition(graph_module)

            def reverse(self, graph_module):
                return ComposeTransformation._reverse_composition(graph_module)

    return ComposeTransformation()
