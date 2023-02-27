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
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper


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
...     model,
...     input_names=["input_ids", "attention_mask", "token_type_ids"],
... )
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
...     model,
...     input_names=["input_ids", "attention_mask", "token_type_ids"],
... )
>>> transformation = {class_name}()
>>> transformed_model = transformation(traced)
>>> restored_model = transformation(transformed_model, reverse=True)
```
"""


def add_docstring(add_example=True):
    def wrapper(class_):
        example_docstring = _EXAMPLE_DOCSTRING
        if "ReversibleTransformation" in (cls.__name__ for cls in class_.mro()):
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

    It  must implement the [`~optimum.fx.optimization.ReversibleTransformation.transform`] method, and be used as a
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
        hash_str = "_".join(f"{k}_{hash(v)}" for k, v in attributes_to_use_for_hashing.items())
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

    It must implement the [`~optimum.fx.optimization.ReversibleTransformation.transform`] and
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
        if linear.bias is not None:
            return linear.bias
        return torch.zeros(linear.out_features, dtype=linear.weight.dtype).to(linear.weight.device)

    @staticmethod
    def _get_linear_module_name(linear_node):
        return linear_node.target.split(".")[-1]

    @staticmethod
    def _linear_node_to_module_and_attribute_name(graph_module, linear_node_target):
        names = linear_node_target.split(".")
        mod = graph_module
        if len(names) > 1:
            for name in names[:-1]:
                mod = getattr(mod, name)
        return mod, names[-1]

    def _merge_linears(
        self,
        graph_module: "GraphModule",
        input_node: "Node",
        linear_nodes: List["Node"],
        linears: List[torch.nn.Linear],
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
        merged_linear = torch.nn.Linear(
            in_features,
            total_out_features,
            bias=use_bias,
        )

        dtype = linears[0].weight.dtype
        device = linears[0].weight.device

        with torch.no_grad():
            new_weight = torch.cat([linear.weight for linear in linears], dim=0).to(dtype=dtype, device=device)
            merged_linear.weight = torch.nn.Parameter(new_weight)
            if use_bias:
                new_bias = torch.cat([MergeLinears._get_bias(linear) for linear in linears], dim=0).to(
                    dtype=dtype, device=device
                )
                merged_linear.bias = torch.nn.Parameter(new_bias)

        linear_module_names = [MergeLinears._get_linear_module_name(node) for node in linear_nodes]
        merged_linear_name = "-".join(linear_module_names + ["merged"])
        fully_qualified_parent_name = linear_nodes[0].target.rsplit(".", maxsplit=1)[0]
        parent_module = graph_module.get_submodule(fully_qualified_parent_name)
        parent_module.add_module(merged_linear_name, merged_linear)
        # for name in linear_module_names:
        for linear_node in linear_nodes:
            mod, name = MergeLinears._linear_node_to_module_and_attribute_name(graph_module, linear_node.target)
            delattr(mod, name)

        graph = graph_module.graph
        with graph.inserting_before(linear_nodes[0]):
            fully_qualified_merged_linear_name = ".".join([fully_qualified_parent_name, merged_linear_name])
            merged_linear_node = graph.call_module(fully_qualified_merged_linear_name, args=(input_node,))
            self.mark_as_transformed(merged_linear_node)
            merged_linear_node.linear_node_targets = [n.target for n in linear_nodes]

        accum_out_features = list(itertools.accumulate([0] + out_features))
        for idx, node in enumerate(linear_nodes):
            node.op = "call_function"
            node.target = operator.getitem
            slice_to_get = slice(accum_out_features[idx], accum_out_features[idx + 1])
            node.args = (merged_linear_node, (Ellipsis, slice_to_get))

    @staticmethod
    def _unmerge_linears(graph_module: "GraphModule", merged_linear_node: "Node", merged_linear: torch.nn.Linear):
        # The linear node targets and the output nodes need to be in the same order.
        # merge_linear_name gives the order in which the weights were concatenated, and we use the slice start index to
        # sort the output nodes since the start index tells when a weight was concatenated.
        linear_node_targets = merged_linear_node.linear_node_targets
        output_nodes = sorted(merged_linear_node.users, key=lambda node: node.args[1][1].start)

        in_features = merged_linear.in_features
        out_features = []
        for node in output_nodes:
            slice_to_get = node.args[1][1]
            out_features.append(slice_to_get.stop - slice_to_get.start)

        linears = [
            torch.nn.Linear(
                in_features,
                out_feat,
                bias=hasattr(merged_linear, "bias"),
                device=merged_linear.weight.device,
                dtype=merged_linear.weight.dtype,
            )
            for out_feat in out_features
        ]

        # fully_qualified_parent_name = merged_linear_node.target.rsplit(".", maxsplit=1)[0]
        # parent_module = graph_module.get_submodule(fully_qualified_parent_name)
        # parent_module_name = merged_linear_node.target.rsplit(".", maxsplit=1)[0]
        for target, node, linear in zip(linear_node_targets, output_nodes, linears):
            with torch.no_grad():
                slice_to_get = node.args[1][1]
                linear.weight = torch.nn.Parameter(merged_linear.weight[slice_to_get.start : slice_to_get.stop])
                if hasattr(merged_linear, "bias"):
                    linear.bias = torch.nn.Parameter(merged_linear.bias[slice_to_get.start : slice_to_get.stop])
            parent_module, name = MergeLinears._linear_node_to_module_and_attribute_name(graph_module, target)
            parent_module.add_module(name, linear)
            node.op = "call_module"
            node.target = target
            node.args = (merged_linear_node.args[0],)

        parent_module, merged_linear_name = MergeLinears._linear_node_to_module_and_attribute_name(
            graph_module, merged_linear_node.target
        )
        delattr(parent_module, merged_linear_name)
        graph_module.graph.erase_node(merged_linear_node)

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        candidates = collections.defaultdict(list)
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                mod = graph_module.get_submodule(node.target)
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
        for node in self.get_transformed_nodes(graph_module):
            self._unmerge_linears(graph_module, node, graph_module.get_submodule(node.target))
        return graph_module


@add_docstring()
class FuseBiasInLinear(ReversibleTransformation):
    """
    Transformation that fuses the bias to the weight in torch.nn.Linear.
    """

    preserves_computation = True

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        torch_ones = _gen_constructor_wrapper(torch.ones)[0]

        def insert_concat(linear_input):
            shape = linear_input.shape[:-1] + (1,)
            return torch.cat([linear_input, torch_ones(shape, device=linear_input.device)], dim=-1)

        tracer = torch.fx.proxy.GraphAppendingTracer(graph_module.graph)
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                module = graph_module.get_submodule(node.target)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    with graph_module.graph.inserting_before(node):
                        n = node.args[0]
                        node.nodes_to_ignore = set()
                        while n is not node:
                            node.nodes_to_ignore.add(n)
                            n = n.next
                        linear_input_proxy = torch.fx.Proxy(node.args[0], tracer)
                        output_proxy = insert_concat(linear_input_proxy)
                        node.start_node = linear_input_proxy.node
                        node.end_node = output_proxy.node
                        node.args = (output_proxy.node,)
                        self.mark_as_transformed(node)
                    new_weight = torch.nn.Parameter(torch.cat([module.weight, module.bias[:, None]], dim=1))
                    module.weight = new_weight
                    module.bias = None
        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in self.get_transformed_nodes(graph_module):
            node.args = (node.start_node,)
            n = node.end_node
            while n is not node.start_node:
                if n not in node.nodes_to_ignore:
                    graph_module.graph.erase_node(n)
                n = n.prev
            self.mark_as_restored(node)
            module = graph_module.get_submodule(node.target)
            new_weight = torch.nn.Parameter(module.weight[:, :-1])
            new_bias = torch.nn.Parameter(module.weight[:, -1].squeeze())
            module.weight = new_weight
            module.bias = new_bias
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
                    self.mark_as_transformed(node)

        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in self.get_transformed_nodes(graph_module):
            node.target = operator.truediv
            x, y = node.args
            node.args = (x, 1 / y)
            self.mark_as_restored(node)

        return graph_module


@add_end_docstrings(_ATTRIBUTES_DOCSTRING)
class FuseBatchNorm2dInConv2d(Transformation):
    """
    Transformation that fuses `nn.BatchNorm2d` following `nn.Conv2d` into a single `nn.Conv2d`.
    The fusion will be done only if the convolution has the batch normalization as sole following node.

    For example, fusion will not be done in the case
    ```
         Conv2d
         /   \\
        /     \\
    ReLU   BatchNorm2d
    ```

    Example:
    ```python
    >>> from transformers.utils.fx import symbolic_trace
    >>> from transformers import AutoModelForImageClassification

    >>> from optimum.fx.optimization import FuseBatchNorm2dInConv2d

    >>> model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    >>> model.eval()  # doctest: +IGNORE_RESULT

    >>> traced_model = symbolic_trace(
    ...     model,
    ...     input_names=["pixel_values"],
    ...     disable_check=True
    ... )

    >>> transformation = FuseBatchNorm2dInConv2d()
    >>> transformed_model = transformation(traced_model)
    ```
    """

    preserves_computation = True

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and node.args[0].op == "call_module":
                if (
                    type(graph_module.get_submodule(node.target)) is torch.nn.BatchNorm2d
                    and type(graph_module.get_submodule(node.args[0].target)) is torch.nn.Conv2d
                ):
                    if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                        continue

                    fused_conv = self.fuse(
                        conv2d=graph_module.get_submodule(node.args[0].target),
                        bn2d=graph_module.get_submodule(node.target),
                    )

                    # replace the old nn.Conv2d by the fused one
                    parent_name, _, name = node.args[0].target.rpartition(".")
                    parent_module = graph_module.get_submodule(parent_name)
                    setattr(parent_module, name, fused_conv)

                    # delete batchnorm from the modules
                    parent_name, _, name = node.target.rpartition(".")
                    parent_module = graph_module.get_submodule(parent_name)
                    delattr(parent_module, name)

                    node.replace_all_uses_with(node.args[0])
                    graph_module.graph.erase_node(node)
        return graph_module

    def fuse(self, conv2d: torch.nn.Conv2d, bn2d: torch.nn.BatchNorm2d):
        # handle the case where there is no bias in the conv or the batchnorm has no learnable parameters
        conv_b = conv2d.bias if conv2d.bias is not None else torch.zeros_like(bn2d.running_mean)
        bn_w = bn2d.weight if bn2d.weight is not None else torch.ones_like(bn2d.running_mean)
        bn_b = bn2d.bias if bn2d.bias is not None else torch.ones_like(bn2d.running_mean)

        bn_var_rsqrt = torch.rsqrt(bn2d.running_var + bn2d.eps)

        conv2d.weight = torch.nn.Parameter(
            conv2d.weight * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv2d.weight.shape) - 1))
        )

        conv2d.bias = torch.nn.Parameter(conv_b - bn2d.running_mean * bn_var_rsqrt * bn_w + bn_b)

        return conv2d


@add_end_docstrings(_ATTRIBUTES_DOCSTRING)
class FuseBatchNorm1dInLinear(Transformation):
    """
    Transformation that fuses `nn.BatchNorm1d` following or preceding `nn.Linear` into a single `nn.Linear`.
    The fusion will be done only if the linear layer has the batch normalization as sole following node, or the batch normalization
    has the linear layer as sole following node.

    For example, fusion will not be done in the case
    ```
         Linear
         /   \\
        /     \\
    ReLU   BatchNorm1d
    ```

    Example:
    ```python
    >>> from transformers.utils.fx import symbolic_trace
    >>> from transformers import AutoModel

    >>> from optimum.fx.optimization import FuseBatchNorm1dInLinear

    >>> model = AutoModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
    >>> model.eval()  # doctest: +IGNORE_RESULT

    >>> traced_model = symbolic_trace(
    ...     model,
    ...     input_names=["input_ids", "attention_mask", "pixel_values"],
    ...     disable_check=True
    ... )

    >>> transformation = FuseBatchNorm1dInLinear()
    >>> transformed_model = transformation(traced_model)
    ```
    """

    preserves_computation = True

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and node.args[0].op == "call_module":
                if (
                    type(graph_module.get_submodule(node.target)) is torch.nn.BatchNorm1d
                    and type(graph_module.get_submodule(node.args[0].target)) is torch.nn.Linear
                ):
                    # handle the case torch.nn.Linear --> torch.nn.BatchNorm1d

                    if len(node.args[0].users) > 1:  # Output of linear is used by other nodes
                        continue

                    candidate_linear = graph_module.get_submodule(node.args[0].target)
                    candidate_batchnorm1d = graph_module.get_submodule(node.target)

                    # will fuse only if the linear output features is equal to the batchnorm num features, this is the case with 2D tensors
                    # the case where the linear input is (N, C, L_in), output is (N, C, L_out) and C = L_out is NOT handled as can not be fused
                    if candidate_linear.weight.shape[0] == candidate_batchnorm1d.weight.shape[0]:
                        fused_linear = self.fuse(
                            linear=candidate_linear, bn1d=candidate_batchnorm1d, bn1d_before=False
                        )

                        # replace the old nn.Linear by the fused one
                        parent_name, _, name = node.args[0].target.rpartition(".")
                        parent_module = graph_module.get_submodule(parent_name)
                        setattr(parent_module, name, fused_linear)

                        # delete batchnorm from the modules
                        parent_name, _, name = node.target.rpartition(".")
                        parent_module = graph_module.get_submodule(parent_name)
                        delattr(parent_module, name)

                        node.replace_all_uses_with(node.args[0])

                        graph_module.graph.erase_node(node)  # delete BatchNorm1d
                elif (
                    type(graph_module.get_submodule(node.target)) is torch.nn.Linear
                    and type(graph_module.get_submodule(node.args[0].target)) is torch.nn.BatchNorm1d
                ):
                    # handle the case torch.nn.BatchNorm1d --> torch.nn.Linear
                    if len(node.args[0].users) > 1:  # Output of batchnorm is used by other nodes
                        continue

                    candidate_linear = graph_module.get_submodule(node.target)
                    candidate_batchnorm1d = graph_module.get_submodule(node.args[0].target)

                    # will fuse only if the linear input features is equal to the batchnorm num features, this is the case with 2D tensors
                    # the case where the linear input is (N, C, L_in) and C = L_in is NOT handled as can not be fused
                    if candidate_batchnorm1d.weight.shape[0] == candidate_linear.weight.shape[1]:
                        fused_linear = self.fuse(linear=candidate_linear, bn1d=candidate_batchnorm1d, bn1d_before=True)

                        # replace the old nn.Linear by the fused one
                        parent_name, _, name = node.target.rpartition(".")
                        parent_module = graph_module.get_submodule(parent_name)
                        setattr(parent_module, name, fused_linear)

                        # delete batchnorm from the modules
                        parent_name, _, name = node.args[0].target.rpartition(".")
                        parent_module = graph_module.get_submodule(parent_name)
                        delattr(parent_module, name)

                        batchnorm_node = node.args[0]
                        node.args[0].replace_all_uses_with(node.args[0].args[0])

                        graph_module.graph.erase_node(batchnorm_node)  # delete BatchNorm1d
        return graph_module

    def fuse(self, linear: torch.nn.Linear, bn1d: torch.nn.BatchNorm1d, bn1d_before: bool):
        # handle the case where there is no bias in the conv or the batchnorm has no learnable parameters
        linear_b = linear.bias if linear.bias is not None else torch.zeros_like(bn1d.running_mean)
        bn_w = bn1d.weight if bn1d.weight is not None else torch.ones_like(bn1d.running_mean)
        bn_b = bn1d.bias if bn1d.bias is not None else torch.ones_like(bn1d.running_mean)

        bn_var_rsqrt = torch.rsqrt(bn1d.running_var + bn1d.eps)

        if bn1d_before:
            linear.bias = torch.nn.Parameter(
                linear.weight @ (-bn_w * bn1d.running_mean * bn_var_rsqrt + bn_b) + linear_b
            )
            linear.weight = torch.nn.Parameter(linear.weight * (bn_w * bn_var_rsqrt)[None, :])
        else:
            linear.bias = torch.nn.Parameter((linear_b - bn1d.running_mean) * bn_var_rsqrt * bn_w + bn_b)
            linear.weight = torch.nn.Parameter(linear.weight * (bn_w * bn_var_rsqrt)[:, None])

        return linear


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
            if hasattr(n1, "transformations"):
                n2.transformations = n1.transformations
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
        return self.transform(graph_module)


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
    ...     model,
    ...     input_names=["input_ids", "attention_mask", "token_type_ids"],
    ... )
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
