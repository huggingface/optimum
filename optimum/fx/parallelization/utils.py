# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import importlib
import operator
from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Graph, Node

from .core import (
    HashableSlice,
    ParameterMeta,
    ParameterSlice,
)


def ensure_divisibility(numerator: int, denominator: int) -> None:
    if numerator % denominator != 0:
        raise RuntimeError(
            f"{numerator} is not divisible by {denominator}, check if the parallel dimension of weight parameters is divisible "
            "by parallelism level(world size of tensor parallel group)"
        )


def is_linear(node: Node) -> bool:
    if node.op != "call_module":
        return False
    mod = node.graph.owning_module
    return isinstance(mod.get_submodule(node.target), nn.Linear)


def is_embedding(node: Node) -> bool:
    if node.op != "call_module":
        return False
    mod = node.graph.owning_module
    return isinstance(mod.get_submodule(node.target), nn.Embedding)


def is_shape_consumer(node: Node) -> bool:
    if node.op == "call_method":
        return node.target in {"view", "reshape", "expand", "resize", "resize_"}
    elif node.op == "call_function":
        return node.target in {torch.reshape}
    return False


def is_transpose(node: Node) -> bool:
    if node.op == "call_method":
        return node.target in {"transpose", "transpose_"}
    elif node.op == "call_function":
        return node.target is torch.transpose
    return False


def is_permute(node: Node) -> bool:
    if node.op == "call_method":
        return node.target in {"permute"}
    elif node.op == "call_function":
        return node.target is torch.permute
    return False


def is_getitem(node: Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def is_output(node: Node) -> bool:
    return node.op == "output"


def is_shape_generator(node: Node) -> bool:
    return node.op == "call_method" and node.target == "size"


def stable_topological_sort(graph: Graph):
    def _args(n: torch.fx.Node) -> List[torch.fx.node.Argument]:
        args: List[torch.fx.node.Argument] = []
        torch.fx.map_arg((n.args, n.kwargs), args.append)
        return args

    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = set()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()
        waiting_for = [x for x in _args(node) if x not in ready]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    assert not waiting and len(ready) == len(graph.nodes)


def meta_init(init_fn):
    @wraps(init_fn)
    def wrapper(*args, **kwargs):
        kwargs["device"] = kwargs.pop("device", torch.device("meta"))
        return init_fn(*args, **kwargs)

    return wrapper


@wraps(nn.Linear.forward)
def meta_aware_linear_forward(*args, **kwargs):
    self = args[0]
    input = args[1]

    if self.weight.device != torch.device("meta"):
        return F.linear(input, self.weight, self.bias)

    orig_device = input.device
    input = input.to("meta")
    meta_output = F.linear(input, self.weight, self.bias)
    return torch.empty_like(meta_output, device=orig_device)


@wraps(nn.Embedding.forward)
def meta_aware_embedding_forward(*args, **kwargs):
    self = args[0]
    input = args[1]

    if self.weight.device != torch.device("meta"):
        return F.embedding(
            input=input,
            weight=self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    orig_device = input.device
    input = input.to("meta")
    meta_output = F.embedding(
        input=input,
        weight=self.weight,
        padding_idx=self.padding_idx,
        max_norm=self.max_norm,
        norm_type=self.norm_type,
        scale_grad_by_freq=self.scale_grad_by_freq,
        sparse=self.sparse,
    )
    return torch.empty_like(meta_output, device=orig_device)


class MetaAwareMethodsPatcher:
    """
    A patcher class which patches `__init__` and `forward` methods on modules which will be put on meta
    devices for memory efficiency purposes during initialization.

    Note that for `__init__` method, it can be unpatched once we have finished the initialization of the
    model, however, for `forward`, we need it to constantly being patched during the whole process in case
    recompile happens and torch dynamo needs meta-aware `forward` to be able to re-capture the graph.
    """

    methods_to_patch: Dict[str, Callable] = [
        ("torch.nn.Linear.__init__", meta_init(nn.Linear.__init__)),
        ("torch.nn.Embedding.__init__", meta_init(nn.Embedding.__init__)),
        ("torch.nn.Linear.forward", meta_aware_linear_forward),
        ("torch.nn.Embedding.forward", meta_aware_embedding_forward),
    ]

    def __init__(self) -> None:
        self.patching_specs = []
        for orig, patch_fn in self.methods_to_patch:
            module_qualified_name, attribute_name = orig.rsplit(".", maxsplit=1)
            try:
                module = importlib.import_module(module_qualified_name)
            except ModuleNotFoundError as e:
                module_qualified_name, module_attribute_name = module_qualified_name.rsplit(".", maxsplit=1)
                module = importlib.import_module(module_qualified_name)
                try:
                    module = getattr(module, module_attribute_name)
                except AttributeError:
                    raise e
            orig_fn = getattr(module, attribute_name)

            # Module, Attribute, Patchee, Patcher, Status
            self.patching_specs.append([module, attribute_name, orig_fn, patch_fn, False])

    def _patch(self, identifier: str):
        for spec in self.patching_specs:
            # already patched
            if spec[-1]:
                continue
            if identifier in spec[1]:
                setattr(spec[0], spec[1], spec[3])
                spec[-1] = True

    def _unpatch(self, identifier: str):
        for spec in self.patching_specs:
            # already patched
            if not spec[-1]:
                continue
            if identifier in spec[1]:
                setattr(spec[0], spec[1], spec[2])
                spec[-1] = False

    def patch_meta_init(
        self,
    ):
        self._patch("init")

    def patch_meta_forward(
        self,
    ):
        self._patch("forward")

    def unpatch_meta_init(
        self,
    ):
        self._unpatch("init")

    def unpatch_meta_forward(
        self,
    ):
        self._unpatch("forward")

    def __enter__(
        self,
    ):
        self.patch_meta_init()
        self.patch_meta_forward()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unpatch_meta_init()


def initialize_parameter_meta(model: nn.Module) -> None:
    parameter_ids = set()
    for name, tensor in model.named_parameters(remove_duplicate=False):
        key = id(tensor)
        if key not in parameter_ids:
            setattr(
                tensor,
                "meta",
                ParameterMeta(dim=0, mapping={HashableSlice(None, None, None): ParameterSlice(source=name)}),
            )
            parameter_ids.add(key)
        else:
            tensor.meta.is_tied = True


@torch.no_grad
def move_model_to_device(model: nn.Module, device: Union[torch.device, str]):
    """
    Move everything except tensors on meta devices on current device
    this function should be called before `intialize_parameter_meta`
    """
    for name, tensor in chain(model.named_parameters(), model.named_buffers()):
        if tensor.device == torch.device("meta"):
            continue
        splits = name.rsplit(".", maxsplit=1)
        if len(splits) == 1:
            parent_mod = model
            attr_name = splits[0]
        else:
            qualified_name = splits[0]
            parent_mod = model.get_submodule(qualified_name)
            attr_name = splits[1]
        new_tensor = tensor.to(device)
        if isinstance(tensor, nn.Parameter):
            new_tensor = nn.Parameter(new_tensor)
        setattr(parent_mod, attr_name, new_tensor)
