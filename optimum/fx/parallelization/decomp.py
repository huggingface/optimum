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
import contextlib
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import SymBool, SymFloat, SymInt
from torch._decomp import core_aten_decompositions
from torch._functorch._aot_autograd.functional_utils import from_fun, to_fun
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode, disable_functional_mode
from torch.fx import Graph, GraphModule, Interpreter, Proxy, traceback
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    _ProxyTensor,
    _SymNodeDict,
    decompose,
    disable_proxy_modes_tracing,
    fetch_object_proxy,
    fetch_sym_proxy,
    get_proxy_slot,
    track_tensor_tree,
)
from torch.fx.proxy import GraphAppendingTracer
from torch.utils.weak import WeakTensorKeyDictionary


def is_leaf_module(m):
    return (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn")) and not isinstance(
        m, torch.nn.Sequential
    )


@contextlib.contextmanager
def trace_decomp_origin():
    creat_node = Graph.create_node

    def create_node_(*args, **kwargs):
        node = creat_node(*args, **kwargs)
        node.meta["traced_from"] = traceback.get_current_meta()["from_node"]
        return node

    try:
        Graph.create_node = create_node_
        yield
    finally:
        Graph.create_node = creat_node


class DecompTracer(GraphAppendingTracer):
    """
    DecompTracer is a tracer class which works together with `DecompositionInterpreter`, it keeps track of tensors and their
    corresponding proxy objects during execution process. When invoked with `create_proxy`, it creates a node in the containing
    graph and associate the output tensor of the node with the created proxy.

    See https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/proxy_tensor.py for more details.
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.symnode_tracker = _SymNodeDict()


class DecompositionInterpreter(Interpreter):
    """
    DecompositionInterpreter takes the high-level graph module, run the iternal nodes following the topo order, and decompose
    high-level pytorch operators into core aten operators by utilizing torch dispatch infrastructure along the way.

    Notes:
        - Certain primitive layers(like `nn.Linear`, `nn.Embedding`, and activation layers) are preserved because we have specific
          heuristic based parallelization strategy for them so that we can conveniently replace them into their parallelized counterparts
          in the orignal graph module.

        - The traced graph is a low-level equivalent representation of the original graph module, and is only used for
          parallel axis propagation and analysis, the original graph module is still used for real execution.
    """

    def __init__(
        self, module: GraphModule, new_graph: Graph, decomposition_table=None, leaf_function_targets=None, **kwargs
    ):
        super().__init__(module, **kwargs)
        self.new_graph = new_graph
        self.tracer = DecompTracer(new_graph)

        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}

        self.leaf_function_targets = leaf_function_targets
        if self.leaf_function_targets is None:
            self.leaf_function_targets = []

        self.fun_mode = FunctionalTensorMode()
        self.mode = ProxyTorchDispatchMode(self.tracer, tracing_mode="real")

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        out = pytree.tree_map_only(FunctionalTensor, lambda x: from_fun(x), out)
        proxy = self.tracer.create_proxy("placeholder", target, args, kwargs)

        with disable_proxy_modes_tracing():
            track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)

        out = pytree.tree_map_only(torch.Tensor, lambda x: to_fun(x), out)
        return out

    def call_function(self, target, args, kwargs):
        if target in self.leaf_function_targets:
            args = pytree.tree_map_only(FunctionalTensor, lambda x: from_fun(x), args)
            kwargs = pytree.tree_map_only(FunctionalTensor, lambda x: from_fun(x), kwargs)

            with disable_proxy_modes_tracing(), disable_functional_mode():
                out = target(*args, **kwargs)

            args, kwargs = pytree.tree_map_only((torch.Tensor,), fetch_object_proxy(self.tracer), (args, kwargs))
            proxy_args, proxy_kwargs = pytree.tree_map_only(
                (SymInt, SymFloat, SymBool),
                fetch_sym_proxy(self.tracer),
                pytree.tree_map_only(_ProxyTensor, lambda e: e.proxy, (args, kwargs)),
            )
            proxy = self.tracer.create_proxy("call_function", target, proxy_args, proxy_kwargs)

            with disable_proxy_modes_tracing():
                track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)

            out = pytree.tree_map_only(torch.Tensor, lambda x: to_fun(x), out)
            return out

        return super().call_function(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if not is_leaf_module(submod):
            return super().call_module(target, args, kwargs)

        args = pytree.tree_map_only(FunctionalTensor, lambda x: from_fun(x), args)
        kwargs = pytree.tree_map_only(FunctionalTensor, lambda x: from_fun(x), kwargs)

        with disable_proxy_modes_tracing(), disable_functional_mode():
            out = submod(*args, **kwargs)

        args, kwargs = pytree.tree_map_only((torch.Tensor,), fetch_object_proxy(self.tracer), (args, kwargs))
        proxy_args, proxy_kwargs = pytree.tree_map_only(
            (SymInt, SymFloat, SymBool),
            fetch_sym_proxy(self.tracer),
            pytree.tree_map_only(_ProxyTensor, lambda e: e.proxy, (args, kwargs)),
        )
        proxy = self.tracer.create_proxy("call_module", target, proxy_args, proxy_kwargs)

        with disable_proxy_modes_tracing():
            track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)

        out = pytree.tree_map_only(torch.Tensor, lambda x: to_fun(x), out)
        return out

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        proxy = Proxy(self.new_graph.get_attr(target), self.tracer)
        with disable_proxy_modes_tracing():
            track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    def output(self, target, args, kwargs):
        args = pytree.tree_map_only(FunctionalTensor, lambda x: from_fun(x), args)
        kwargs = pytree.tree_map_only(FunctionalTensor, lambda x: from_fun(x), kwargs)
        out = super().output(target, args, kwargs)

        def unwrap(e):
            return get_proxy_slot(e, self.tracer, e, lambda x: x.proxy.node)

        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args, **kwargs):
        with self.fun_mode:
            args = pytree.tree_map_only(torch.Tensor, lambda x: to_fun(x), args)
            kwargs = pytree.tree_map_only(torch.Tensor, lambda x: to_fun(x), kwargs)
            with traceback.preserve_node_meta(), trace_decomp_origin(), decompose(self.decomposition_table), self.mode:
                return super().run(*args, **kwargs)


def decompose_and_functionalize(
    graph_module: GraphModule,
    decomposition_table: Dict[torch._ops.OperatorBase, Callable] = core_aten_decompositions(),
    leaf_function_targets: List[Callable] = [F.scaled_dot_product_attention, F.cross_entropy],
) -> Callable:
    """
    API to decompose and functionalize a high-level graph module.

    Args:
        graph_module (`GraphModule`):
            The high-level graph module to be decomposed and functionalized.
        decomposition_table (`Dict[torch._ops.OperatorBase, Callable]`, defaults to `core_aten_decompostions()`):
            The lookup table which maps high-level torch op to their equivalent low-level implementation.
        leaf_function_targets (`List[Callable]`, defaults to `[F.scaled_dot_product_attention]`):
            Functions which will not be traced through for convenience, `F.scaled_dot_product_attention` is
            treated as a leaf function by default so that we don't have to deal with all detailed version of
            sdpas in the traced graph.

    Returns:
        Callable: a wrapper which returns the traced low-level graph when called with concrete arguments.
    """
    new_graph = Graph(owning_module=graph_module)
    interp = DecompositionInterpreter(graph_module, new_graph, decomposition_table, leaf_function_targets)

    def wrapper(*args, **kwargs):
        interp.run(*args, **kwargs)
        return new_graph

    return wrapper
