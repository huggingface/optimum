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
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node

from .decomp import decompose_and_functionalize
from .op_registry import REGISTRY, FallbackParallelAxisPropagateHandler
from .utils import (
    ensure_divisibility,
    is_embedding,
    is_linear,
    is_shape_consumer,
    stable_topological_sort,
)


if TYPE_CHECKING:
    from .core import Config, ParallelExecutionCtx, ParameterMeta


class PassBase(ABC):
    """
    Base class for parallelization targeted passes.
    """

    need_rerun_when_recompile: bool = True

    @classmethod
    def signature(cls) -> str:
        return cls.__name__

    @abstractmethod
    def run(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> GraphModule:
        """
        Args:
            graph_module (`GraphModule`):
                graph module before processing.
            ctx (`ParallelExecutionCtx`):
                dynamic execution context which gathers and preserves information along processing.
            config (`Config`):
                static config to include instructions which persists the whole process.

        Returns:
            GraphModule: graph module after processed by the current pass.
        """
        raise NotImplementedError

    def __call__(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> GraphModule:
        # skip running when recompilation happens
        if not self.need_rerun_when_recompile and ctx.compile_times > 0:
            return graph_module

        graph_module = self.run(graph_module, ctx=ctx, config=config)
        if config.lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module


class AnalyzeBase(PassBase):
    """
    Base class for passes which only runs for analytical purposes and preserve graph structure
    during processing. Analytical passes are often prerequisite passes which provide information
    for passes later on to actually change the graph.

    Passes inheriting from `AnalyzeBase` places the class signature as a meta key in `node.meta`,
    which is a dict storing meta information related with a fx Node, such as the shape and dtype of
    output. Look-up APIs are exposed as classmethod so that passes using them won't need to create
    concrete instances.
    """

    @classmethod
    def meta_key(cls) -> str:
        # place class-wise unique meta_key in `meta` to prevent duplicate fields
        return cls.signature()

    @classmethod
    def get_stored_field_info(cls, node: Node, field: Any, must_have: bool = False) -> Any:
        if not cls.already_executed_per_node(node):
            if not must_have:
                return None
            else:
                raise RuntimeError(
                    f"Can't find information related with {cls.__name__} in the current node `{node}` "
                    f"make sure {cls.__name__} has run and marked it"
                )

        info: Dict[Any, Any] = node.meta[cls.meta_key()]
        if field not in info:
            if must_have:
                raise KeyError(f"Invalid query field {field} for {cls.__name__}, valid fields are {list(info.keys())}")
            return None

        return info[field]

    @classmethod
    def already_executed_per_node(cls, node: Node) -> bool:
        return cls.meta_key() in node.meta

    def place_marker_per_node(self, node: Node, info: Dict[Any, Any]) -> None:
        if self.already_executed_per_node(node):
            raise RuntimeError(
                f"Node {node} has already been marked by the current pass, check if "
                "the current pass has already been executed in the pipeline"
            )

        node.meta[self.meta_key()] = info

    def clear_marker_per_node(self, node: Node) -> None:
        key = self.meta_key()
        if key in node.meta:
            node.meta.pop(key)

    def clean_all(self, graph_module: GraphModule) -> None:
        g: Graph = graph_module.graph
        for node in g.nodes:
            self.clear_marker_per_node(node)


class ParallelAxisSolverPass(AnalyzeBase):
    """
    A pass which tries to automatically identify parallel layers in the graph. There are three steps
    involved to find a possible parallel solution given the traced graph module and process group.

        - Decompostion & Functionalization
            The vanilla graph traced by dynamo frontend is a high-level graph which contains high-level
            pytorch ops, and there could be thousands of them, which makes graph analysis hard in order
            to cover all cases. So we decompose the high-level graph into low-level graph which only
            conrtains core aten ops, which is a much smaller set. And functionalization is also needed
            to remove inplace ops in the graph so that we get `aten.Add` instead of `aten.Add_` in the
            graph, which furthur reduces the op set we need to consider.

        - Parallel Axis Propagation
            We need to write parallel axis propagation rules for aten ops in the decomposed and functionalized
            graph, note that we don't need to cover every possible parallelization strategy because in general
            only certain ops(usually involves computation) can be parallelized in transformer models. And we just
            need to write rules for a subset of core aten op set in order to support most of the transformer models.

        - Backtracking Search
            After we have defined parallel axis propagation rules for each op in the graph, we do a brute force
            backtracking search to try to find a possible solution which respects the propagation rule of every
            op in the graph.


        Note that there are several practical concerns

            - Time Complexity. Although brute force backtracking introduces an exponential time complexity, we reduces
                the search space by injecting human heuristics. First, we only consider parallelization on the head dimension
                (for tensor parallel) or the sequence dimension(to support sequence parallel), then at any time the tensor is
                parallelized on at most one dimension. Second, we only allow axis switch around certain layers(like `nn.Linear`
                or `nn.Embedding), and all other ops fall into their places by the parallel axis of their input and rules we write.

            - Optimal Solution. Note that since we return the first solution we find, then it might not be optimal in terms of
                memory consumption and communication overhead. But again we can adjust the order of search and try parallelize
                as much as we can first before fall back to non-parallelized search paths. And we don't pay too much attention
                on calculating communication overhead because in practice they are bounded under the constraint that only certain
                layers are allowed to communicate.

    Our goal is not to solve an optimization problem which tries to give a best solution of parallelizing any model under memory/hardware
    constraints, but rather a cheap solution which relieves you from writing boilerplate code for parallelizing layers of different models.
    """

    def trace_back(self, graph_module: GraphModule, decomp_graph: Graph) -> None:
        node_map = {node.name: node for node in graph_module.graph.nodes}

        for node in decomp_graph.nodes:
            if "traced_from" in node.meta:
                node_name, _ = node.meta["traced_from"][0]
                assert node_name in node_map, f"un-recognized node origin {node_name} not in graph being traced"
                orig_node = node_map[node_name]
                self.clear_marker_per_node(orig_node)
                self.place_marker_per_node(
                    orig_node, {"parallel_axis": self.get_stored_field_info(node, field="parallel_axis")}
                )

    def run(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> GraphModule:
        graph: Graph = decompose_and_functionalize(graph_module)(*ctx.example_inputs)
        stable_topological_sort(graph)

        nodes = list(graph.nodes)

        def search(idx: int):
            if idx == len(nodes):
                return True

            node = nodes[idx]
            if node.op == "call_function" and REGISTRY.is_supported(node.target):
                prop_cls = REGISTRY.mapping[node.target]
            else:
                prop_cls = FallbackParallelAxisPropagateHandler

            prop = prop_cls(node, self.meta_key(), config)
            axis_candidates = prop.propagate()
            for axis in axis_candidates:
                self.place_marker_per_node(node, {"parallel_axis": axis})
                if search(idx + 1):
                    return True
                self.clear_marker_per_node(node)

            return False

        if not search(0):
            raise RuntimeError("Failed to find a solution to automatically parallelize ops in graph in greedy way.")

        self.trace_back(graph_module, graph)
        return graph_module


class ParallelLayerAnnotatePass(AnalyzeBase):
    """
    This pass annotates layers which have different parallel axis(requires communication inside the layer) in their
    input and output tensors. Since heuristics applied during the searching process respect traditional classical ways of
    parallelizing layers(like Megatron-style `ColumnLinear` or `RowLinear`), we are guaranteed to match a valid replacement
    annotation according to parallelization strategy of input and output tensors.
    """

    def run(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> GraphModule:
        for node in graph_module.graph.nodes:
            if is_linear(node):
                axis_before = ParallelAxisSolverPass.get_stored_field_info(node.args[0], "parallel_axis")
                axis_after = ParallelAxisSolverPass.get_stored_field_info(node, "parallel_axis")
                info = {}
                if axis_before is None:
                    info["axis"] = "column"
                    info["gather_output"] = True if axis_after is None else False
                elif axis_before == 1:
                    assert (
                        config.enable_sequence_parallel
                    ), "illegal parallel axis for sequence parallelism deactivated setting"
                    info["axis"] = "column"
                    info["sequence_parallel"] = True
                    info["gather_output"] = True if axis_after is None else False
                elif axis_before == 2:
                    info["axis"] = "row"
                    info["input_is_parallel"] = True
                    if axis_after == 1:
                        assert (
                            config.enable_sequence_parallel
                        ), "illegal parallel axis for sequence parallelism deactivated setting"
                        info["sequence_parallel"] = True
                    else:
                        info["sequence_parallel"] = False
                self.place_marker_per_node(node, info)

            elif is_embedding(node):
                axis_before = ParallelAxisSolverPass.get_stored_field_info(node.args[0], "parallel_axis")
                axis_after = ParallelAxisSolverPass.get_stored_field_info(node, "parallel_axis")
                assert axis_before is None and axis_after in [1, None]
                info = {"axis": "vocab"}
                if axis_after == 1:
                    assert (
                        config.enable_sequence_parallel
                    ), "illegal parallel axis for sequence parallelism deactivated setting"
                    info["sequence_parallel"] = True
                else:
                    info["sequence_parallel"] = False
                self.place_marker_per_node(node, info)

        return graph_module


class ParallelLayerReplacePass(PassBase):
    """
    A pass which modifies graph according to information provided by previous analytical passes, in general it does two things for now:
        1. replaces linears and embedding layers with their parallel counterparts.
        2. modifies hard-coded arguments like the number of attention heads in the graph by dividing it by parallelism level.
    """

    @staticmethod
    def propagate_meta(
        ctx: "ParallelExecutionCtx",
        mod: Union[nn.Linear, nn.Embedding],
        new_mod: Union[nn.Linear, nn.Embedding],
        axis: str,
    ) -> None:
        world_size, tp_rank = dist.get_world_size(ctx.tp_group), dist.get_rank(ctx.tp_group)

        def get_current_slice(shape: Tuple[int], axis: int = 0) -> slice:
            ensure_divisibility(shape[axis], world_size)
            return slice(shape[axis] // world_size * tp_rank, shape[axis] // world_size * (tp_rank + 1))

        # modify meta information
        weight_meta: "ParameterMeta" = copy.deepcopy(getattr(mod.weight, "meta"))
        weight_meta.need_initialize = True
        weight_meta.is_parallel = True
        weight_meta.dim = 0 if axis in {"column", "vocab"} else 1
        for _, Slice in weight_meta.mapping.items():
            Slice.index = get_current_slice(Slice.shape, weight_meta.dim)
        setattr(new_mod.weight, "meta", weight_meta)

        if hasattr(new_mod, "bias") and new_mod.bias is not None:
            bias_meta: "ParameterMeta" = copy.deepcopy(getattr(mod.bias, "meta"))
            bias_meta.need_initialize = True
            bias_meta.init_fn = torch.zero_
            if weight_meta.dim == 0:
                bias_meta.dim = 0
                bias_meta.is_parallel = True
                for _, Slice in bias_meta.mapping.items():
                    Slice.index = get_current_slice(Slice.shape, 0)
            setattr(new_mod.bias, "meta", bias_meta)

    def handle_linear(self, node: Node, ctx: "ParallelExecutionCtx") -> None:
        graph_module = node.graph.owning_module
        axis = ParallelLayerAnnotatePass.get_stored_field_info(node, field="axis")
        if axis is None:
            return

        assert axis in {"column", "row"}
        prefix_and_field = node.target.rsplit(".", maxsplit=1)
        if len(prefix_and_field) == 2:
            parent_mod = graph_module.get_submodule(prefix_and_field[0])
            field = prefix_and_field[1]
        else:
            parent_mod = graph_module
            field = node.target

        mod: nn.Linear = graph_module.get_submodule(node.target)
        key, layer_cache, backend = node.target, ctx.parallel_layer_cache, ctx.backend
        if key in layer_cache:
            new_mod = layer_cache[key]
        else:
            assert ctx.compile_times == 0, "illegal path for recompilation"
            if axis == "column":
                gather_output = ParallelLayerAnnotatePass.get_stored_field_info(
                    node, field="gather_output", must_have=True
                )
                # TODO: enable sequence parallel
                new_mod = backend.create_column_parallel_linear(mod, ctx, False, gather_output)
            else:
                input_is_parallel = ParallelLayerAnnotatePass.get_stored_field_info(
                    node, field="input_is_parallel", must_have=True
                )
                # TODO: enable sequence parallel
                new_mod = backend.create_row_parallel_linear(mod, ctx, False, input_is_parallel)
            self.propagate_meta(ctx, mod, new_mod, axis)
            layer_cache[key] = new_mod
        setattr(parent_mod, field, new_mod)

    def handle_embedding(self, node: Node, ctx: "ParallelExecutionCtx") -> None:
        graph_module = node.graph.owning_module
        axis = ParallelLayerAnnotatePass.get_stored_field_info(node, field="axis")
        if axis is None:
            return

        assert axis in {"vocab"}, "Only support parallelization on vocab dim for now."
        prefix_and_field = node.target.rsplit(".", maxsplit=1)
        if len(prefix_and_field) == 2:
            parent_mod = graph_module.get_submodule(prefix_and_field[0])
            field = prefix_and_field[1]
        else:
            parent_mod = graph_module
            field = node.target

        mod: nn.Embedding = graph_module.get_submodule(node.target)
        key, layer_cache, backend = node.target, ctx.parallel_layer_cache, ctx.backend
        if key in layer_cache:
            new_mod = layer_cache[key]
        else:
            assert ctx.compile_times == 0, "illegal path for recompilation"
            # TODO: enable sequence parallel
            new_mod = backend.create_parallel_embedding(mod, ctx, False)
            self.propagate_meta(ctx, mod, new_mod, axis)
            layer_cache[key] = new_mod
        setattr(parent_mod, field, new_mod)

    def handle_hard_coded_axis_param(self, node: Node, ctx: "ParallelExecutionCtx") -> None:
        def extract_shape_from_node(node: Node) -> List[Any]:
            if "size" in node.kwargs:
                return list(node.kwargs["size"])
            elif "shape" in node.kwargs:
                return list(node.kwargs["shape"])
            elif isinstance(node.args[1], tuple):
                return list(node.args[1])
            else:
                return list(node.args[1:])

        def update(node: Node, new_shape: List[Any], parallel_axis: int):
            if "size" in node.kwargs:
                node.update_kwarg("size", tuple(new_shape))
            elif "shape" in node.kwargs:
                node.update_kwarg("shape", tuple(new_shape))
            elif isinstance(node.args[1], tuple):
                node.update_arg(1, tuple(new_shape))
            else:
                node.update_arg(parallel_axis + 1, shape[parallel_axis])

        parallel_axis = ParallelAxisSolverPass.get_stored_field_info(node, field="parallel_axis")
        if parallel_axis is None:
            return

        shape = extract_shape_from_node(node)
        assert parallel_axis < len(shape)
        if not isinstance(shape[parallel_axis], int) or shape[parallel_axis] == -1:
            return
        world_size = ctx.tp_group.size()
        assert shape[parallel_axis] % world_size == 0
        shape[parallel_axis] = shape[parallel_axis] // world_size
        update(node, shape, parallel_axis)

    def run(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> GraphModule:
        for node in graph_module.graph.nodes:
            if is_linear(node):
                self.handle_linear(node, ctx)
            elif is_embedding(node):
                self.handle_embedding(node, ctx)
            # correct the attention head num in parallel setting
            elif is_shape_consumer(node):
                self.handle_hard_coded_axis_param(node, ctx)
        return graph_module


class PassPipeline:
    """
    `PassPipeline` ensembles a list of passes and execute them one by one as provided in the list,
    it can be iterated and appended after initialization for flexibility.
    """

    def __init__(self, passes: List[PassBase] = []) -> None:
        self._passes = passes

    def __iter__(
        self,
    ):
        return self._passes.__iter__()

    def append(self, PASS: PassBase) -> None:
        self._passes.append(PASS)

    def __call__(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> GraphModule:
        for PASS in self._passes:
            graph_module = PASS(graph_module=graph_module, ctx=ctx, config=config)

        if config.clean_markers_after_all_passes:
            for PASS in self._passes:
                if isinstance(PASS, AnalyzeBase):
                    PASS.clean_all(graph_module)
        return graph_module
