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

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node

from .core import Config, ParallelExecutionCtx, ParameterMeta
from .decomp import decompose_and_functionalize
from .distributed import scatter
from .op_registry import REGISTRY, FallbackParallelAxisPropagateHandler
from .parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelCrossEntropyLoss,
    VocabParallelEmbedding,
    sharded_cross_entropy_wrapper_fn,
)
from .utils import (
    is_cross_entropy,
    is_embedding,
    is_linear,
    is_shape_consumer,
    stable_topological_sort,
)


class PassBase(ABC):
    """
    Base class for parallelization targeted passes.
    """

    need_rerun_when_recompile: bool = True

    @classmethod
    def signature(cls) -> str:
        return cls.__name__

    @abstractmethod
    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
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

    def __call__(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
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

    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
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

    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
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

            elif is_cross_entropy(node):
                axis_before = ParallelAxisSolverPass.get_stored_field_info(node.args[0], "parallel_axis")
                if axis_before is not None:
                    self.place_marker_per_node(node, {"axis": "vocab"})

        return graph_module


class ParallelLayerReplacePass(PassBase):
    """
    A pass which modifies graph according to information provided by previous analytical passes, in general it does two things for now:
        1. replaces linears and embedding layers with their parallel counterparts.
        2. modifies hard-coded arguments like the number of attention heads in the graph by dividing it by parallelism level.
    """

    @staticmethod
    def handle_linear(node: Node, ctx: ParallelExecutionCtx) -> None:
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
        key, layer_cache = node.target, ctx.parallel_layer_cache
        if key in layer_cache:
            new_mod = layer_cache[key]
        else:
            if axis == "column":
                gather_output = ParallelLayerAnnotatePass.get_stored_field_info(
                    node, field="gather_output", must_have=True
                )
                new_mod = ColumnParallelLinear(ctx, mod, gather_output)
            else:
                input_is_parallel = ParallelLayerAnnotatePass.get_stored_field_info(
                    node, field="input_is_parallel", must_have=True
                )
                new_mod = RowParallelLinear(ctx, mod, input_is_parallel)
            layer_cache[key] = new_mod
        setattr(parent_mod, field, new_mod)

    @staticmethod
    def handle_embedding(node: Node, ctx: ParallelExecutionCtx) -> None:
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
        key, layer_cache = node.target, ctx.parallel_layer_cache
        if key in layer_cache:
            new_mod = layer_cache[key]
        else:
            assert ctx.compile_times == 0, "illegal path for recompilation"
            new_mod = VocabParallelEmbedding(ctx, mod)
            layer_cache[key] = new_mod
        setattr(parent_mod, field, new_mod)

    @staticmethod
    def handle_cross_entropy(node: Node, ctx: ParallelExecutionCtx) -> None:
        axis = ParallelLayerAnnotatePass.get_stored_field_info(node, field="axis")
        if axis is None:
            return

        assert axis in {"vocab"}, "Only support parallelization on vocab dim for now."
        if node.op == "call_module":
            graph_module = node.graph.owning_module
            prefix_and_field = node.target.rsplit(".", maxsplit=1)
            if len(prefix_and_field) == 2:
                parent_mod = graph_module.get_submodule(prefix_and_field[0])
                field = prefix_and_field[1]
            else:
                parent_mod = graph_module
                field = node.target

            mod: nn.CrossEntropyLoss = graph_module.get_submodule(node.target)
            key, layer_cache = node.target, ctx.parallel_layer_cache
            if key in layer_cache:
                new_mod = layer_cache[key]
            else:
                assert ctx.compile_times == 0, "illegal path for recompilation"
                new_mod = VocabParallelCrossEntropyLoss(ctx, reduction=mod.reduction)
                layer_cache[key] = new_mod
            setattr(parent_mod, field, new_mod)
        else:
            node.target = sharded_cross_entropy_wrapper_fn(process_group=ctx.tp_group)

    @staticmethod
    def handle_hard_coded_axis_param(node: Node, ctx: ParallelExecutionCtx) -> None:
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

    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        for node in graph_module.graph.nodes:
            if is_linear(node):
                self.handle_linear(node, ctx)
            elif is_embedding(node):
                self.handle_embedding(node, ctx)
            elif is_cross_entropy(node):
                self.handle_cross_entropy(node, ctx)
            # correct the attention head num in parallel setting
            elif is_shape_consumer(node):
                self.handle_hard_coded_axis_param(node, ctx)
        return graph_module


class InitializeOrLoadWeightsPass(PassBase):
    """
    Weights loading and intialization pass, will initialize parameters on current rank and load weights from disk
    if necessary.
    """

    def run(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        world_size = dist.get_world_size(ctx.tp_group)
        tp_rank = dist.get_rank(ctx.tp_group)

        new_parameters, tied_parameters, param_cache = [], {}, ctx.param_cache
        for name, param in sorted(graph_module.named_parameters(remove_duplicate=False)):
            # skip initializing new params when recompilation happens
            if name in param_cache:
                new_parameters.append((name, param_cache[name]))
                continue

            param_meta: ParameterMeta = getattr(param, "meta")
            # skip already initialized/loaded tied parameters
            if param_meta.is_tied and id(param) in tied_parameters:
                new_parameters.append((name, tied_parameters[id(param)]))
                continue

            shape = [
                param.size(dim) // world_size if dim == param_meta.dim and param_meta.is_parallel else param.size(dim)
                for dim in range(param.ndim)
            ]

            if not param_meta.is_parallel and param.device == ctx.current_device:
                new_param = param
            else:
                new_param = nn.Parameter(
                    torch.zeros(*shape, dtype=param.dtype, device=ctx.current_device),
                    requires_grad=param.requires_grad,
                )

            # load weights if possible
            for source, target in sorted(param_meta.mapping.items()):
                if target.source in ctx.weight_map:
                    from safetensors import safe_open

                    with safe_open(ctx.weight_map[target.source], framework="pt", device="cpu") as fp:
                        tensor_slice = fp.get_slice(target.source)
                        source_index = [
                            source.to_slice() if dim == param_meta.dim else slice(None, None, None)
                            for dim in range(param.ndim)
                        ]
                        load_index = [
                            target.index if dim == param_meta.dim else slice(None, None, None)
                            for dim in range(param.ndim)
                        ]

                        tensor = tensor_slice[load_index].contiguous()
                        tensor = torch.empty_like(tensor).copy_(tensor)
                        with torch.no_grad():
                            new_param.data[source_index].copy_(tensor)

            # weights initialization
            if param_meta.need_initialize:
                for source, target in sorted(param_meta.mapping.items()):
                    if target.source in ctx.weight_map:
                        continue
                    if not param_meta.is_parallel or tp_rank == 0:
                        # initialize weight on master rank
                        weight = torch.empty(*target.shape, dtype=param.dtype, device="cpu")
                        init_fn = param_meta.init_fn if param_meta.init_fn else config.weight_init_fn
                        init_fn(weight)
                        weight = weight.to(ctx.current_device)
                    else:
                        weight = None
                    index = [
                        source.to_slice() if dim == param_meta.dim else slice(None, None, None)
                        for dim in range(param.ndim)
                    ]
                    with torch.no_grad():
                        if param_meta.is_parallel:
                            scatter(ctx.tp_group, weight, new_param.data[index], scatter_dim=param_meta.dim)
                        else:
                            new_param.data[index].copy_(weight)
            setattr(new_param, "meta", param_meta)

            if id(new_param) != id(param):
                new_parameters.append((name, new_param))
            if param_meta.is_tied:
                tied_parameters[id(param)] = new_param

        for name, new_param in new_parameters:
            prefix_and_field = name.rsplit(".", maxsplit=1)
            if len(prefix_and_field) == 2:
                parent_mod = graph_module.get_submodule(prefix_and_field[0])
                field = prefix_and_field[1]
            else:
                parent_mod = graph_module
                field = name
            if name not in param_cache:
                param_cache[name] = new_param
            setattr(parent_mod, field, new_param)

        return graph_module


def build_parallel_pass_pipeline() -> PassPipeline:
    """
    Ensemble a pass pipeline which contains the following passes:
        1. `ParallelAxisSolverPass` to find a parallelization solution of tensors in the graph.
        2. `ParallelLayerAnnotatePass` to annotate parallelized layers according to the solution found in the first step.
        3. `ParallelLinearReplacePass` to do the actual replacement and modification of hard-coded attributes.
        4. `InitializeOrLoadWeightsPass` to load or initialize weights for parameters.

    Returns:
        PassPipeline: the pipeline used for automatic parallelism.
    """
    return PassPipeline(
        [
            ParallelAxisSolverPass(),
            ParallelLayerAnnotatePass(),
            ParallelLayerReplacePass(),
            InitializeOrLoadWeightsPass(),
        ]
    )


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

    def __call__(self, graph_module: GraphModule, ctx: ParallelExecutionCtx, config: Config) -> GraphModule:
        for PASS in self._passes:
            graph_module = PASS(graph_module=graph_module, ctx=ctx, config=config)

        if config.clean_markers_after_all_passes:
            for PASS in self._passes:
                if isinstance(PASS, AnalyzeBase):
                    PASS.clean_all(graph_module)
        return graph_module
