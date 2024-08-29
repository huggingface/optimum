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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule

from ..distributed import scatter
from ..parallel_layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from ..passes import (
    ParallelAxisSolverPass,
    ParallelLayerAnnotatePass,
    ParallelLayerReplacePass,
    PassPipeline,
)


if TYPE_CHECKING:
    from ..core import Config, ParallelExecutionCtx, ParameterMeta


class BackEnd(ABC):
    @abstractmethod
    def create_column_parallel_linear(
        self,
        mod: nn.Linear,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        gather_output: bool,
        contiguous_chunks: Optional[Tuple[int]] = None,
    ) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def create_row_parallel_linear(
        self,
        mod: nn.Linear,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        input_is_parallel: bool,
        contiguous_chunks: Optional[Tuple[int]] = None,
    ) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def create_parallel_embedding(
        self,
        mod: nn.Embedding,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        contiguous_chunks: Optional[Tuple[int]] = None,
    ) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def post_process(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> nn.Module:
        return graph_module

    @abstractmethod
    def init_parallelization_pass_pipeline(
        self,
    ) -> PassPipeline:
        """
        Ensemble a pass pipeline which contains the following passes:
            1. `ParallelAxisSolverPass` to find a parallelization solution of tensors in the graph.
            2. `ParallelLayerAnnotatePass` to annotate parallelized layers according to the solution found in the first step.
            3. `ParallelLinearReplacePass` to do the actual replacement and modification of hard-coded attributes.

        Returns:
            PassPipeline: the pipeline used for automatic parallelism.
        """
        return PassPipeline(
            [
                ParallelAxisSolverPass(),
                ParallelLayerAnnotatePass(),
                ParallelLayerReplacePass(),
            ]
        )


class DefaultBackend(BackEnd):
    def create_column_parallel_linear(
        self,
        mod: nn.Linear,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        gather_output: bool,
        contiguous_chunks: Tuple[int] | None = None,
    ) -> nn.Module:
        if sequence_parallel or contiguous_chunks is not None:
            raise NotImplementedError(
                "DefaultBackend does not support `sequence_parallel=True` or specifying contiguous chunks for now"
            )
        return ColumnParallelLinear(parallel_ctx, mod, gather_output)

    def create_row_parallel_linear(
        self,
        mod: nn.Linear,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        input_is_parallel: bool,
        contiguous_chunks: Tuple[int] | None = None,
    ) -> nn.Module:
        if sequence_parallel or contiguous_chunks is not None:
            raise NotImplementedError(
                "DefaultBackend does not support `sequence_parallel=True` or specifying contiguous chunks for now"
            )
        return RowParallelLinear(parallel_ctx, mod, input_is_parallel)

    def create_parallel_embedding(
        self,
        mod: nn.Embedding,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        contiguous_chunks: Tuple[int] | None = None,
    ) -> nn.Module:
        if sequence_parallel or contiguous_chunks is not None:
            raise NotImplementedError(
                "DefaultBackend does not support `sequence_parallel=True` or specifying contiguous chunks for now"
            )

        return VocabParallelEmbedding(parallel_ctx, mod)

    def post_process(self, graph_module: GraphModule, ctx: "ParallelExecutionCtx", config: "Config") -> nn.Module:
        world_size = dist.get_world_size(ctx.tp_group)
        tp_rank = dist.get_rank(ctx.tp_group)

        new_parameters, tied_parameters, param_cache = [], {}, ctx.param_cache
        for name, param in sorted(graph_module.named_parameters(remove_duplicate=False)):
            # skip initializing new params when recompilation happens
            if name in param_cache:
                new_parameters.append((name, param_cache[name]))
                continue

            param_meta: "ParameterMeta" = getattr(param, "meta")
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
