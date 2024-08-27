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

import torch.nn as nn
from torch.fx import GraphModule

from ..parallel_layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from ..passes import (
    InitializeOrLoadWeightsPass,
    ParallelAxisSolverPass,
    ParallelLayerAnnotatePass,
    ParallelLayerReplacePass,
    PassPipeline,
)


if TYPE_CHECKING:
    from ..core import ParallelExecutionCtx


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
    def post_process(self, graph_module: GraphModule, parallel_ctx: "ParallelExecutionCtx") -> nn.Module:
        return graph_module

    @abstractmethod
    def init_parallelization_pass_pipeline(
        self,
    ) -> PassPipeline:
        raise NotImplementedError


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

    def init_parallelization_pass_pipeline(self):
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
