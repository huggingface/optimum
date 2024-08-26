from typing import TYPE_CHECKING, Optional, Tuple

import torch.nn as nn

# Nanotron
from nanotron.config import Config
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch.fx import GraphModule

from ..passes import (
    ParallelAxisSolverPass,
    ParallelLayerAnnotatePass,
    ParallelLayerReplacePass,
    PassPipeline,
)
from .base import BackEnd


if TYPE_CHECKING:
    from ..core import ParallelExecutionCtx


class NanotronBackend(BackEnd):
    def __init__(self, nanotron_config: Config) -> None:
        self.config = nanotron_config

    def create_column_parallel_linear(
        self,
        mod: nn.Linear,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        gather_output: bool,
        contiguous_chunks: Optional[Tuple[int]] = None,
    ) -> TensorParallelColumnLinear:
        if gather_output:
            raise ValueError(
                "Nanotron backend does not support `gather_output=True` in `TensorParallelColumnLinear` yet"
            )

        if sequence_parallel and self.config.parallelism.tp_mode != TensorParallelLinearMode.REDUCE_SCATTER:
            raise ValueError(
                "`sequence_parallel` can not be activated when `tp_mode` is not set to `REDUCE_SCATTER` in nanotron backend"
            )

        tp_mode = TensorParallelLinearMode.REDUCE_SCATTER if sequence_parallel else TensorParallelLinearMode.ALL_REDUCE
        return TensorParallelColumnLinear(
            in_features=mod.in_features,
            out_features=mod.out_features,
            pg=parallel_ctx.tp_group,
            mode=tp_mode,
            bias=mod.bias is not None,
            device=parallel_ctx.current_device,
            dtype=mod.weight.dtype,
            async_communication=self.config.parallelism.tp_linear_async_communication,
            contiguous_chunks=contiguous_chunks,
        )

    def create_row_parallel_linear(
        self,
        mod: nn.Linear,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        input_is_parallel: bool,
        contiguous_chunks: Optional[Tuple[int]] = None,
    ) -> TensorParallelRowLinear:
        if not input_is_parallel:
            raise ValueError(
                "Nanotron backend does not support `input_is_parallel=True` in `TensorParallelRowLinear` yet"
            )

        if sequence_parallel and self.config.parallelism.tp_mode != TensorParallelLinearMode.REDUCE_SCATTER:
            raise ValueError(
                "`sequence_parallel` can not be activated when `tp_mode` is not set to `REDUCE_SCATTER` in nanotron backend"
            )

        tp_mode = TensorParallelLinearMode.REDUCE_SCATTER if sequence_parallel else TensorParallelLinearMode.ALL_REDUCE
        return TensorParallelRowLinear(
            in_features=mod.in_features,
            out_features=mod.out_features,
            pg=parallel_ctx.tp_group,
            mode=tp_mode,
            bias=mod.bias is not None,
            device=parallel_ctx.current_device,
            dtype=mod.weight.dtype,
            async_communication=self.config.parallelism.tp_linear_async_communication,
            contiguous_chunks=contiguous_chunks,
        )

    def create_parallel_embedding(
        self,
        mod: nn.Embedding,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        contiguous_chunks: Optional[Tuple[int]] = None,
    ) -> TensorParallelEmbedding:
        if sequence_parallel and self.config.parallelism.tp_mode != TensorParallelLinearMode.REDUCE_SCATTER:
            raise ValueError(
                "`sequence_parallel` can not be activated when `tp_mode` is not set to `REDUCE_SCATTER` in nanotron backend"
            )

        tp_mode = TensorParallelLinearMode.REDUCE_SCATTER if sequence_parallel else TensorParallelLinearMode.ALL_REDUCE
        return TensorParallelEmbedding(
            num_embeddings=mod.num_embeddings,
            embedding_dim=mod.embedding_dim,
            pg=parallel_ctx.tp_group,
            mode=tp_mode,
            padding_idx=mod.padding_idx,
            max_norm=mod.max_norm,
            norm_type=mod.norm_type,
            scale_grad_by_freq=mod.scale_grad_by_freq,
            sparse=mod.sparse,
            device=parallel_ctx.current_device,
            dtype=mod.weight.dtype,
            contiguous_chunks=contiguous_chunks,
        )

    def post_process(self, graph_module: GraphModule, parallel_ctx: "ParallelExecutionCtx") -> nn.Module:
        for name, param in graph_module.named_parameters():
            if not isinstance(param, NanotronParameter):
                prefix_and_field = name.rsplit(".", maxsplit=1)
                if len(prefix_and_field) == 2:
                    parent_mod = graph_module.get_submodule(prefix_and_field[0])
                    field = prefix_and_field[1]
                else:
                    parent_mod = graph_module
                    field = name

                assert (
                    param.device == parallel_ctx.current_device
                ), "all parameters should already be on the current device"
                new_param = NanotronParameter(param.detach(), param.requires_grad)
                setattr(parent_mod, field, new_param)

    def init_parallelization_pass_pipeline(self) -> PassPipeline:
        """
        For nanotron backend, parameter initialization and checkpoint loading is handled outside.
        """
        return PassPipeline(
            [
                ParallelAxisSolverPass(),
                ParallelLayerAnnotatePass(),
                ParallelLayerReplacePass(),
            ]
        )
