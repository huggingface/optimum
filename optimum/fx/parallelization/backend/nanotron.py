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
# Nanotron specific imports
import importlib.util
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Tuple

import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule

from ..core import Config, ParallelExecutionCtx, ParameterMeta
from .base import BackEnd


# Check if nanotron is installed
_nanotron_available = importlib.util.find_spec("nanotron") is not None

if TYPE_CHECKING:
    from nanotron.config import Config as NanotronConfig
    from nanotron.parallel import ParallelContext
    from nanotron.parallel.tensor_parallel.nn import (
        TensorParallelColumnLinear,
        TensorParallelEmbedding,
        TensorParallelRowLinear,
    )


class NanotronBackend(BackEnd):
    """
    Backend class which glues optimum fx parallelization context and nanotron context.
    """

    def __init__(self, nanotron_config: "NanotronConfig", nanotron_context: "ParallelContext") -> None:
        if not _nanotron_available:
            raise ImportError("Nanotron is not installed. Please install it to use NanotronBackend.")

        self.config = nanotron_config
        self.context = nanotron_context

    def create_column_parallel_linear(
        self,
        mod: nn.Linear,
        parallel_ctx: "ParallelExecutionCtx",
        sequence_parallel: bool,
        gather_output: bool,
        contiguous_chunks: Optional[Tuple[int]] = None,
    ) -> "TensorParallelColumnLinear":
        from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
        from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear

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
    ) -> "TensorParallelRowLinear":
        from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
        from nanotron.parallel.tensor_parallel.nn import TensorParallelRowLinear

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
    ) -> "TensorParallelEmbedding":
        from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
        from nanotron.parallel.tensor_parallel.nn import TensorParallelEmbedding

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

    def post_process(
        self, graph_module: GraphModule, parallel_ctx: "ParallelExecutionCtx", config: "Config"
    ) -> nn.Module:
        from nanotron.parallel.parameters import NanotronParameter
        from nanotron.parallel.tied_parameters import tie_parameters

        param_cache, tied_parameter_groups = parallel_ctx.param_cache, defaultdict(list)
        for name, param in graph_module.named_parameters():
            param_meta: "ParameterMeta" = getattr(param, "meta")
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

                if name not in param_cache:
                    new_param = NanotronParameter(param.detach(), param.requires_grad)
                    param_cache[name] = new_param
                else:
                    raise RuntimeError(
                        "Found already initialized parameter which is not a nanotron parameter in parameter cache!"
                    )
                setattr(parent_mod, field, new_param)
            elif name not in param_cache:
                param_cache[name] = param

            # now we have NanotronParameter anyway
            nanotron_param: NanotronParameter = param_cache[name]
            # we have tied the parameter, in the very first compilation.
            if nanotron_param.is_tied:
                continue

            # not tied, must be the very first compilation
            assert parallel_ctx.compile_times == 0, "illegal path for recompilation"
            host_name = param_meta.tied_to if param_meta.tied_to is not None else name
            tied_parameter_groups[host_name].append(name)

        # take care of weights tying
        for _, groups in tied_parameter_groups:
            # just one parameter in the group, no need for tying
            if len(groups) == 1:
                continue

            ties = [
                (
                    target,
                    (
                        self.context.get_global_rank(
                            # TODO: modify this accordingly when ep is supported
                            ep_rank=0,
                            # TODO: modify this accordingly when pp is supported
                            pp_rank=0,
                            dp_rank=dist.get_rank(self.context.dp_pg),
                            tp_rank=dist.get_rank(self.context.tp_pg),
                        ),
                    ),
                )
                for target in groups
            ]
            # no new parameters will be created because we make sure every param is already a NanotronParameter
            tie_parameters(graph_module, ties, self.context, dist.ReduceOp.SUM)

        return graph_module
