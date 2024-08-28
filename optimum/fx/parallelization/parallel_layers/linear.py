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
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..core import (
    ParallelExecutionCtx,
    ParameterMeta,
)
from ..distributed import (
    differentiable_all_gather,
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_scatter,
)
from ..utils import ensure_divisibility


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        ctx(`ParallelExecutionCtx`): parallel execution context which contains runtime information.
        linear(`torch.nn.Linear`): the original linear module being replaced.
        gather_output(`bool`, defaults to `True`): whether gathering output in the end of forward.
    """

    def __init__(self, ctx: ParallelExecutionCtx, linear: nn.Linear, gather_output: bool = True) -> None:
        super(ColumnParallelLinear, self).__init__()
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        tp_rank = dist.get_rank(self.process_group)
        ensure_divisibility(linear.out_features, world_size)

        out_features = linear.out_features // world_size
        bias = linear.bias is not None

        # modify meta information
        weight_meta = getattr(linear.weight, "meta", None)
        assert isinstance(
            weight_meta, ParameterMeta
        ), "should have run `initialize_parameter_meta` after moving model to current device"

        if weight_meta.is_modified_meta:
            assert weight_meta.is_tied, "only tied parameters could already have modified meta"
        else:
            weight_meta.need_initialize = True
            weight_meta.is_parallel = True
            weight_meta.dim = 0
            for _, Slice in weight_meta.mapping.items():
                Slice.index = slice(tp_rank * out_features, (tp_rank + 1) * out_features)
            weight_meta.is_modified_meta = True

        # skip creating actual parameters
        self.weight = linear.weight
        self.gather_output = gather_output

        if bias:
            bias_meta = getattr(linear.bias, "meta", None)
            assert isinstance(
                bias_meta, ParameterMeta
            ), "should have run `initialize_parameter_meta` after moving model to current device"

            if bias_meta.is_modified_meta:
                assert bias_meta.is_tied, "only tied parameters could already have modified meta"
            else:
                bias_meta.need_initialize = True
                bias_meta.is_parallel = True
                bias_meta.init_fn = torch.zero_
                bias_meta.dim = 0
                for _, Slice in bias_meta.mapping.items():
                    Slice.index = slice(tp_rank * out_features, (tp_rank + 1) * out_features)
                bias_meta.is_modified_meta = True
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = differentiable_identity(input, self.process_group)
        output = F.linear(input, self.weight, self.bias)
        if self.gather_output:
            output = differentiable_all_gather(output, self.process_group)
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        ctx(`ParallelExecutionCtx`): parallel execution context which contains runtime information.
        linear(`torch.nn.Linear`): the original linear module being replaced.
        input_is_parallel(`bool`, defaults to `True`): whether the input tensor has already been parallelized.
    """

    def __init__(self, ctx: ParallelExecutionCtx, linear: nn.Linear, input_is_parallel: bool = False) -> None:
        super(RowParallelLinear, self).__init__()
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        tp_rank = dist.get_rank(self.process_group)
        ensure_divisibility(linear.in_features, world_size)

        in_features = linear.in_features // world_size
        bias = linear.bias is not None

        # modify meta information
        weight_meta = getattr(linear.weight, "meta", None)
        assert isinstance(
            weight_meta, ParameterMeta
        ), "should have run `initialize_parameter_meta` after moving model to current device"

        if weight_meta.is_modified_meta:
            assert weight_meta.is_tied, "only tied parameters could already have modified meta"
        else:
            weight_meta.need_initialize = True
            weight_meta.is_parallel = True
            weight_meta.dim = 1
            for _, Slice in weight_meta.mapping.items():
                Slice.index = slice(tp_rank * in_features, (tp_rank + 1) * in_features)
            weight_meta.is_modified_meta = True

        # skip creating actual parameters
        self.weight = linear.weight
        self.input_is_parallel = input_is_parallel

        if bias:
            bias_meta = getattr(linear.bias, "meta", None)
            assert isinstance(
                bias_meta, ParameterMeta
            ), "should have run `initialize_parameter_meta` after moving model to current device"
            if bias_meta.is_modified_meta:
                assert bias_meta.is_tied, "only tied parameters could already have modified meta"
            else:
                bias_meta.need_initialize = True
                bias_meta.init_fn = torch.zero_
                bias_meta.is_modified_meta = True
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            input = differentiable_scatter(input, self.process_group)

        output = F.linear(input, self.weight)
        output = differentiable_all_reduce_sum(output, self.process_group)

        if self.bias is not None:
            output = output + self.bias
        return output
