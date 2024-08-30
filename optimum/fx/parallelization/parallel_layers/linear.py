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
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..distributed import (
    differentiable_all_gather,
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_scatter,
)
from ..utils import ensure_divisibility


if TYPE_CHECKING:
    from ..core import ParallelExecutionCtx


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

    def __init__(self, ctx: "ParallelExecutionCtx", linear: nn.Linear, gather_output: bool = True) -> None:
        super(ColumnParallelLinear, self).__init__()
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        ensure_divisibility(linear.out_features, world_size)

        out_features = linear.out_features // world_size
        bias = linear.bias is not None

        self.weight = nn.Parameter(
            torch.empty((out_features, linear.in_features), dtype=linear.weight.dtype, device=ctx.current_device),
            linear.weight.requires_grad,
        )
        self.gather_output = gather_output

        if bias:
            self.bias = nn.Parameter(
                torch.empty((out_features,), dtype=linear.bias.dtype, device=ctx.current_device),
                linear.bias.requires_grad,
            )
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

    def __init__(self, ctx: "ParallelExecutionCtx", linear: nn.Linear, input_is_parallel: bool = False) -> None:
        super(RowParallelLinear, self).__init__()
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        ensure_divisibility(linear.in_features, world_size)

        in_features = linear.in_features // world_size
        bias = linear.bias is not None

        self.weight = nn.Parameter(
            torch.empty((linear.out_features, in_features), dtype=linear.weight.dtype, device=ctx.current_device),
            linear.weight.requires_grad,
        )
        self.input_is_parallel = input_is_parallel

        if bias:
            self.bias = nn.Parameter(
                torch.empty((linear.out_features,), dtype=linear.bias.dtype, device=ctx.current_device),
                linear.bias.requires_grad,
            )
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
