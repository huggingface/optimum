import torch
import torch.nn as nn
import torch.distributed as dist
from ..dist import (
    differentiable_all_gather,
    differentiable_scatter,
    differentiable_all_reduce_sum,
)


class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        gather_output: bool = True,
    ) -> None:
        self.process_group = process_group
        self.word_size = process_group.size()
        assert out_features % self.word_size == 0

        super().__init__(in_features, out_features // self.word_size, bias, device, dtype)
        self.gather_output = gather_output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.gather_output:
            output = differentiable_all_gather(output, self.process_group)
        return output


class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        input_is_parallel: bool = False,
    ) -> None:
        self.process_group = process_group
        self.word_size = process_group.size()
        assert in_features % self.word_size == 0

        super().__init__(in_features // self.word_size, out_features, bias, device, dtype)
        self.input_is_parallel = input_is_parallel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            input = differentiable_scatter(input, self.process_group)

        output = super().forward(input)
        output = differentiable_all_reduce_sum(output, self.process_group)
        return output
