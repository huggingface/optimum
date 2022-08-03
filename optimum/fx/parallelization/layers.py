import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn


class TensorParallelColumnLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        return out


class TensorParallelRowLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, self.weight, None)
        torch.distributed.all_reduce(out, group=self.process_group)
        if self.bias is not None:
            return out + self.bias
        else:
            return out
