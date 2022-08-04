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

        ### DEBUG @thomasw21:: Check that shard model output the same as the non sharded version
        out_from_tp_ranks = [torch.empty_like(out) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(out_from_tp_ranks, out, group=self.process_group)
        sharded_out = torch.cat(out_from_tp_ranks, dim=-1)

        weight_from_tp_ranks = [torch.empty_like(self.weight) for _ in range(self.process_group.size())]
        bias_from_tp_ranks = [torch.empty_like(self.bias) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(weight_from_tp_ranks, self.weight, group=self.process_group)
        torch.distributed.all_gather(bias_from_tp_ranks, self.bias, group=self.process_group)
        weight = torch.cat(weight_from_tp_ranks, dim=0)
        bias = torch.cat(bias_from_tp_ranks, dim=0)
        baseline_out = F.linear(input, weight, bias)

        torch.testing.assert_close(sharded_out, baseline_out, atol=0.0, rtol=0.0)
        ###

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
        # Note the the unsharded equivalent requires us to sum over bias instead of averaging.
        out = super().forward(input)
        torch.distributed.all_reduce(out, group=self.process_group)

        ### DEBUG @thomasw21:: Check that shard model output the same as the non sharded version
        sharded_out = out

        input_from_tp_ranks = [torch.empty_like(input) for _ in range(self.process_group.size())]
        weight_from_tp_ranks = [torch.empty_like(self.weight) for _ in range(self.process_group.size())]
        bias = self.bias.clone()
        torch.distributed.all_gather(input_from_tp_ranks, input, group=self.process_group)
        torch.distributed.all_gather(weight_from_tp_ranks, self.weight, group=self.process_group)
        torch.distributed.all_reduce(bias, group=self.process_group)
        input = torch.cat(input_from_tp_ranks, dim=-1)
        weight = torch.cat(weight_from_tp_ranks, dim=1)
        baseline_out = F.linear(input, weight, bias)

        if self.process_group.rank() == 0:
            torch.testing.assert_close(bias, self.bias, atol=0.0, rtol=0.0)
        torch.distributed.barrier(self.process_group)
        torch.testing.assert_close(sharded_out, baseline_out)
        ###

        return out
