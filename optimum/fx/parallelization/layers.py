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
        self.process_group = process_group
        self.tp_world_size = process_group.size()

        assert out_features % self.tp_world_size == 0
        self.block_size = out_features // self.tp_world_size

        super().__init__(in_features, self.block_size, bias=bias, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


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
        self.process_group = process_group
        self.tp_world_size = process_group.size()

        assert in_features % self.tp_world_size == 0
        self.block_size = in_features // self.tp_world_size

        super().__init__(self.block_size, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Note the the unsharded equivalent requires us to sum over bias instead of averaging.
        out = super().forward(input)
        torch.distributed.all_reduce(out, group=self.process_group)
        return out

class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        process_group: torch.distributed.ProcessGroup,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None
    ):
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        # TODO @thomasw21 fix and remove that constraint
        assert num_embeddings % self.tp_world_size == 0
        self.block_size = num_embeddings // self.tp_world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.tp_rank * self.block_size
        self.max_id = (self.tp_rank + 1) * self.block_size

        super().__init__(self.max_id - self.min_id, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Sanity check
        if torch.any(0 > input | input >= self.weight.shape[0]):
            raise IndexError(f"Input is required to be in [0, {self.weight.shape[0]}[")

        # `0` if input is in the correct interval, else `1`
        input_mask = self.min_id < input | input >= self.max_id
        input = input - self.min_id
        # default all out of bounds values to `0`
        input[input_mask] = 0
        out = super().forward(input)
        out[input_mask] = 0
        torch.distributed.all_reduce(out, group=self.process_group)
        return out
