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

from ..core import ParallelExecutionCtx
from ..distributed import differentiable_all_reduce_sum
from ..utils import ensure_divisibility


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer parallelized in vocabulary dimension.

    Arguments:
        ctx(`ParallelExecutionCtx`): parallel execution context which contains runtime information.
        embedding(`torch.nn.Embedding`): the original embedding module being replaced.
    """

    def __init__(self, ctx: ParallelExecutionCtx, embedding: nn.Embedding):
        super(VocabParallelEmbedding, self).__init__()
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        tp_rank = dist.get_rank(self.process_group)
        ensure_divisibility(embedding.num_embeddings, world_size)

        num_embeddings = embedding.num_embeddings // world_size

        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse
        self.vocab_start_idx = tp_rank * num_embeddings
        self.vocab_end_idx = (tp_rank + 1) * num_embeddings

        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding.embedding_dim), dtype=embedding.weight.dtype, device=ctx.current_device
            ),
            requires_grad=embedding.weight.requires_grad,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_mask = (input < self.vocab_start_idx) | (input >= self.vocab_end_idx)
        masked_input = input.clone() - self.vocab_start_idx
        masked_input[input_mask] = 0

        output = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        output[input_mask, :] = 0.0
        output = differentiable_all_reduce_sum(output, self.process_group)
        return output
