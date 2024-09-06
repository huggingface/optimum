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
from functools import wraps
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..core import ParallelExecutionCtx


# Adapted from https://github.com/huggingface/nanotron/blob/main/src/nanotron/parallel/tensor_parallel/functional.py
class _ShardedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sharded_logits: torch.Tensor,  # (batch_size, length, sharded_hidden_size)
        target: torch.Tensor,  # (batch_size, length)
        group: dist.ProcessGroup,
    ):
        # Maximum value along last dimension across all GPUs.
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        # Subtract the maximum value.
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)

        # Get the shard's indices
        sharded_hidden_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_hidden_size
        end_index = start_index + sharded_hidden_size

        # Create a mask of valid ids (1 means it needs to be masked).
        target_mask = (target < start_index) | (target >= end_index)
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, shard-size] and target to a 1-D tensor of size [*].
        logits_2d = sharded_logits.view(-1, sharded_hidden_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        sharded_hidden_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_hidden_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def sharded_cross_entropy(sharded_logits: torch.Tensor, target: torch.Tensor, process_group: dist.ProcessGroup):
    return _ShardedCrossEntropy.apply(sharded_logits, target, process_group)


def sharded_cross_entropy_wrapper_fn(process_group: dist.ProcessGroup):
    @wraps(sharded_cross_entropy)
    def wrapper(
        sharded_logits: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        if weight is not None or ignore_index != -100 or label_smoothing != 0.0:
            raise ValueError(
                "Does not support weighted mode, index ignoring and label smoothing in current parallel cross entropy implementation."
            )
        loss: torch.Tensor = sharded_cross_entropy(sharded_logits, target, process_group)

        if size_average is not None or reduce is not None:
            size_average = True if size_average is None else size_average
            reduce = True if reduce is None else reduce

            if size_average and reduce:
                reduction = "mean"
            elif reduce:
                reduction = "sum"
            else:
                reduction = "none"

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

    return wrapper


class VocabParallelCrossEntropyLoss(nn.Module):
    """
    Simple parallel cross entropy implementation which does not support weighted mode and label smoothing yet.
    """

    def __init__(self, ctx: ParallelExecutionCtx, reduction: str = "mean") -> None:
        super(VocabParallelCrossEntropyLoss, self).__init__()
        self.process_group = ctx.tp_group
        self.reduction = reduction

    def forward(self, sharded_logits: torch.Tensor, target: torch.Tensor):
        loss: torch.Tensor = _ShardedCrossEntropy.apply(sharded_logits, target, self.process_group)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
