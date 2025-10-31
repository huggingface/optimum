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

from ..utils import ensure_divisibility


def all_reduce(group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs an all-reduce operation on a tensor across all processes in the group.
    
    Args:
        group: The process group to perform the all-reduce operation on
        tensor: The input tensor to reduce across all processes
        
    Returns:
        The tensor after all-reduce operation (sum across all processes)
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, group=group)
    return tensor


def all_gather(group: dist.ProcessGroup, tensor: torch.Tensor, gather_dim: int = -1) -> torch.Tensor:
    """
    Gathers tensors from all processes in the group along the specified dimension.
    
    Args:
        group: The process group to gather tensors from
        tensor: The input tensor to gather from each process
        gather_dim: The dimension along which to gather tensors (default: -1)
        
    Returns:
        A tensor containing all gathered tensors concatenated along the gather dimension
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    gather_dim = (gather_dim + tensor.ndim) % tensor.ndim
    shape = [tensor.size(dim) * world_size if dim == gather_dim else tensor.size(dim) for dim in range(tensor.ndim)]
    if gather_dim != 0:
        shape[0], shape[gather_dim] = shape[gather_dim], shape[0]
    tensors = torch.empty(*shape, dtype=tensor.dtype, device=tensor.device)

    if gather_dim != 0:
        tensor = tensor.transpose(0, gather_dim)
    tensor = tensor.contiguous()

    dist.all_gather_into_tensor(tensors, tensor, group=group)
    if gather_dim != 0:
        tensors = tensors.transpose(0, gather_dim).contiguous()
    return tensors


def split(group: dist.ProcessGroup, tensor: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    """
    Splits a tensor along the specified dimension and returns the chunk for the current process.
    
    Args:
        group: The process group to determine the current rank
        tensor: The input tensor to split
        split_dim: The dimension along which to split the tensor (default: -1)
        
    Returns:
        The tensor chunk corresponding to the current process rank
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    rank = dist.get_rank(group)
    size = tensor.size()
    ensure_divisibility(size[split_dim], world_size)
    tensors = torch.split(tensor, size[split_dim] // world_size, dim=split_dim)
    tensor = tensors[rank].contiguous()

    return tensor


def scatter(
    group: dist.ProcessGroup, tensor: torch.Tensor, output_tensor: torch.Tensor, scatter_dim: int = 0
) -> torch.Tensor:
    """
    Scatters a tensor from rank 0 to all processes in the group along the specified dimension.
    
    Args:
        group: The process group to scatter the tensor to
        tensor: The input tensor to scatter (only used on rank 0)
        output_tensor: The output tensor to store the scattered chunk
        scatter_dim: The dimension along which to scatter the tensor (default: 0)
        
    Returns:
        The output tensor containing the scattered chunk for the current process
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        output_tensor.copy_(tensor)
        return tensor

    rank = dist.get_rank(group)
    if rank == 0:
        size = tensor.size()
        ensure_divisibility(size[scatter_dim], world_size)
        tensors = torch.split(tensor, size[scatter_dim] // world_size, dim=scatter_dim)
        scatter_list = [tensor.contiguous() for tensor in tensors]
        output_tensor.copy_(scatter_list[rank])
    else:
        scatter_list = None
    dist.scatter(tensor=output_tensor, scatter_list=scatter_list, src=0, group=group)
    return output_tensor


class DifferentiableIdentity(torch.autograd.Function):
    """
    A differentiable identity function that performs all-reduce on gradients during backward pass.
    """
    
    @staticmethod
    def forward(ctx, tensor, group: dist.ProcessGroup):
        """
        Forward pass that returns the input tensor unchanged.
        
        Args:
            ctx: Context object to save information for backward pass
            tensor: Input tensor
            group: Process group for gradient synchronization
            
        Returns:
            The input tensor unchanged
        """
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass that performs all-reduce sum on the gradient.
        
        Args:
            ctx: Context object containing saved information
            grad_output: Gradient from the next layer
            
        Returns:
            Tuple of gradients for input arguments (tensor gradient, None for group)
        """
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    """
    A differentiable all-reduce sum operation that maintains gradients through the operation.
    """
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """
        Forward pass that performs all-reduce sum on the input tensor.
        
        Args:
            ctx: Context object to save information for backward pass
            tensor: Input tensor to reduce
            group: Process group for the all-reduce operation
            
        Returns:
            The tensor after all-reduce sum operation
        """
        ctx.group = group
        return all_reduce(group=group, tensor=tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass that returns the gradient unchanged.
        
        Args:
            ctx: Context object containing saved information
            grad_output: Gradient from the next layer
            
        Returns:
            Tuple of gradients for input arguments (gradient unchanged, None for group)
        """
        return grad_output, None


class DifferentiableScatter(torch.autograd.Function):
    """
    A differentiable scatter operation that performs all-gather on gradients during backward pass.
    """
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim: int = -1) -> torch.Tensor:
        """
        Forward pass that splits the tensor and returns the chunk for the current process.
        
        Args:
            ctx: Context object to save information for backward pass
            tensor: Input tensor to scatter
            group: Process group for the scatter operation
            dim: Dimension along which to scatter (default: -1)
            
        Returns:
            The tensor chunk for the current process
        """
        ctx.group = group
        ctx.dim = dim
        return split(group=group, tensor=tensor, split_dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass that performs all-gather on the gradient.
        
        Args:
            ctx: Context object containing saved information
            grad_output: Gradient from the next layer
            
        Returns:
            Tuple of gradients for input arguments (gathered gradient, None for group and dim)
        """
        return DifferentiableAllGather.apply(grad_output, group=ctx.group, dim=ctx.dim), None, None


class DifferentiableAllGather(torch.autograd.Function):
    """
    A differentiable all-gather operation that performs scatter on gradients during backward pass.
    """
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim: int = -1) -> torch.Tensor:
        """
        Forward pass that gathers tensors from all processes along the specified dimension.
        
        Args:
            ctx: Context object to save information for backward pass
            tensor: Input tensor to gather
            group: Process group for the all-gather operation
            dim: Dimension along which to gather (default: -1)
            
        Returns:
            The gathered tensor containing all process chunks
        """
        ctx.group = group
        ctx.dim = dim
        return all_gather(group=group, tensor=tensor, gather_dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass that scatters the gradient to the current process chunk.
        
        Args:
            ctx: Context object containing saved information
            grad_output: Gradient from the next layer
            
        Returns:
            Tuple of gradients for input arguments (scattered gradient, None for group and dim)
        """
        return DifferentiableScatter.apply(grad_output, group=ctx.group, dim=ctx.dim), None, None


def differentiable_all_reduce_sum(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """
    Applies differentiable all-reduce sum operation on a tensor.
    
    Args:
        tensor: Input tensor to reduce
        group: Process group for the all-reduce operation
        
    Returns:
        The tensor after all-reduce sum operation with gradient support
    """
    return DifferentiableAllReduceSum.apply(tensor, group)


def differentiable_identity(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """
    Applies differentiable identity operation that synchronizes gradients during backward pass.
    
    Args:
        tensor: Input tensor
        group: Process group for gradient synchronization
        
    Returns:
        The input tensor unchanged with gradient synchronization support
    """
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_gather(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1) -> torch.Tensor:
    """
    Applies differentiable all-gather operation on a tensor.
    
    Args:
        tensor: Input tensor to gather
        group: Process group for the all-gather operation
        dim: Dimension along which to gather (default: -1)
        
    Returns:
        The gathered tensor with gradient support
    """
    return DifferentiableAllGather.apply(tensor, group, dim)


def differentiable_scatter(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1) -> torch.Tensor:
    """
    Applies differentiable scatter operation on a tensor.
    
    Args:
        tensor: Input tensor to scatter
        group: Process group for the scatter operation
        dim: Dimension along which to scatter (default: -1)
        
    Returns:
        The scattered tensor chunk with gradient support
    """
    return DifferentiableScatter.apply(tensor, group, dim)
