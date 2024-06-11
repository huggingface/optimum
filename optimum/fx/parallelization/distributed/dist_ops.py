import torch
import torch.distributed as dist

def all_reduce(group: dist.ProcessGroup, tensor : torch.Tensor) -> torch.Tensor:
    word_size = dist.get_world_size(group)
    if word_size == 1:
        return tensor
    
    dist.all_reduce(tensor, group=group)
    return tensor


def all_gather(group: dist.ProcessGroup, tensor: torch.Tensor, gather_dim = -1) -> torch.Tensor:
    word_size = dist.get_world_size(group)
    if word_size == 1:
        return tensor
    rank = dist.get_rank(group = group)

    tensor = tensor.contiguous()
    tensors = [torch.empty_like(tensor) for _ in range(word_size)]
    tensors[rank] = tensor
    
    dist.all_gather(tensors, tensor, group=group)
    return torch.cat(tensors, dim=gather_dim)


def split(group: dist.ProcessGroup, tensor: torch.Tensor, split_dim = -1) -> torch.Tensor:
    word_size = dist.get_world_size(group)
    if word_size == 1:
        return tensor

    rank = dist.get_rank(group)

    assert tensor.size()[split_dim] % word_size == 0

    tensors = torch.split(tensor, word_size, dim = split_dim)

    tensor = tensors[rank].contiguous()

    return tensor


class DifferentiableIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group: dist.ProcessGroup):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return all_reduce(group=group, tensor=tensor)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Any:
        return grad_output, None


class DifferentiableScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim = -1) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim
        return split(group=group, tensor=tensor, split_dim = dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return DifferentiableAllGather.apply(grad_output, group = ctx.group, dim = ctx.dim), None, None


class DifferentiableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim
        return all_gather(group = group, tensor = tensor, gather_dim = dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return DifferentiableScatter.apply(grad_output, group = ctx.group, dim = ctx.dim), None, None


def differentiable_all_reduce_sum(tensor: torch.Tensor, group: dist.ProcessGroup):
    return DifferentiableAllReduceSum.apply(tensor, group)

def differentiable_identity(tensor: torch.Tensor,  group: dist.ProcessGroup):
    return DifferentiableIdentity.apply(tensor, group)

def differentiable_all_gather(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1):
    return DifferentiableAllGather.apply(tensor, group, dim)

def differentiable_scatter(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1):
    return DifferentiableScatter.apply(tensor, group, dim)