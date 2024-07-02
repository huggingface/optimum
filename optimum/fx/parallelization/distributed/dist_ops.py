import torch
import torch.distributed as dist

def all_reduce(group: dist.ProcessGroup, tensor : torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    
    dist.all_reduce(tensor, group=group)
    return tensor

def all_gather(group: dist.ProcessGroup, tensor: torch.Tensor, gather_dim: int = -1) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    rank = dist.get_rank(group = group)

    tensor = tensor.contiguous()
    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    tensors[rank] = tensor
    
    dist.all_gather(tensors, tensor, group=group)
    return torch.cat(tensors, dim=gather_dim)

def split(group: dist.ProcessGroup, tensor: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    rank = dist.get_rank(group)
    size = tensor.size()
    assert size[split_dim] % world_size == 0
    tensors = torch.split(tensor, size[split_dim] // world_size, dim = split_dim)
    tensor = tensors[rank].contiguous()

    return tensor

def scatter(group: dist.ProcessGroup, tensor: torch.Tensor, output_tensor: torch.Tensor, scatter_dim: int = 0) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    rank = dist.get_rank(group)
    if rank == 0:
        size = tensor.size()
        assert size[scatter_dim] % world_size == 0
        tensors = torch.split(tensor, size[scatter_dim] // world_size, dim=scatter_dim)
        scatter_list = [tensor.contiguous() for tensor in tensors]
        output_tensor = scatter_list[rank]
    else:
        scatter_list = None
    dist.scatter(tensor=output_tensor, scatter_list=scatter_list, src=0, group=group)
    return output_tensor


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
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim: int = -1) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim
        return split(group=group, tensor=tensor, split_dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return DifferentiableAllGather.apply(grad_output, group=ctx.group, dim=ctx.dim), None, None


class DifferentiableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim: int = -1) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim
        return all_gather(group=group, tensor=tensor, gather_dim=dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return DifferentiableScatter.apply(grad_output, group=ctx.group, dim=ctx.dim), None, None


def differentiable_all_reduce_sum(tensor: torch.Tensor, group: dist.ProcessGroup):
    return DifferentiableAllReduceSum.apply(tensor, group)

def differentiable_identity(tensor: torch.Tensor,  group: dist.ProcessGroup):
    return DifferentiableIdentity.apply(tensor, group)

def differentiable_all_gather(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1):
    return DifferentiableAllGather.apply(tensor, group, dim)

def differentiable_scatter(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1):
    return DifferentiableScatter.apply(tensor, group, dim)