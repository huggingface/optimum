import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from typing import Callable
from ..core import (
    ParallelExecutionCtx,
    ParallelParameterMapping,
    ParameterMeta,
)
from ..distributed import (
    differentiable_identity,
    differentiable_all_gather,
    differentiable_scatter,
    differentiable_all_reduce_sum,
    scatter,
)


class ColumnParallelLinear(nn.Linear):
    """
    Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        ctx: parallel execution context which contains runtime information.
        linear: the original linear module being replaced.
        gather_output: whether gathering output in the end of forward.
        init_fn: weight initialization function.
    """
    def __init__(
        self,
        ctx: ParallelExecutionCtx,
        linear: nn.Linear,
        gather_output: bool = True,
        init_fn: Callable = partial(nn.init.normal_, mean=0, std=0.02),
    ) -> None:
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        assert linear.out_features % world_size == 0

        in_features = linear.in_features
        out_features = linear.out_features // world_size
        bias = linear.bias is not None
        device = ctx.current_device
        dtype = linear.weight.dtype

        super().__init__(in_features, out_features, bias, device, dtype)
        self.gather_output = gather_output
        tp_rank = dist.get_rank(self.process_group)

        parameter_mapping, key = ctx.parameter_mapping, id(linear.weight)
        assert key in parameter_mapping, "should have run `initialize_paramter_mapping` after moving model to current device"
        original_linear_weight_meta = parameter_mapping[key].meta

        # initialize the weight if not in weight_map
        need_intialize = original_linear_weight_meta.source not in ctx.weight_map
        if need_intialize:
            # initialize on cpu
            master_weight = torch.empty_like(linear.weight, device='cpu')
            init_fn(master_weight)
            with torch.no_grad():
                self.weight.copy_(master_weight[tp_rank * out_features : (tp_rank + 1) * out_features, :])

        # update parameter mapping corresponding to original linear weight and bias
        linear_weight_mapping = ParallelParameterMapping(
            id=id(self.weight),
            meta=ParameterMeta(
                source=original_linear_weight_meta.source,
                dim=0,
                index=slice(tp_rank * out_features, (tp_rank + 1) * out_features)
            ),
            parallel_dim=0
        )
        parameter_mapping.pop(key)
        parameter_mapping[linear_weight_mapping.id] = linear_weight_mapping

        if bias:
            key = id(linear.bias)
            assert key in parameter_mapping
            original_linear_bias_meta = parameter_mapping[key].meta
            linear_bias_mapping = ParallelParameterMapping(
                id=id(self.bias),
                meta=ParameterMeta(
                    source=original_linear_bias_meta.source,
                    dim=0,
                    index=slice(tp_rank * out_features, (tp_rank + 1) * out_features)
                ),
                parallel_dim=0
            )

            parameter_mapping.pop(key)
            parameter_mapping[linear_bias_mapping.id] = linear_bias_mapping
            self.bias.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = differentiable_identity(input, self.process_group)
        output = super().forward(input)
        if self.gather_output:
            output = differentiable_all_gather(output, self.process_group)
        return output


class RowParallelLinear(nn.Linear):
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
        ctx: parallel execution context which contains runtime information.
        linear: the original lineat module being replaced.
        input_is_parallel: whether the input tensor has already been parallelized.
        init_fn: weight initialization function.
    """
    def __init__(
        self,
        ctx: ParallelExecutionCtx,
        linear: nn.Linear,
        input_is_parallel: bool = False,
        init_fn: Callable = partial(nn.init.normal_, mean=0, std=0.02),
    ) -> None:
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        assert linear.in_features % world_size == 0

        in_features = linear.in_features // world_size
        out_features = linear.out_features
        bias = linear.bias is not None
        device = ctx.current_device
        dtype = linear.weight.dtype

        super().__init__(in_features, out_features, bias, device, dtype)
        self.input_is_parallel = input_is_parallel
        tp_rank = dist.get_rank(self.process_group)

        parameter_mapping, key = ctx.parameter_mapping, id(linear.weight)
        assert key in parameter_mapping, "should have run `initialize_paramter_mapping` after moving model to current device"
        original_linear_weight_meta = parameter_mapping[key].meta

        need_intialize = original_linear_weight_meta.source not in ctx.weight_map
        if need_intialize:
            # initialize on cpu
            master_weight = torch.empty_like(linear.weight, device='cpu')
            init_fn(master_weight)
            with torch.no_grad():
                self.weight.copy_(master_weight[:, tp_rank * in_features : (tp_rank + 1) * in_features])
        
        # update parameter mapping corresponding to original linear weight and bias
        linear_weight_mapping = ParallelParameterMapping(
            id=id(self.weight),
            meta=ParameterMeta(
                source=original_linear_weight_meta.source,
                dim=1,
                index=slice(tp_rank * in_features, (tp_rank + 1) * in_features)
            ),
            parallel_dim=1
        )
        parameter_mapping.pop(key)
        parameter_mapping[linear_weight_mapping.id] = linear_weight_mapping

        if bias:
            key = id(linear.bias)
            assert key in parameter_mapping
            linear_bias_mapping = parameter_mapping[key]
            parameter_mapping.pop(key)
            linear_bias_mapping.id = id(self.bias)
            parameter_mapping[linear_bias_mapping.id] = linear_bias_mapping
            self.bias.zero_()


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            input = differentiable_scatter(input, self.process_group)

        output = F.linear(input, self.weight)
        output = differentiable_all_reduce_sum(output, self.process_group)

        if self.bias is not None:
            output = output + self.bias
        return output
