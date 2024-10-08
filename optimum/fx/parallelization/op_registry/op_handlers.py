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
from abc import abstractmethod
from typing import Any, List, Optional

import torch
from torch.fx import Node

from ..core import Config
from ..utils import is_activation, is_cross_entropy, is_cross_entropy_parallel_compatible, is_embedding, is_linear


class Registry:
    """
    Registry class handles registration of parallel axis propagation handlers of different aten ops.
    To support a new aten op, you need to register the corresponding handler class by decorating it with `register` function.
    """

    def __init__(self) -> None:
        self.mapping = {}

    def register(self, op_types):
        def wrapper(cls):
            if isinstance(op_types, (list, tuple)):
                for op_type in op_types:
                    self.mapping[op_type] = cls
            else:
                self.mapping[op_types] = cls
            return cls

        return wrapper

    def is_supported(self, op_type) -> bool:
        return op_type in self.mapping


REGISTRY = Registry()


class OpParallelAxisPropagateHandler:
    def __init__(self, node: Node, meta_key: str, config: Config) -> None:
        self.node = node
        self.meta_key = meta_key
        self.config = config

    def extract_axis(self, arg: Any) -> Optional[int]:
        if not isinstance(arg, Node):
            return None
        return arg.meta[self.meta_key].get("parallel_axis", None)

    @abstractmethod
    def propagate(self) -> List[int]:
        raise NotImplementedError


@REGISTRY.register(
    [
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.bitwise_not.default,
        torch.ops.aten.abs.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.acos.default,
        torch.ops.aten.acosh.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.asin.default,
        torch.ops.aten.asinh.default,
        torch.ops.aten.atan.default,
        torch.ops.aten.atanh.default,
        torch.ops.aten.ceil.default,
        torch.ops.aten.clamp.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.cosh.default,
        torch.ops.aten.erf.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.trunc.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.tan.default,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.sinh.default,
        torch.ops.aten.sign.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.round.default,
        torch.ops.aten.remainder.Scalar,
        torch.ops.aten.relu.default,
        torch.ops.aten.reciprocal.default,
        torch.ops.aten.neg.default,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.native_dropout.default,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.logical_not.default,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.log.default,
        torch.ops.aten.log10.default,
        torch.ops.aten.log2.default,
        torch.ops.aten.log1p.default,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.isnan.default,
        torch.ops.aten.isinf.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.gelu.default,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.fmod.Scalar,
        torch.ops.aten.floor.default,
        torch.ops.aten.fill.Scalar,
        torch.ops.aten.div.Scalar_mode,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_xor.Scalar,
    ]
)
class UnaryOpParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        arg = self.node.all_input_nodes[0]
        axis = self.extract_axis(arg)
        return [axis]


@REGISTRY.register(
    [
        torch.ops.aten.atan2.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Tensor_mode,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.fmod.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.logical_and.default,
        torch.ops.aten.logical_or.default,
        torch.ops.aten.logical_xor.default,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.maximum.default,
        torch.ops.aten.minimum.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.remainder.Tensor,
        torch.ops.aten.sub.Tensor,
    ]
)
class BinaryOpParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        input_nodes = self.node.all_input_nodes
        # only one node
        if len(input_nodes) == 1:
            return UnaryOpParallelAxisPropagateHandler(self.node, self.meta_key, self.config).propagate()

        assert len(input_nodes) == 2, "binary op should have exact two nodes as inputs"
        lhs_shape, rhs_shape = input_nodes[0].meta["val"].shape, input_nodes[1].meta["val"].shape
        lhs_axis = self.extract_axis(input_nodes[0])
        rhs_axis = self.extract_axis(input_nodes[1])
        i, j = len(lhs_shape) - 1, len(rhs_shape) - 1
        while i >= 0 and j >= 0:
            k = max(lhs_shape[i], rhs_shape[j])
            assert (
                k % min(lhs_shape[i], rhs_shape[j]) == 0
            ), f"shape {lhs_shape} and {rhs_shape} are not broadcastable!"
            i -= 1
            j -= 1

        if i < 0 and lhs_axis is not None:
            lhs_axis += j + 1
        if j < 0 and rhs_axis is not None:
            rhs_axis += i + 1

        if lhs_axis is None:
            return [rhs_axis]
        elif rhs_axis is None:
            return [lhs_axis]
        elif lhs_axis != rhs_axis:
            return []
        return [lhs_axis]


@REGISTRY.register(
    [
        torch.ops.aten.amax.default,
        torch.ops.aten.amin.default,
        torch.ops.aten.any.dim,
        torch.ops.aten._log_softmax.default,
        torch.ops.aten._softmax.default,
        torch.ops.aten.cumsum.default,
        torch.ops.aten.mean.dim,
        # torch.ops.aten.min.dim,
        # torch.ops.aten.max.dim,
        torch.ops.aten.var.dim,
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.prod.dim_int,
    ]
)
class ReductionOpParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def extract_dims(
        self,
    ) -> List[int]:
        ndim = self.node.meta["val"].ndim
        dims = None
        if "dim" in self.node.kwargs:
            dims = self.node.kwargs["dim"]
        elif len(self.node.args) > 1 and isinstance(self.node.args[1], (int, list)):
            dims = self.node.args[1]

        if isinstance(dims, int):
            dims = [dims]
        if not dims:
            dims = list(range(ndim))
        dims = [(dim + ndim) % ndim for dim in dims]

        keepdim = False
        if "keepdim" in self.node.kwargs:
            keepdim = self.node.kwargs
        elif len(self.node.args) > 2 and isinstance(self.node.args[2], bool):
            keepdim = self.node.args[2]

        return dims, keepdim

    def propagate(self) -> List[int]:
        dims, keepdim = self.extract_dims()
        arg = self.node.all_input_nodes[0]
        axis = self.extract_axis(arg)
        if axis in dims:
            return []
        if axis is None:
            return [None]
        if keepdim:
            return [axis]
        return [axis - sum([1 if dim < axis else 0 for dim in dims])]


@REGISTRY.register(torch.ops.aten.view.default)
class ViewLikeOpParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        arg = self.node.args[0]
        axis = self.extract_axis(arg)
        if axis is None:
            return [None]
        shape_before, shape_after = arg.meta["val"].shape, self.node.meta["val"].shape
        size = 1
        for i in range(len(shape_before) - 1, axis - 1, -1):
            size *= shape_before[i]

        cur, i, res = 1, len(shape_after) - 1, []
        while cur <= size and i >= 0:
            cur *= shape_after[i]
            if cur == size:
                res.append(i)
            i -= 1

        return res


@REGISTRY.register(torch.ops.aten.unsqueeze.default)
class UnsqueezeParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        arg, dim = self.node.args[0], self.node.args[1]
        ndim = arg.meta["val"].ndim
        axis = self.extract_axis(arg)
        if axis is None:
            return [None]
        dim = (dim + ndim) % ndim
        if dim <= axis:
            return [axis + 1]
        return [axis]


@REGISTRY.register(
    [
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
    ]
)
class SqueezeParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        arg, dims = self.node.args[0], self.node.args[1]
        axis = self.extract_axis(arg)
        if axis is None:
            return [None]

        ndim = self.node.args[0].meta["val"].ndim
        if isinstance(dims, int):
            dims = [dims]
        dims = [(dim + ndim) % ndim for dim in dims]
        if axis in dims:
            # being conservative
            return []
        return [axis - sum([1 if dim < axis else 0 for dim in dims])]


@REGISTRY.register(torch.ops.aten.permute.default)
class PermuteParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        arg, dims = self.node.args[0], self.node.args[1]
        ndim = arg.meta["val"].ndim
        axis = self.extract_axis(arg)
        if axis is None:
            return [None]

        for i, dim in enumerate(dims):
            if (dim + ndim) % ndim == axis:
                return [i]
        return []


@REGISTRY.register(torch.ops.aten.slice.Tensor)
class SliceParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        arg, slice_dim = self.node.args[0], self.node.args[1]
        axis = self.extract_axis(arg)
        if axis is None:
            return [None]
        ndim = arg.meta["val"].ndim
        slice_dim = (slice_dim + ndim) % ndim
        if slice_dim == axis:
            # slice on the parallel axis is not allowed, except it's a nop
            start, stop, step = 0, arg.meta["val"].shape[axis], 1
            if len(self.node.args) > 2:
                start = self.node.args[2]
            elif len(self.node.args) > 3:
                stop = self.node.args[3]
            elif len(self.node.args) > 4:
                step = self.node.args[4]
            if start == 0 and stop >= arg.meta["val"].shape[axis] and step == 1:
                return [axis]
            return []
        return [axis]


@REGISTRY.register(torch.ops.aten.expand.default)
class ExpandParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        arg, size = self.node.args[0], self.node.args[1]
        axis = self.extract_axis(arg)
        if axis is None:
            return [None]
        assert len(size) >= arg.meta["val"].ndim, "input size must be broadcastable to the target size in expand"
        return [axis + len(size) - arg.meta["val"].ndim]


@REGISTRY.register(torch.ops.aten.cat.default)
class CatParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        nodes, cat_axis = self.node.all_input_nodes, self.node.args[1]
        axis, ndim = self.extract_axis(nodes[0]), nodes[0].meta["val"].ndim
        cat_axis = (cat_axis + ndim) % ndim
        if cat_axis == axis:
            return []
        for i in range(1, len(nodes)):
            if self.extract_axis(nodes[i]) != axis:
                return []
        return [axis]


@REGISTRY.register(torch.ops.aten.constant_pad_nd.default)
class PadParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        pad, ndim = self.node.args[1], self.node.args[0].meta["val"].ndim
        axis = self.extract_axis(self.node.args[0])
        if axis is None:
            return [None]
        if axis >= ndim - pad // 2:
            return []
        return [axis]


@REGISTRY.register(torch.ops.aten.copy.default)
class CopyParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        dst, src = self.node.all_input_nodes
        axis_dst = self.extract_axis(dst)
        axis_src = self.extract_axis(src)
        if axis_dst != axis_src:
            return []
        return [axis_dst]


@REGISTRY.register(torch.nn.functional.scaled_dot_product_attention)
class SpdaAttnParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        q, k, v = self.node.args[:3]
        q_axis = self.extract_axis(q)
        # parallel axis must be the head dimension if being parallelized
        if q_axis != self.extract_axis(k) or q_axis != self.extract_axis(v) or q_axis not in {None, 1}:
            return []
        return [q_axis]


class FallbackParallelAxisPropagateHandler(OpParallelAxisPropagateHandler):
    def propagate(self) -> List[int]:
        # by default we don't parallelize inputs and constants(except parameters embeded in modules)
        if self.node.op in ["placeholder", "get_attr"]:
            return [None]
        elif self.node.op == "output":
            # does not care about if output is being parallelized right now, because if the output is loss,
            # then it must be not parallelized as long as it comes from sharded cross entropy.
            # TODO: append all-gather comm ops before all parallelized output nodes if instructed.
            input_arg = self.node.all_input_nodes[0]
            axis = self.extract_axis(input_arg)
            return [axis]
        elif is_linear(self.node):
            input_arg = self.node.all_input_nodes[0]
            axis = self.extract_axis(input_arg)
            if axis is None:
                # with input being not parallelized, output can be parallelized on the head dimension,
                # i.e., `ColumnLinear`, or not being parallelized by all-gather at the end
                return [2, None]
            elif self.config.enable_sequence_parallel and axis == 1:
                # with input being parallelized on sequence dimension, output can be parallelized on
                # the head dimension, i.e., `ColumnLinear` with sequence parallel, or not being parallelized
                # by all-gather at the end
                return [2, None]
            elif axis == 2:
                # with input being parallelized on head dimension, output can be parallelized on the
                # sequence dimension or not parallelized by all-reduce at the end, i.e., `RowLinear`
                # when sp is not enabled
                return [1, None] if self.config.enable_sequence_parallel else [None]
            else:
                return []
        elif is_embedding(self.node):
            input_arg = self.node.all_input_nodes[0]
            axis = self.extract_axis(input_arg)
            if axis is None:
                # only support the embedding parameter being parallelized on `vocab` dim or not parallelized for now,
                # the output can be parallelized on sequence dim or not parallelized
                return [1, None] if self.config.enable_sequence_parallel else [None]
            else:
                return []
        elif is_cross_entropy(self.node):
            logits = self.node.all_input_nodes[0]
            axis = self.extract_axis(logits)
            if axis is None or (
                is_cross_entropy_parallel_compatible(self.node) and axis == logits.meta["val"].ndim - 1
            ):
                # for cross entropy, the input logits parallel axis can only be the last axis or None
                return [None]
            else:
                return []
        elif is_activation(self.node):
            return UnaryOpParallelAxisPropagateHandler(self.node, self.meta_key, self.config).propagate()

        # last resort, if no input is being parallelized, then we make output also not parallelized,
        # this will give us relief on writing policies for strange ops which don't actually need
        # parallelization in most cases
        if all(self.extract_axis(arg) is None for arg in self.node.all_input_nodes):
            return [None]

        raise NotImplementedError(f"don't know how to propagate axis for {self.node.target}")
