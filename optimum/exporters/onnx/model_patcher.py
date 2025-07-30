# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import dataclasses
import functools
import inspect
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import torch
import transformers
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet

from ...utils import is_transformers_version, logging
from ._traceable_cache import TraceableCache


if is_transformers_version(">=", "4.43") and is_transformers_version("<", "4.48"):
    from transformers.models.clip.modeling_clip import CLIPAttention, CLIPSdpaAttention
if is_transformers_version(">=", "4.48"):
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache
if is_transformers_version(">=", "4.53"):
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        _ignore_causal_mask_sdpa,
        and_masks,
        causal_mask_function,
        eager_mask,
        padding_mask_function,
        prepare_padding_mask,
        sdpa_mask,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
if is_transformers_version(">=", "4.53.1"):
    from transformers.masking_utils import find_packed_sequence_indices

if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel

    from .base import OnnxConfig


logger = logging.get_logger(__name__)


def patch_everywhere(attribute_name: str, patch: Any, module_name_prefix: Optional[str] = None):
    """
    Finds all occurences of `attribute_name` in the loaded modules and patches them with `patch`.

    Args:
        attribute_name (`str`):
            The name of attribute to patch.
        patch (`Any`):
            The patch for the attribute.
        module_name_prefix (`Optional[str]`, defaults to `None`):
            If set, only module names starting with this prefix will be considered for patching.
    """
    # sys.modules may be updated while being iterated over, hence the list copy.
    for name in list(sys.modules):
        module = sys.modules[name]
        if module_name_prefix is not None and not name.startswith(module_name_prefix):
            continue
        if hasattr(module, attribute_name):
            setattr(module, attribute_name, patch)


def override_arguments(args, kwargs, forward_signature, model_kwargs: Dict[str, Any]):
    """
    Override the args and kwargs with the argument values from model_kwargs, following the signature forward_signature corresponding to args and kwargs.
    """
    args = list(args)

    for argument in model_kwargs:
        if argument in forward_signature.parameters:
            argument_index = list(forward_signature.parameters.keys()).index(argument)
            if argument in kwargs or len(args) <= argument_index:
                kwargs[argument] = model_kwargs[argument]
            else:
                args[argument_index] = model_kwargs[argument]
        else:
            kwargs[argument] = model_kwargs[argument]

    return args, kwargs


@dataclasses.dataclass
class PatchingSpec:
    """
    Data class that holds patching specifications.

    Args:
        o: Module / object where the op to patch is located
        name: Name of the op to monkey patch
        custom_op: Custom op that patches the original op
        orig_op: Original op that is being patched
        op_wrapper: Wrapper (optional) that wraps both the original and custom ops.
            It is useful for ops that are class or static methods for instance.
    """

    o: Any
    name: str
    custom_op: Callable
    orig_op: Optional[Callable] = None
    op_wrapper: Optional[Callable] = None


# An ONNX-export-compatible version of `tensor.unfold`. Without this, we get:
# torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of operator Unfold, input size not accessible.
# See https://github.com/pytorch/pytorch/issues/81871 for more information
def onnx_compatible_unfold(input_tensor, dimension, size, step):
    """
    Custom implementation of torch.unfold without using torch.unfold.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        dimension (int): The dimension to unfold.
        size (int): The size of each slice.
        step (int): The step size between slices.

    Returns:
        torch.Tensor: The unfolded tensor.
    """
    # Check if dimension is within the valid range
    if not (-input_tensor.dim() <= dimension < input_tensor.dim()):
        raise ValueError(
            f"Dimension out of range (expected to be in range of [{-input_tensor.dim()}, {input_tensor.dim() - 1}], but got {dimension})"
        )

    # Normalize negative dimension
    dimension = dimension % input_tensor.dim()

    # Compute the shape of the unfolded output
    input_size = input_tensor.size(dimension)
    num_slices = (input_size - size) // step + 1

    # Permute dimension to the end for easier indexing
    input_tensor = input_tensor.transpose(dimension, -1)

    # Extract slices
    slices = []
    for i in range(num_slices):
        start = i * step
        end = start + size
        slices.append(input_tensor[..., start:end])

    # Stack slices and permute dimensions back
    result = torch.stack(slices, dim=-2).transpose(dimension, -2)
    return result


# An ONNX-export-compatible version of `tensor.repeat_interleave`.
# Without this, we get the following error: https://github.com/pytorch/pytorch/issues/145100
# NOTE: This implementation is only necessary for export with dynamo=False (dynamo=True works correctly).
# and can be removed once Optimum switches to dynamo-based exports
def onnx_compatible_repeat_interleave(input_tensor, repeats, dim=None, output_size=None):
    """
    Custom implementation of torch.repeat_interleave without using torch.repeat_interleave.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        repeats (int or torch.Tensor): The number of repetitions for each element.
        dim (int, optional): The dimension along which to repeat. Defaults to None.

    Returns:
        torch.Tensor: The repeated tensor.
    """
    if isinstance(repeats, int) or (torch.is_tensor(repeats) and repeats.dim() == 0):
        if dim is None:
            return input_tensor.flatten().unsqueeze(1).expand(-1, repeats).flatten()
        repeats = torch.full((input_tensor.shape[dim],), repeats, dtype=torch.long, device=input_tensor.device)

    if dim is None:
        return onnx_compatible_repeat_interleave(input_tensor.flatten(), repeats, 0)

    if dim != 0:
        input_tensor = input_tensor.transpose(0, dim)

    # Create expand mask
    max_repeats = repeats.max()
    expanded = input_tensor.unsqueeze(1).expand(-1, max_repeats, *input_tensor.shape[1:])
    mask = torch.arange(max_repeats, device=input_tensor.device) < repeats.unsqueeze(1)
    result = expanded[mask]

    if dim != 0:
        result = result.transpose(0, dim)

    return result


# Custom implementation of torch.linalg.matrix_norm not using torch.linalg.matrix_norm, torch.norm or torch.linalg.norm.
def onnx_compatible_linalg_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> torch.Tensor:
    if ord != 2:
        raise ValueError(
            f"Only ord=2 is supported by onnx_compatible_linalg_norm, but got ord={ord}. "
            "Please extend this function to support other norms."
        )

    if dim is None:
        dim = (-2, -1)

    norm = torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=keepdim))
    if dtype is not None:
        norm = norm.to(dtype)
    if out is not None:
        out.copy_(norm)

    return norm


original_triu = torch.triu
original_tril = torch.tril


# Custom implementation of torch.tril that doesn't fail on int32 tensors.
def onnx_compatible_tril(input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if input_tensor.dtype == torch.int32:
        return original_tril(input_tensor.to(torch.int64), *args, **kwargs).to(torch.int32)
    else:
        return original_tril(input_tensor, *args, **kwargs)


# Custom implementation of torch.triu that doesn't fail on int32 tensors.
def onnx_compatible_triu(input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if input_tensor.dtype == torch.int32:
        return original_triu(input_tensor.to(torch.int64), *args, **kwargs).to(torch.int32)
    else:
        return original_triu(input_tensor, *args, **kwargs)


original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention


# A patched `torch.nn.functional.scaled_dot_product_attention` that doesn't fail during tracing
# from passing `is_causal` as a tensor (which is usually obtained with tensor shapes comparisons).
def traceable_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    if isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_weights = original_scaled_dot_product_attention(
        query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs
    )

    return attn_weights


# No-op bfloat16 casting to avoid issues with legacy ONNX export which cast to complex128
def noop_bfloat16_casting(self):
    return self


UNSUPPORTED_OPS_PATCHING_SPEC = [
    PatchingSpec(torch, "tril", onnx_compatible_tril, torch.tril),
    PatchingSpec(torch, "triu", onnx_compatible_triu, torch.triu),
    PatchingSpec(torch.Tensor, "unfold", onnx_compatible_unfold, torch.Tensor.unfold),
    PatchingSpec(torch.linalg, "norm", onnx_compatible_linalg_norm, torch.linalg.norm),
    PatchingSpec(torch.Tensor, "bfloat16", noop_bfloat16_casting, torch.Tensor.bfloat16),
    PatchingSpec(torch.Tensor, "repeat_interleave", onnx_compatible_repeat_interleave, torch.Tensor.repeat_interleave),
    # TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
    PatchingSpec(torch.Tensor, "__len__", lambda x: x.shape[0], torch.Tensor.__len__),
    PatchingSpec(
        torch.nn.functional,
        "scaled_dot_product_attention",
        traceable_scaled_dot_product_attention,
        torch.nn.functional.scaled_dot_product_attention,
    ),
]


# A patched version of https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/masking_utils.py#L602
# That returns a tensor of zeros with the same shape as position_ids indicating no packed sequence indices.
def find_packed_sequence_indices_patched(position_ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(position_ids)


# Custom vectorized implementation of sdpa_mask without using vmap
def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset

    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
def eager_mask_without_vmap(*args, **kwargs) -> Optional[torch.Tensor]:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), torch.finfo(dtype).min)
    return mask


from torch.onnx.symbolic_opset14 import (  # noqa: E402
    _attention_scale,
    _causal_attention_mask,
    _onnx_symbolic,
    _type_utils,
    jit_utils,
    symbolic_helper,
)


@_onnx_symbolic("aten::__ior_")
@symbolic_helper.parse_args("v", "v")
def __ior_(g: jit_utils.GraphContext, self: torch._C.Value, other: torch._C.Value) -> torch._C.Value:
    return g.op("Or", self, other)


@_onnx_symbolic("aten::scaled_dot_product_attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v", "b")
def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: Optional[torch._C.Value] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[torch._C.Value] = None,
    enable_gqa: bool = False,
):
    assert (not is_causal) or (
        is_causal and symbolic_helper._is_none(attn_mask)
    ), "is_causal and attn_mask cannot be set at the same time"
    assert not enable_gqa, "conversion of scaled_dot_product_attention not implemented if enable_gqa is True"

    if symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (key_transposed_axes[-2], key_transposed_axes[-1])
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
        attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
    elif _type_utils.JitScalarType.from_value(attn_mask) == _type_utils.JitScalarType.BOOL:
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
        attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
        # when using scaled dot product attention with a boolean mask, we replace NaN values in attn_weight with 0.0
        attn_weight = g.op(
            "Where", g.op("IsNaN", attn_weight), g.op("Constant", value_t=torch.tensor([0.0])), attn_weight
        )
    elif _type_utils.JitScalarType.from_value(attn_mask) in (
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
        attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
    else:
        raise ValueError(f"Unsupported type for attn_mask: {_type_utils.JitScalarType.from_value(attn_mask)}")

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )

    return g.op("MatMul", attn_weight, value)


class ModelPatcher:
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._model = model

        patching_specs = config.PATCHING_SPECS or []
        patching_specs.extend(UNSUPPORTED_OPS_PATCHING_SPEC)

        self._patching_specs = []
        for spec in patching_specs:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

        self.orig_forward_name = "forward" if hasattr(self._model, "forward") else "call"
        self.orig_forward = getattr(self._model, self.orig_forward_name)

        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        # TODO: remove that once we got rid of OnnxConfigWithLoss or we implemented it better.
        if config.__class__.__name__ == "OnnxConfigWithLoss":
            self.real_config = config._onnx_config
        else:
            self.real_config = config

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            if is_transformers_version(">=", "4.48"):
                if "past_key_values" in signature.parameters:
                    pkv_index = list(signature.parameters.keys()).index("past_key_values")

                    if (
                        pkv_index < len(args)  # pkv is in args
                        and isinstance(args[pkv_index], (list, tuple))
                        and isinstance(args[pkv_index][0], (list, tuple))
                    ):
                        if len(args[pkv_index][0]) == 2:
                            args[pkv_index] = DynamicCache.from_legacy_cache(args[pkv_index])
                        elif len(args[pkv_index][0]) == 4:
                            args[pkv_index] = EncoderDecoderCache.from_legacy_cache(args[pkv_index])
                        else:
                            raise ValueError(
                                f"past_key_values should have either 2 or 4 elements, but it has {len(args[pkv_index][0])} elements"
                            )
                    elif (
                        "past_key_values" in kwargs  # pkv is in kwargs
                        and isinstance(kwargs["past_key_values"], (list, tuple))
                        and isinstance(kwargs["past_key_values"][0], (list, tuple))
                    ):
                        if len(kwargs["past_key_values"][0]) == 2:
                            kwargs["past_key_values"] = DynamicCache.from_legacy_cache(kwargs["past_key_values"])
                        elif len(kwargs["past_key_values"][0]) == 4:
                            kwargs["past_key_values"] = EncoderDecoderCache.from_legacy_cache(
                                kwargs["past_key_values"]
                            )
                        else:
                            raise ValueError(
                                f"past_key_values should have either 2 or 4 elements, but it has {len(kwargs['past_key_values'][0])} elements"
                            )

            outputs = self.orig_forward(*args, **kwargs)

            # This code block handles different cases of the filterd_outputs input to align it with the expected
            # format of outputs. It is common for the output type of a model to vary, such as tensor, list,
            # tuple, etc. For Transformers models, the output is encapsulated in a ModelOutput object that
            # contains the output names of the model. In the case of Timm classification models, the output
            # is of type tensor. By default, it is assumed that the output names mentioned in the ONNX config
            # match the outputs in order.
            filtered_outputs = {}
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                    if (
                        onnx_output_name in config.outputs
                        or (allow_past_in_outputs and name.startswith("past_key_values"))
                        or any(key.startswith(onnx_output_name) for key in config.outputs.keys())
                    ):
                        filtered_outputs[name] = value
            elif isinstance(outputs, (list, tuple)):
                outputs_list = list(config.outputs.keys())
                filtered_outputs = dict(zip(outputs_list, outputs))
            else:
                if len(config.outputs) > 1:
                    num_outputs = len(config.outputs)
                    outputs_str = ", ".join(config.outputs.keys())
                    raise ValueError(
                        f"config.outputs should have only one outputs, but it has {num_outputs} keys: {outputs_str}"
                    )
                else:
                    name = list(config.outputs.keys())[0]
                    filtered_outputs[name] = outputs
                name = list(config.outputs.keys())[0]
                filtered_outputs[name] = outputs

            if is_transformers_version(">=", "4.48"):
                if isinstance(filtered_outputs.get("past_key_values"), (DynamicCache, EncoderDecoderCache)):
                    filtered_outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

            return filtered_outputs

        self.patched_forward = patched_forward

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    def __enter__(self):
        self.patch_ops()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

        if is_transformers_version(">=", "4.44") and is_transformers_version("<", "4.50"):
            self.original_cache_class = transformers.cache_utils.Cache
            transformers.cache_utils.Cache = TraceableCache

        if is_transformers_version(">=", "4.53"):
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask_without_vmap)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

        if is_transformers_version(">=", "4.53.1"):
            self.original_find_packed_sequence_indices = find_packed_sequence_indices
            transformers.masking_utils.find_packed_sequence_indices = find_packed_sequence_indices_patched

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)

        if is_transformers_version(">=", "4.44") and is_transformers_version("<", "4.50"):
            transformers.cache_utils.Cache = self.original_cache_class

        if is_transformers_version(">=", "4.53"):
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask)

        if is_transformers_version(">=", "4.53.1"):
            transformers.masking_utils.find_packed_sequence_indices = self.original_find_packed_sequence_indices

    def __call__(self, *args, **kwargs):
        if getattr(self._model, self.orig_forward_name) is self.orig_forward:
            logger.warning("Running the non-patched model")
        return self._model(*args, **kwargs)


class Seq2SeqModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        # use_cache is by default set to False with pix2struct, so we need to set it to
        # True to export with past key value
        if model.config.model_type == "pix2struct" and allow_past_in_outputs:
            model.config.text_config.use_cache = True

        # Re-use the patched forward method from the parent class
        self.super_patched_forward = self.patched_forward

        @functools.wraps(self.super_patched_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.super_patched_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            outputs = self.super_patched_forward(*args, **kwargs)

            # Filter out cross attention past key values output from the decoder using KV cache, as they are constants.
            filtered_outputs = {}
            for name, value in outputs.items():
                onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                if (
                    onnx_output_name in config.outputs
                    or (allow_past_in_outputs and name.startswith("past_key_values"))
                    or any(key.startswith(onnx_output_name) for key in config.outputs.keys())
                ):
                    if name != "past_key_values":
                        if self.real_config._behavior == "decoder" and name == "encoder_last_hidden_state":
                            # Who cares about the encoder outputs in the decoder?
                            continue
                        else:
                            filtered_outputs[name] = value
                    else:
                        if self.real_config._behavior == "monolith" or (
                            self.real_config._behavior == "decoder"
                            and (self.real_config.is_merged or not self.real_config.use_past_in_inputs)
                        ):
                            filtered_outputs[name] = value
                        elif self.real_config._behavior == "decoder" and self.real_config.use_past_in_inputs:
                            # The filtering happens here. The decoder with use_past_in_inputs=True corresponds to the autoregressive one.
                            filtered_outputs[name] = tuple([v[:2] for v in value])
            return filtered_outputs

        self.patched_forward = patched_forward


class VisionEncoderDecoderPatcher(Seq2SeqModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)
        use_cache = hasattr(self.real_config, "use_past")

        if config._behavior == "decoder" and model.config.decoder.model_type == "trocr" and use_cache:
            model.decoder.model.decoder.config.use_cache = True


class DecoderModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)


class MgpstrModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            # logits is a tuple, so we unpack it and return them as separate outputs
            char_logits, bpe_logits, wp_logits = self.orig_forward(*args, **kwargs).logits

            return {
                "char_logits": char_logits,
                "bpe_logits": bpe_logits,
                "wp_logits": wp_logits,
            }

        self.patched_forward = patched_forward


class SAMModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        def patched_forward(
            pixel_values=None,
            input_points=None,
            input_labels=None,
            image_embeddings=None,
            image_positional_embeddings=None,
            return_dict=True,
            **kwargs,
        ):
            if config.variant == "monolith":
                return self.orig_forward(
                    pixel_values=pixel_values,
                    input_points=input_points,
                    input_labels=input_labels,
                    image_embeddings=image_embeddings,
                    return_dict=return_dict,
                    **kwargs,
                )
            elif config.variant == "split":
                # return_dict = get_argument(args, kwargs, signature, "return_dict")
                if config.vision_encoder:
                    # pixel_values = get_argument(args, kwargs, signature, "pixel_values")
                    image_positional_embeddings = model.get_image_wide_positional_embeddings()

                    # repeat with batch size
                    batch_size = pixel_values.shape[0]
                    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

                    vision_outputs = model.vision_encoder(
                        pixel_values,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=return_dict,
                    )
                    image_embeddings = vision_outputs[0]

                    if not return_dict:
                        return (image_embeddings, image_positional_embeddings)
                    else:
                        return {
                            "image_embeddings": image_embeddings,
                            "image_positional_embeddings": image_positional_embeddings,
                        }
                else:
                    if input_points is None:
                        raise ValueError("input_points is required to export the prompt encoder / mask decoder.")

                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        input_points=input_points,
                        input_labels=input_labels,
                        input_boxes=None,  # Not supported in the ONNX export
                        input_masks=None,  # Not supported in the ONNX export
                    )

                    low_res_masks, iou_predictions, _ = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_positional_embeddings=image_positional_embeddings,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,  # Not supported in the ONNX export
                        attention_similarity=None,  # Not supported in the ONNX export
                        target_embedding=None,  # Not supported in the ONNX export
                        output_attentions=False,
                    )

                    if not return_dict:
                        return (iou_predictions, low_res_masks)
                    else:
                        return {"iou_scores": iou_predictions, "pred_masks": low_res_masks}

        self.patched_forward = patched_forward


def patched_speecht5_prenet_forward(
    self,
    input_values: torch.Tensor,
    speaker_embeddings: Optional[torch.Tensor] = None,
):
    # Dropout is always applied, even when evaluating. See §2.2 in https://arxiv.org/abs/1712.05884.

    inputs_embeds = input_values
    for layer in self.layers:
        inputs_embeds = torch.nn.functional.relu(layer(inputs_embeds))

        # NOTE: we patch the prenet to avoid using torch.nn.functional.dropout, that is exported as a `Dropout` node in the ONNX
        # that is ignored during inference by some runtimes as ONNX Runtime.
        # Reference: https://github.com/microsoft/onnxruntime/issues/9333 & https://github.com/microsoft/onnxruntime/issues/5549
        mask = torch.rand(inputs_embeds.shape, device=inputs_embeds.device) > self.config.speech_decoder_prenet_dropout
        inputs_embeds = inputs_embeds * mask / (1 - self.config.speech_decoder_prenet_dropout)

        # inputs_embeds = nn.functional.dropout(
        #     inputs_embeds, self.config.speech_decoder_prenet_dropout, training=True
        # )

    inputs_embeds = self.final_layer(inputs_embeds)
    inputs_embeds = self.encode_positions(inputs_embeds)

    if speaker_embeddings is not None:
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings)
        speaker_embeddings = speaker_embeddings.unsqueeze(1)
        speaker_embeddings = speaker_embeddings.expand(-1, inputs_embeds.size(1), -1)
        inputs_embeds = torch.cat([inputs_embeds, speaker_embeddings], dim=-1)
        inputs_embeds = torch.nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))

    return inputs_embeds


class SpeechT5ModelPatcher(ModelPatcher):
    def __enter__(self):
        self.patch_ops()
        self._model.speecht5.decoder.prenet.forward = types.MethodType(
            patched_speecht5_prenet_forward, self._model.speecht5.decoder.prenet
        )
        setattr(self._model, self.orig_forward_name, self.patched_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)
        self._model.speecht5.decoder.prenet.forward = types.MethodType(
            self.original_speecht5_prenet_forward, self._model.speecht5.decoder.prenet
        )

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        self.original_speecht5_prenet_forward = model.speecht5.decoder.prenet.forward

        model.vocoder = model_kwargs["vocoder_model"].eval()

        def patched_forward(
            input_ids=None,
            speaker_embeddings=None,
            encoder_outputs=None,
            past_key_values=None,
            output_sequence=None,
            spectrogram=None,
            encoder_attention_mask=None,
        ):
            use_cache = self.real_config.use_past and self.real_config.variant == "with-past"
            if self.real_config._behavior == "encoder":
                encoder_attention_mask = torch.ones_like(input_ids)

                encoder_out = model.speecht5.encoder(
                    input_values=input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True,
                )
                # downsample encoder attention mask
                if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
                    encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
                        encoder_out[0].shape[1], encoder_attention_mask
                    )

                result = {
                    "encoder_outputs": encoder_out.last_hidden_state,
                    "encoder_attention_mask": encoder_attention_mask,
                }

            elif self.real_config._behavior == "decoder":
                # TODO: and self.real_config.use_past_in_inputs
                encoder_hidden_states = encoder_outputs[0]

                decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)

                # Run the decoder layers on the last element of the prenet output.
                decoder_out = model.speecht5.decoder.wrapped_decoder(
                    hidden_states=decoder_hidden_states[:, -1:],
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=False,
                    return_dict=True,
                )

                last_decoder_output = decoder_out.last_hidden_state[0, -1]
                past_key_values = decoder_out.past_key_values

                # Predict the new mel spectrum for this step in the sequence.
                spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
                spectrum = spectrum.view(model.config.reduction_factor, model.config.num_mel_bins)

                # NOTE: extending the spectrogram should is to be handled outside of the ONNX.
                # spectrogram.append(spectrum)

                # Extend the output sequence with the new mel spectrum.
                output_sequence = torch.cat(
                    (output_sequence, spectrum[-1].view(1, 1, model.config.num_mel_bins)), dim=1
                )

                # Predict the probability that this is the stop token.
                prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

                result = {
                    "output_sequence_out": output_sequence,
                    "spectrum": spectrum,
                    "prob": prob,
                    "past_key_values": past_key_values,
                }
            elif self.real_config.is_postnet_and_vocoder:
                # NOTE: the following concatenation is expected to be handled outside of the ONNX:
                # spectrogram = torch.cat(spectrogram, dim=0).unsqueeze(0)
                spectrogram = spectrogram.unsqueeze(0)
                spectrogram = model.speech_decoder_postnet.postnet(spectrogram)
                spectrogram = spectrogram.squeeze(0)

                waveform = model.vocoder(spectrogram)

                result = {"waveform": waveform}
            else:
                raise ValueError("Should not happen")

            # Filter out cross attention past key values output from the decoder using KV cache, as they are constants.
            filterd_outputs = {}
            for name, value in result.items():
                if name != "past_key_values":
                    filterd_outputs[name] = value
                else:
                    if self.real_config._behavior == "decoder" and (
                        self.real_config.is_merged or not self.real_config.use_past_in_inputs
                    ):
                        filterd_outputs[name] = value
                    elif self.real_config._behavior == "decoder" and self.real_config.use_past_in_inputs:
                        # The filtering happens here. The decoder with use_past_in_inputs=True corresponds to the autoregressive one.
                        filterd_outputs[name] = tuple([v[:2] for v in value])

            return filterd_outputs

        self.patched_forward = patched_forward


class SentenceTransformersTransformerPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        def patched_forward(input_ids, attention_mask):
            result = self.orig_forward({"input_ids": input_ids, "attention_mask": attention_mask})

            if "input_ids" in result:
                del result["input_ids"]
            if "attention_mask" in result:
                del result["attention_mask"]
            if "all_layer_embeddings" in result:
                del result["all_layer_embeddings"]

            return result

        self.patched_forward = patched_forward


class SentenceTransformersCLIPPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        def patched_forward(input_ids, attention_mask, pixel_values):
            vision_outputs = model[0].model.vision_model(pixel_values=pixel_values)
            image_embeds = model[0].model.visual_projection(vision_outputs[1])

            text_outputs = model[0].model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeds = model[0].model.text_projection(text_outputs[1])

            if len(model) > 1:
                image_embeds = model[1:](image_embeds)
                text_embeds = model[1:](text_embeds)

            return {"text_embeds": text_embeds, "image_embeds": image_embeds}

        self.patched_forward = patched_forward


# Triu with possible dynamic `diagonal` argument. Not possible with torch.triu unfortunately.
def triu_onnx(x, diagonal=0):
    l, w = x.shape
    arange_rows = torch.arange(l, device=x.device)

    arange_cols = torch.arange(w, device=x.device)
    mask = arange_cols.expand(l, w)

    arange_rows = arange_rows[:, None] + diagonal
    mask = mask >= arange_rows
    return x.masked_fill(mask == 0, 0)


def patched_build_delay_pattern_mask(self, input_ids: torch.Tensor, pad_token_id: int, max_length: int = None):
    # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
    input_ids = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
    bsz, num_codebooks, seq_len = input_ids.shape

    max_length = max_length if max_length is not None else self.generation_config.max_length
    input_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1

    channel_codebooks = num_codebooks // 2 if self.config.audio_channels == 2 else num_codebooks
    # we only apply the mask if we have a large enough seq len - otherwise we return as is
    if max_length < 2 * channel_codebooks - 1:
        raise NotImplementedError("Not supported in ONNX export. Please open an issue in Optimum repository.")

    # fill the shifted ids with the prompt entries, offset by the codebook idx
    for codebook in range(channel_codebooks):
        if self.config.audio_channels == 1:
            # mono channel - loop over the codebooks one-by-one
            input_ids_shifted[:, codebook, codebook : seq_len + codebook] = input_ids[:, codebook]
        else:
            # left/right channels are interleaved in the generated codebooks, so handle one then the other
            input_ids_shifted[:, 2 * codebook, codebook : seq_len + codebook] = input_ids[:, 2 * codebook]
            input_ids_shifted[:, 2 * codebook + 1, codebook : seq_len + codebook] = input_ids[:, 2 * codebook + 1]

    # construct a pattern mask that indicates the positions of padding tokens for each codebook
    # first fill the upper triangular part (the EOS padding)
    # NOTE: We could use torch.bool here, but PyTorch the complains with `The exported ONNX model failed ONNX shape inference.`
    # Using int8 leads to `Could not find an implementation for Where`
    delay_pattern = triu_onnx(
        torch.ones((channel_codebooks, max_length), dtype=torch.int32), diagonal=max_length - channel_codebooks + 1
    )

    # NOTE: We could use torch.bool here, but PyTorch the complains with `The exported ONNX model failed ONNX shape inference.`
    # Using int32 leads to `Could not find an implementation for Trilu`, hence int64 here

    # then fill the lower triangular part (the BOS padding)
    delay_pattern = delay_pattern + torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.int64))
    delay_pattern = delay_pattern.to(torch.bool)

    if self.config.audio_channels == 2:
        # for left/right channel we need to duplicate every row of the pattern mask in an interleaved fashion
        delay_pattern = delay_pattern.repeat_interleave(2, dim=0)

    mask = ~delay_pattern.to(input_ids.device)
    input_ids = mask * input_ids_shifted + ~mask * pad_token_id

    # find the first position to start generating - this is the first place we have the -1 token
    # and will always be in the first codebook (since it has no codebook offset)
    first_codebook_ids = input_ids[:, 0, :]
    start_ids = (first_codebook_ids == -1).nonzero()[:, 1]

    # TODO: Is this OK?
    first_start_id = start_ids.min()

    # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
    pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
    input_ids_edited = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
    return {"input_ids_edited": input_ids_edited, "delay_pattern_mask": pattern_mask}


class MusicgenModelPatcher(Seq2SeqModelPatcher):
    def __enter__(self):
        self.patch_ops()
        if self.real_config.model_part == "build_delay_pattern_mask":
            # For build_delay_pattern_mask, we need to override the signature too.
            self._model.forward = types.MethodType(patched_build_delay_pattern_mask, self._model)
        else:
            setattr(self._model, self.orig_forward_name, self.patched_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        if self.real_config.model_part == "build_delay_pattern_mask":
            self._model.forward = self.original_decoder_forward
        else:
            setattr(self._model, self.orig_forward_name, self.orig_forward)

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        if config.model_part == "build_delay_pattern_mask":
            self.original_decoder_forward = self.orig_forward
        elif config.model_part == "encodec_decode":
            # EncodecModel.forward -> EncodecModel.decode
            @functools.wraps(self.orig_forward)
            def patched_forward(
                input_values: Optional["torch.Tensor"] = None,
                padding_mask: Optional["torch.Tensor"] = None,
                audio_codes: Optional["torch.Tensor"] = None,
                bandwidth: Optional[float] = None,
                audio_scales: Optional["torch.Tensor"] = None,
                return_dict: Optional[bool] = None,
            ):
                chunk_length = self.real_config._config.audio_encoder.chunk_length
                if chunk_length is None:
                    if audio_scales is not None:
                        audio_scales = audio_scales[0]

                    if len(audio_codes) != 1:
                        raise ValueError(f"Expected one frame, got {len(audio_codes)}")
                    audio_values = self._model._decode_frame(audio_codes[0], audio_scales)
                else:
                    raise ValueError("Not supported, a meaningful error should have been raised ahead.")
                    decoded_frames = []

                    for frame, scale in zip(audio_codes, audio_scales):
                        frames = self._model._decode_frame(frame, scale)
                        decoded_frames.append(frames)

                    audio_values = self._model._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)

                # truncate based on padding mask
                if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
                    audio_values = audio_values[..., : padding_mask.shape[-1]]

                return {"audio_values": audio_values}

            self.patched_forward = patched_forward


class CLIPModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_transformers_version(">=", "4.43") and is_transformers_version("<", "4.48"):
            self.original_sdpa_forward = CLIPSdpaAttention.forward
            CLIPSdpaAttention.forward = CLIPAttention.forward

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version(">=", "4.43") and is_transformers_version("<", "4.48"):
            CLIPSdpaAttention.forward = self.original_sdpa_forward


class VitPoseModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Set dataset_index (defaulting to COCO=0), otherwise we will get an error like:
        # ValueError: dataset_index must be provided when using multiple experts (num_experts=6). Please provide dataset_index to the forward pass.
        if model.config.backbone_config.num_experts > 1:
            model_kwargs["dataset_index"] = torch.tensor(0, device=model.device)

        super().__init__(config, model, model_kwargs)


# https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L228
def qwen3_moe_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # TODO: we loop over all possible experts instead of hitted ones to avoid issues in graph execution.
    # expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class Qwen3MoeModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_transformers_version(">=", "4.53"):
            self.original_moe_forward = Qwen3MoeSparseMoeBlock.forward
            Qwen3MoeSparseMoeBlock.forward = qwen3_moe_forward_patched

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version(">=", "4.53"):
            Qwen3MoeSparseMoeBlock.forward = self.original_moe_forward
