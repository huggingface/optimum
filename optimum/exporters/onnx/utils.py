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
"""Utility functions."""

import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import packaging
import torch
from transformers.utils import is_tf_available, is_torch_available

from ...utils import (
    DIFFUSERS_MINIMUM_VERSION,
    ORT_QUANTIZE_MINIMUM_VERSION,
    check_if_diffusers_greater,
    is_diffusers_available,
    logging,
)
from ...utils.import_utils import _diffusers_version, check_if_torch_greater
from ..tasks import TasksManager
from .constants import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME


logger = logging.get_logger()


if is_diffusers_available():
    if not check_if_diffusers_greater(DIFFUSERS_MINIMUM_VERSION.base_version):
        raise ImportError(
            f"We found an older version of diffusers {_diffusers_version} but we require diffusers to be >= {DIFFUSERS_MINIMUM_VERSION}. "
            "Please update diffusers by running `pip install --upgrade diffusers`"
        )
    from diffusers.models.cross_attention import CrossAttnProcessor

if TYPE_CHECKING:
    from .base import OnnxConfig

    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin, StableDiffusionPipeline


def check_onnxruntime_requirements(minimum_version: packaging.version.Version):
    """
    Checks that ONNX Runtime is installed and if version is recent enough.

    Args:
        minimum_version (`packaging.version.Version`):
            The minimum version allowed for the onnxruntime package.

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "ONNX Runtime doesn't seem to be currently installed. "
            "Please install ONNX Runtime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )

    ort_version = packaging.version.parse(onnxruntime.__version__)
    if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
        raise ImportError(
            f"We found an older version of ONNX Runtime ({onnxruntime.__version__}) "
            f"but we require the version to be >= {minimum_version} to enable all the conversions options.\n"
            "Please update ONNX Runtime by running `pip install --upgrade onnxruntime`"
        )


def get_encoder_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], config: "OnnxConfig"
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "OnnxConfig"]]:
    """
    Returns the encoder and decoder parts of the model and their subsequent onnx configs.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the encoder and decoder parts of the model.
    """
    models_for_export = {}

    encoder_model = model.get_encoder()
    encoder_onnx_config = config.with_behavior("encoder")
    models_for_export[ONNX_ENCODER_NAME] = (encoder_model, encoder_onnx_config)

    decoder_onnx_config = config.with_behavior("decoder", use_past=False)
    models_for_export[ONNX_DECODER_NAME] = (model, decoder_onnx_config)

    if config.use_past:
        decoder_onnx_config_with_past = config.with_behavior("decoder", use_past=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (model, decoder_onnx_config_with_past)

    return models_for_export


def get_decoder_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: "OnnxConfig",
) -> Dict[str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel"], "OnnxConfig"]]:
    """
    Returns two versions of the decoder that can be used together to perform fast generation:

        1. The first one takes regular inputs, and outputs the result along with past key/values.
        2. The second one takes regular inputs and past key/values, and outputs the result along with the updated past
        key/values.


    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[PreTrainedModel, TFPreTrainedModel], OnnxConfig]]: A Dict containing the model and
        onnx configs for the encoder and decoder parts of the model.
    """
    models_for_export = {}

    onnx_config = config.__class__(
        model.config, task=config.task, use_past_in_inputs=False, use_present_in_outputs=True
    )
    models_for_export[ONNX_DECODER_NAME] = (model, onnx_config)

    if config.use_past:
        onnx_config_with_past = config.__class__(model.config, task=config.task, use_past=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (model, onnx_config_with_past)

    return models_for_export


def get_stable_diffusion_models_for_export(
    pipeline: "StableDiffusionPipeline",
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "OnnxConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent onnx configs.

    Args:
        pipeline ([`StableDiffusionPipeline`]):
            The model to export.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the different components of the model.
    """
    models_for_export = {}
    # To be applied for torch v2.0.0 > x > v2.0.2
    if check_if_torch_greater("2.0.0") and not check_if_torch_greater("2.0.2"):
        register_custom_scaled_dot_product_attention_export()

    # Text encoder
    text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.text_encoder, exporter="onnx", task="feature-extraction"
    )
    text_encoder_onnx_config = text_encoder_config_constructor(pipeline.text_encoder.config)
    models_for_export["text_encoder"] = (pipeline.text_encoder, text_encoder_onnx_config)

    # U-NET
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.unet, exporter="onnx", task="semantic-segmentation", model_type="unet"
    )
    unet_onnx_config = onnx_config_constructor(pipeline.unet.config)

    # PyTorch does not support the ONNX export of torch.nn.functional.scaled_dot_product_attention
    models_for_export["unet"] = (pipeline.unet, unet_onnx_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample)["latent_dist"].sample()}
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder, exporter="onnx", task="semantic-segmentation", model_type="vae-encoder"
    )
    vae_onnx_config = vae_config_constructor(vae_encoder.config)
    models_for_export["vae_encoder"] = (vae_encoder, vae_onnx_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    if hasattr(vae_decoder.decoder.mid_block.attentions[0], "_use_2_0_attn"):
        vae_decoder.decoder.mid_block.attentions[0]._use_2_0_attn = False
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder, exporter="onnx", task="semantic-segmentation", model_type="vae-decoder"
    )
    vae_onnx_config = vae_config_constructor(vae_decoder.config)
    models_for_export["vae_decoder"] = (vae_decoder, vae_onnx_config)

    return models_for_export


def recursive_to_device(value: Union[Tuple, List, "torch.Tensor"], device: str):
    if isinstance(value, tuple):
        value = list(value)
        for i, val in enumerate(value):
            value[i] = recursive_to_device(val, device)
        value = tuple(value)
    elif isinstance(value, list):
        for i, val in enumerate(value):
            value[i] = recursive_to_device(val, device)
    elif isinstance(value, torch.Tensor):
        value = value.to(device)

    return value


def recursive_to_dtype(
    value: Union[Tuple, List, "torch.Tensor"], dtype: Optional[torch.dtype], start_dtype: Optional[torch.dtype] = None
):
    if dtype is None:
        return value

    if isinstance(value, tuple):
        value = list(value)
        for i, val in enumerate(value):
            value[i] = recursive_to_dtype(val, dtype)
        value = tuple(value)
    elif isinstance(value, list):
        for i, val in enumerate(value):
            value[i] = recursive_to_dtype(val, dtype)
    elif isinstance(value, torch.Tensor):
        if start_dtype is None or (start_dtype is not None and value.dtype == start_dtype):
            value = value.to(dtype=dtype)

    return value


def register_custom_scaled_dot_product_attention_export():
    @torch.onnx.symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v")
    def scaled_dot_product_attention(
        g: torch.onnx._internal.jit_utils.GraphContext,
        query: torch._C.Value,
        key: torch._C.Value,
        value: torch._C.Value,
        attn_mask: Optional[torch._C.Value] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[torch._C.Value] = None,
    ):
        assert (not is_causal) or (
            is_causal and torch.onnx.symbolic_helper._is_none(attn_mask)
        ), "is_causal and attn_mask cannot be set at the same time"

        scale = torch.onnx.symbolic_helper._maybe_get_const(scale, "f")
        if scale is None:
            scale = _attention_scale(g, query)

        if is_causal:
            attn_mask = _causal_attention_mask(g, query, key)
        key_shape_builtin = torch.onnx.symbolic_helper._get_tensor_rank(key)
        key_transposed_axes = list(range(key_shape_builtin))
        key_transposed_axes[-1], key_transposed_axes[-2] = (
            key_transposed_axes[-2],
            key_transposed_axes[-1],
        )
        key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)
        query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
        key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
        mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)
        if attn_mask is None or torch.onnx.symbolic_helper._is_none(attn_mask):
            mul_qk_add = mul_qk
        elif torch.onnx._type_utils.JitScalarType.from_value(attn_mask) == torch.onnx._type_utils.JitScalarType.BOOL:
            # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
            const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
            const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
            attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
            mul_qk_add = g.op("Add", mul_qk, attn_mask)
        elif torch.onnx._type_utils.JitScalarType.from_value(attn_mask) == torch.onnx._type_utils.JitScalarType.FLOAT:
            mul_qk_add = g.op("Add", mul_qk, attn_mask)
        else:
            raise ValueError(
                f"Unsupported type for attn_mask: {torch.onnx._type_utils.JitScalarType.from_value(attn_mask)}"
            )

        attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)

        if dropout_p != 0:
            attn_weight = g.op(
                "Dropout",
                attn_weight,
                g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
            )

        return g.op("MatMul", attn_weight, value)

    def _attention_scale(g: torch.onnx._internal.jit_utils.GraphContext, query: torch._C.Value) -> torch._C.Value:
        """Calculate the scale factor for the attention result.
        Args:
            query: Tensor of shape [..., L, E]
        Returns:
            Scalar scale factor := 1 / math.sqrt(query.size(-1))
        """
        query_shape = g.op("Shape", query)
        query_shape_last = g.op(
            "Slice",
            query_shape,
            g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
            g.op("Constant", value_t=torch.tensor([torch.onnx._constants.INT64_MAX], dtype=torch.int64)),
        )
        embedding_size = g.op(
            "Cast",
            query_shape_last,
            to_i=torch.onnx._type_utils.JitScalarType.from_value(query).onnx_type(),
        )
        const_one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float))
        scale = g.op("Div", const_one, g.op("Sqrt", embedding_size))
        return scale

    def _causal_attention_mask(
        g: torch.onnx._internal.jit_utils.GraphContext, query: torch._C.Value, key: torch._C.Value
    ) -> torch._C.Value:
        """Create a causal mask for the given query and key tensors.
        Equivalent to::
            mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_mask = torch.zeros(L, S, dtype=torch.float)
            attn_mask = attn_mask.masked_fill(not mask, -float('inf'))
        Args:
            query: Tensor of shape [..., L, E]
            key: Tensor of shape [..., S, E]
        Returns:
            Tensor of shape [L, S]
        """

        query_shape = g.op("Shape", query)
        key_shape = g.op("Shape", key)

        last_idx = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        second_last_idx = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
        target_length = g.op("Slice", query_shape, second_last_idx, last_idx)
        source_length = g.op("Slice", key_shape, second_last_idx, last_idx)
        # attn_mask = torch.ones(L, S) := {
        size = g.op("Concat", target_length, source_length, axis_i=0)
        const_one = g.op("Constant", value_t=torch.tensor([1.0]))
        attn_mask = g.op("Expand", const_one, size)
        # }
        attn_mask = g.op("Trilu", attn_mask, upper_i=0)
        # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", g.op("Equal", attn_mask, const_zero), const_neg_inf, const_zero)
        return attn_mask

    torch.onnx.register_custom_op_symbolic(
        "aten::scaled_dot_product_attention", scaled_dot_product_attention, opset_version=14
    )
