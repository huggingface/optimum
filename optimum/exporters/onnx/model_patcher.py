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
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

from ...configuration_utils import _transformers_version
from ...utils import logging


if _transformers_version > version.parse("4.34.99"):
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
if _transformers_version >= version.parse("4.36"):
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
else:
    _prepare_4d_causal_attention_mask = None
    _prepare_4d_causal_attention_mask_for_sdpa = None
    AttentionMaskConverter = None

if _transformers_version >= version.parse("4.42"):
    from transformers.cache_utils import SlidingWindowCache, StaticCache

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


class ModelPatcher:
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._model = model

        patching_specs = config.PATCHING_SPECS
        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
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

            outputs = self.orig_forward(*args, **kwargs)

            # This code block handles different cases of the filterd_outputs input to align it with the expected
            # format of outputs. It is common for the output type of a model to vary, such as tensor, list,
            # tuple, etc. For Transformers models, the output is encapsulated in a ModelOutput object that
            # contains the output names of the model. In the case of Timm classification models, the output
            # is of type tensor. By default, it is assumed that the output names mentioned in the ONNX config
            # match the outputs in order.
            filterd_outputs = {}
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                    if (
                        onnx_output_name in config.outputs
                        or (allow_past_in_outputs and name.startswith("past_key_values"))
                        or any(key.startswith(onnx_output_name) for key in config.outputs.keys())
                    ):
                        filterd_outputs[name] = value
            elif isinstance(outputs, (list, tuple)):
                outputs_list = list(config.outputs.keys())
                dict(zip(outputs_list, outputs))
            else:
                if len(config.outputs) > 1:
                    num_outputs = len(config.outputs)
                    outputs_str = ", ".join(config.outputs.keys())
                    raise ValueError(
                        f"config.outputs should have only one outputs, but it has {num_outputs} keys: {outputs_str}"
                    )
                else:
                    name = list(config.outputs.keys())[0]
                    filterd_outputs[name] = outputs
                name = list(config.outputs.keys())[0]
                filterd_outputs[name] = outputs
            return filterd_outputs

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

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)

    def __call__(self, *args, **kwargs):
        if getattr(self._model, self.orig_forward_name) is self.orig_forward:
            logger.warning("Running the non-patched model")
        return self._model(*args, **kwargs)


class Seq2SeqModelPatcher(ModelPatcher):
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

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            outputs = self.orig_forward(*args, **kwargs)

            # Filter out cross attention past key values output from the decoder using KV cache, as they are constants.
            filterd_outputs = {}
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
                            filterd_outputs[name] = value
                    else:
                        if self.real_config._behavior == "monolith" or (
                            self.real_config._behavior == "decoder"
                            and (self.real_config.is_merged or not self.real_config.use_past_in_inputs)
                        ):
                            filterd_outputs[name] = value
                        elif self.real_config._behavior == "decoder" and self.real_config.use_past_in_inputs:
                            # The filtering happens here. The decoder with use_past_in_inputs=True corresponds to the autoregressive one.
                            filterd_outputs[name] = tuple([v[:2] for v in value])
            return filterd_outputs

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


def _unmask_unattended_patched(
    expanded_mask: torch.Tensor,
    min_dtype: float,
):
    return expanded_mask


def _make_causal_mask_patched(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # We add self in the signature because `self._make_causal_mask` is used elsewhere in the class definition, despite the method being a staticmethod.
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1

        # NOTE: adding dtype=torch.int64 here for triu to be supported by ORT: https://github.com/microsoft/onnxruntime/issues/16189
        context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int64), diagonal=diagonal)
        mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


_make_causal_mask_patched_staticmethod = staticmethod(_make_causal_mask_patched)
_unmask_unattended_patched_staticmethod = staticmethod(_unmask_unattended_patched)


# Adapted from _prepare_4d_causal_attention_mask
def _prepare_4d_causal_attention_mask_for_sdpa_patched(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.

    In case no token is masked in the `attention_mask` argument, we simply set it to `None` for the cases `query_length == 1` and
    `key_value_length == query_length`, and rely instead on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    # NOTE: For the ONNX export we remove the setting of attention_mask to None in some specific cases, and we do NOT call _unmask_unattended
    # that can not be exported to ONNX and is very specific to PyTorch memory-efficient attention backend anyway.

    return attention_mask


class DecoderModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()
        if AttentionMaskConverter is not None:
            # TODO: Remove this _make_causal_mask patch if once transformers if much above 4.35
            AttentionMaskConverter._make_causal_mask = _make_causal_mask_patched_staticmethod

            if _transformers_version >= version.parse("4.36"):
                AttentionMaskConverter._unmask_unattended = _unmask_unattended_patched_staticmethod

        if _transformers_version >= version.parse("4.36"):
            patch_everywhere(
                "_prepare_4d_causal_attention_mask_for_sdpa", _prepare_4d_causal_attention_mask_for_sdpa_patched
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if AttentionMaskConverter is not None:
            # TODO: Remove this _make_causal_mask patch if once transformers if much above 4.35
            AttentionMaskConverter._make_causal_mask = staticmethod(self.original_make_causal)

            if _transformers_version >= version.parse("4.36"):
                AttentionMaskConverter._unmask_unattended = staticmethod(self.original_unmask_unattended)

        if _transformers_version >= version.parse("4.36"):
            patch_everywhere(
                "_prepare_4d_causal_attention_mask_for_sdpa", self.original_prepare_4d_causal_attention_mask_for_sdpa
            )

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        if _transformers_version >= version.parse("4.36"):
            self.original_prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa
            self.original_unmask_unattended = AttentionMaskConverter._unmask_unattended

        # TODO: Remove this if once transformers if much above 4.35
        if AttentionMaskConverter is not None:
            self.original_make_causal = AttentionMaskConverter._make_causal_mask


def falcon_build_alibi_tensor_patched(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    # NOTE: remove the .bfloat16() cast here as PyTorch ONNX export rather casts to complex128 if this is used, resulting in a onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph error.
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


class FalconModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        self.patch_ops()

        if self.real_config.task == "text-generation":
            patch_everywhere(
                "build_alibi_tensor",
                falcon_build_alibi_tensor_patched,
                module_name_prefix="transformers.models.falcon.modeling_falcon",
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.restore_ops()

        setattr(self._model, self.orig_forward_name, self.orig_forward)

        if self.real_config.task == "text-generation":
            patch_everywhere(
                "build_alibi_tensor",
                self.build_alibi_tensor_original,
                module_name_prefix="transformers.models.falcon.modeling_falcon",
            )

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)
        self.build_alibi_tensor_original = transformers.models.falcon.modeling_falcon.build_alibi_tensor


class WavLMModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            model_kwargs = self.model_kwargs
            # setting output_attentions=True in the model input to avoid calling torch.nn.functional.scaled_dot_product_attention
            # in https://github.com/huggingface/transformers/blob/v4.27.1/src/transformers/models/wavlm/modeling_wavlm.py#L496
            # that calls https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/functional.py#L5334
            model_kwargs["output_attentions"] = True
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=model_kwargs)

            outputs = self.orig_forward(*args, **kwargs)

            filterd_outputs = {}
            for name, value in outputs.items():
                onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                if (
                    onnx_output_name in config.outputs
                    or (allow_past_in_outputs and name.startswith("past_key_values"))
                    or any(key.startswith(onnx_output_name) for key in config.outputs.keys())
                ):
                    filterd_outputs[name] = value
            return filterd_outputs

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
    def __enter__(self):
        super().__enter__()
        if _transformers_version >= version.parse("4.42") and self.real_config._config.model_type == "mistral":
            self._model[0].auto_model._update_causal_mask = types.MethodType(
                _update_causal_mask_patched, self._model[0].auto_model
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if _transformers_version >= version.parse("4.42") and self.real_config._config.model_type == "mistral":
            self._model[0].auto_model._update_causal_mask = types.MethodType(
                self._update_causal_mask_original, self._model[0].auto_model
            )

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        if _transformers_version >= version.parse("4.42") and self.real_config._config.model_type == "mistral":
            self._update_causal_mask_original = self._model[0].auto_model._update_causal_mask

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


def _update_causal_mask_patched(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    use_cache: bool,
    output_attentions: bool,
):
    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if self._attn_implementation == "flash_attention_2":
        if attention_mask is not None and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.

    # cache_position must be valid here no matter which cache we use
    past_seen_tokens = cache_position[0] if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)
    using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

    if (
        self.config._attn_implementation == "sdpa"
        and not (using_static_cache or using_sliding_window_cache)
        and not output_attentions
    ):
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            sliding_window=self.config.sliding_window,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    # SlidingWindowCache
    if using_sliding_window_cache:
        target_length = max(sequence_length, self.config.sliding_window)
    # StaticCache
    elif using_static_cache:
        target_length = past_key_values.get_max_length()
    # DynamicCache or no cache
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        exclude_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        if self.config.sliding_window is not None:
            if not using_sliding_window_cache or sequence_length > self.config.sliding_window:
                # ---------------- NOTE: This part is patched -----------------------------
                exclude_mask = torch.bitwise_or(
                    exclude_mask,
                    torch.arange(target_length, device=device)
                    <= (cache_position.reshape(-1, 1) - self.config.sliding_window),
                )
                # ---------------- NOTE: patch end ----------------------------------------

        causal_mask *= exclude_mask
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

    # if (
    #     self.config._attn_implementation == "sdpa"
    #     and attention_mask is not None
    #     and attention_mask.device.type == "cuda"
    #     and not output_attentions
    # ):
    #     # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    #     # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    #     # Details: https://github.com/pytorch/pytorch/issues/110213
    #     causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


class MistralModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()
        if AttentionMaskConverter is not None:
            # TODO: Remove this _make_causal_mask patch if once transformers if much above 4.35
            AttentionMaskConverter._make_causal_mask = _make_causal_mask_patched_staticmethod

            if _transformers_version >= version.parse("4.36"):
                AttentionMaskConverter._unmask_unattended = _unmask_unattended_patched_staticmethod

        if _transformers_version >= version.parse("4.36"):
            patch_everywhere(
                "_prepare_4d_causal_attention_mask_for_sdpa", _prepare_4d_causal_attention_mask_for_sdpa_patched
            )

        if _transformers_version >= version.parse("4.42"):
            if hasattr(self._model, "model"):
                self._model.model._update_causal_mask = types.MethodType(
                    _update_causal_mask_patched, self._model.model
                )
            else:
                self._model._update_causal_mask = types.MethodType(_update_causal_mask_patched, self._model)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if AttentionMaskConverter is not None:
            # TODO: Remove this _make_causal_mask patch if once transformers if much above 4.35
            AttentionMaskConverter._make_causal_mask = staticmethod(self.original_make_causal)

            if _transformers_version >= version.parse("4.36"):
                AttentionMaskConverter._unmask_unattended = staticmethod(self.original_unmask_unattended)

        if _transformers_version >= version.parse("4.36"):
            patch_everywhere(
                "_prepare_4d_causal_attention_mask_for_sdpa", self.original_prepare_4d_causal_attention_mask_for_sdpa
            )

        if _transformers_version >= version.parse("4.42"):
            if hasattr(self._model, "model"):
                self._model.model._update_causal_mask = types.MethodType(
                    self._update_causal_mask_original, self._model.model
                )
            else:
                self._model._update_causal_mask = types.MethodType(self._update_causal_mask_original, self._model)

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        if _transformers_version >= version.parse("4.36"):
            self.original_prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa
            self.original_unmask_unattended = AttentionMaskConverter._unmask_unattended

        # TODO: Remove this if once transformers if much above 4.35
        if AttentionMaskConverter is not None:
            self.original_make_causal = AttentionMaskConverter._make_causal_mask

        if _transformers_version >= version.parse("4.42"):
            if hasattr(self._model, "model"):
                self._update_causal_mask_original = self._model.model._update_causal_mask
            else:
                self._update_causal_mask_original = self._model._update_causal_mask
