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
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

from packaging import version
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.falcon.modeling_falcon import build_alibi_tensor
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

from ...configuration_utils import _transformers_version
from ...utils import logging


if _transformers_version > version.parse("4.34.99"):
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
else:
    _prepare_4d_causal_attention_mask = None
    AttentionMaskConverter = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel

    from .base import OnnxConfig

logger = logging.get_logger(__name__)


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


def falcon_model_forward_without_kv_reformatting(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    # TODO: We may remove this patch once https://github.com/huggingface/transformers/pull/26933 is merged & released in Transformers.

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    # NOTE: here we removed the _convert_to_rw_cache call

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    hidden_states = inputs_embeds

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    # Compute alibi tensor: check build_alibi_tensor documentation
    past_key_values_length = 0
    if past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[1]  # 1 because RW-cache, not standard format
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=hidden_states.device)
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    if self.use_alibi:
        alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
    else:
        alibi = None
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            # NOTE: here we use expand(batch_size, -1) instead of transformers view(-1, seq_length) that is bugged
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

    # 4d mask is passed through the layers
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask[i],
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi=alibi,
        )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # NOTE: here we removed the _convert_cache_to_standard_format call

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


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


_make_causal_mask_patched = staticmethod(_make_causal_mask_patched)


class DecoderModelPatcher(ModelPatcher):
    def __enter__(self):
        # TODO: Remove this if once transformers if much above 4.35
        if AttentionMaskConverter is not None:
            AttentionMaskConverter._make_causal_mask = _make_causal_mask_patched

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO: Remove this if once transformers if much above 4.35
        # TODO: We should unpatch it - however `self._make_causal_mask` may still be called later which raises issues with this simple patch strategy.
        # We need to find a proper solution.
        # if AttentionMaskConverter is not None:
        #     AttentionMaskConverter._make_causal_mask = self.original_make_causal
        pass

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        # TODO: Remove this if once transformers if much above 4.35
        if AttentionMaskConverter is not None:
            self.original_make_causal = AttentionMaskConverter._make_causal_mask


class FalconModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        self.patch_ops()

        if self.real_config.task == "text-generation":
            self._model.transformer.forward = types.MethodType(
                falcon_model_forward_without_kv_reformatting, self._model.transformer
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.restore_ops()

        setattr(self._model, self.orig_forward_name, self.orig_forward)

        if self.real_config.task == "text-generation":
            self._model.transformer.forward = types.MethodType(
                self.original_model_transformer_forward, self._model.transformer
            )

    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        if config.task == "text-generation":
            self.original_model_transformer_forward = model.transformer.forward

        self._model = model

        self.orig_forward_name = "forward" if hasattr(self._model, "forward") else "call"
        self.orig_forward = getattr(self._model, self.orig_forward_name)

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            model_kwargs = self.model_kwargs
            # setting output_attentions=True in the model input to avoid calling torch.nn.functional.scaled_dot_product_attention
            # in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/falcon/modeling_falcon.py#L425
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
            image_embeddings=None,
            image_positional_embeddings=None,
            return_dict=True,
            **kwargs,
        ):
            if config.variant == "monolith":
                return self.orig_forward(
                    pixel_values=pixel_values,
                    input_points=input_points,
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
                    if input_points is not None:
                        input_labels = torch.ones_like(
                            input_points[:, :, :, 0], dtype=torch.int, device=input_points.device
                        )
                    else:
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
