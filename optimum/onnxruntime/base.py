#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Defines the base classes that are used to perform inference with ONNX Runtime of Transformers models."""

from abc import abstractmethod
from typing import Dict, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from onnxruntime import InferenceSession

from ..utils import NormalizedConfigManager
from ..utils.logging import warn_once
from .io_binding import TypeHelper
from .modeling_ort import ORTModel
from .utils import get_ordered_input_names, logging


logger = logging.get_logger(__name__)


class ORTModelPart:
    """
    For multi-file ONNX models, such as encoder-decoder models, represents a part of the model.
    It has its own `onnxruntime.InferenceSession`, and can perform a forward pass.
    """

    _prepare_onnx_inputs = ORTModel._prepare_onnx_inputs
    _prepare_onnx_outputs = ORTModel._prepare_onnx_outputs

    def __init__(
        self,
        session: InferenceSession,
        parent_model: "ORTModel",
    ):
        self.session = session
        self.parent_model = parent_model
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(
            self.parent_model.config.model_type
        )(self.parent_model.config)
        self.main_input_name = self.parent_model.main_input_name
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.input_dtypes = {input_key.name: input_key.type for input_key in session.get_inputs()}
        self.output_dtypes = {output_key.name: output_key.type for output_key in session.get_outputs()}

        self._ordered_input_names = get_ordered_input_names(self.input_names.keys(), func=self.forward)

    @property
    def device(self):
        return self.parent_model.device

    @property
    def dtype(self):
        for dtype in self.input_dtypes.values():
            torch_dtype = TypeHelper.ort_type_to_torch_type(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = TypeHelper.ort_type_to_torch_type(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTEncoder(ORTModelPart):
    """
    Encoder part of the encoder-decoder model for ONNX Runtime inference.
    """

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.parent_model.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.parent_model.use_io_binding:
            model_inputs = [input_ids]
            if "attention_mask" in self.input_names:
                model_inputs.append(attention_mask)
            io_binding, output_shapes, output_buffers = self.parent_model._prepare_io_binding(
                self.session,
                *model_inputs,
                ordered_input_names=self._ordered_input_names,
            )

            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoderForSeq2Seq(ORTModelPart):
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.
    """

    def __init__(
        self,
        session: InferenceSession,
        parent_model: "ORTModel",
    ):
        super().__init__(session, parent_model)

        # TODO: make this less hacky.
        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]

        # To handle the old case when past_key_values were following the format: past_key_values_{idx}
        if len(self.key_value_input_names) == 0:
            self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        if len(self.key_value_output_names) == 0:
            self.key_value_output_names = [key for key in self.output_names if "key_values" in key]

        if self.parent_model.use_cache is True and len(self.key_value_output_names) == 0:
            raise RuntimeError("Could not find the past key values in the provided model.")

        self.use_past_in_outputs = len(self.key_value_output_names) > 0
        self.use_past_in_inputs = len(self.key_value_input_names) > 0
        self.use_fp16 = False
        for inp in session.get_inputs():
            if "past_key_values" in inp.name and inp.type == "tensor(float16)":
                self.use_fp16 = True
                break

        # We may use ORTDecoderForSeq2Seq for vision-encoder-decoder models, where models as gpt2
        # can be used but do not support KV caching for the cross-attention key/values, see:
        # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/gpt2/modeling_gpt2.py#L302-L311
        # This attribute is used to avoid returning cross-attention KV-cache in this case.
        self.no_cross_attention_cache = getattr(self.parent_model, "no_cross_attention_cache", False)

        if (not self.parent_model.use_merged and self.use_past_in_inputs) or self.no_cross_attention_cache:
            self.num_pkv = 2
        else:
            # When using a merged model, we always have the same number of output whether we use past key values or not,
            # and in the case past key values are used, empty tensors are given as cross-attention past key values as they
            # are constants
            self.num_pkv = 4

        self.past_key_values_cross_attention_output_names = set()
        for output_name in self.output_names:
            if output_name.startswith("present") and "encoder" in output_name:
                self.past_key_values_cross_attention_output_names.add(output_name)

        self.use_legacy_outputs = (
            self.parent_model.use_merged is False and len(self.past_key_values_cross_attention_output_names) > 0
        )

    def compute_past_key_values_output_shapes(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        use_cache_branch: Optional[bool],
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Dict[str, int]:
        batch_size = input_ids.size(0)

        num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads

        sequence_length = input_ids.size(1)
        encoder_sequence_length = encoder_hidden_states.size(1)
        if past_key_values is not None and use_cache_branch is not False:
            # Here, use_cache_branch may be None in the case of separate decoder without/with past, or True if the with past branch
            # of a merged decoder is used
            sequence_length += past_key_values[0].size(2)

        self_attn_shape = (batch_size, num_attention_heads, sequence_length, embed_size_per_head)

        if past_key_values is not None and use_cache_branch is True:
            cross_attn_shape = (0, num_attention_heads, 1, embed_size_per_head)
        else:
            cross_attn_shape = (batch_size, num_attention_heads, encoder_sequence_length, embed_size_per_head)

        past_key_values_shapes = {}
        for idx, name in enumerate(self.key_value_output_names):
            is_self_attn = idx % 4 < 2
            # decoder with past does not ouput cross attention key/values as they are constants
            past_key_values_shapes[name] = self_attn_shape if (is_self_attn or self.num_pkv == 2) else cross_attn_shape
        return past_key_values_shapes

    def get_outputs_not_to_bind(self, use_merged_cache: bool) -> Set[str]:
        result = {
            output_name
            for output_name in self.output_names
            if (not output_name.startswith("present") and output_name not in {"loss", "logits"})
        }
        if use_merged_cache is True:
            # When using a merged decoder and the use cache branch, we output 0-dim tensors that IO Binding do not support.
            # Therefore, we do not bind them.
            result = result.union(self.past_key_values_cross_attention_output_names)
        return result

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache_branch: None = None,
    ) -> Seq2SeqLMOutput:
        # Adding use_cache_branch in the signature here is just a hack for IO Binding

        use_torch = isinstance(input_ids, torch.Tensor)
        self.parent_model.raise_on_numpy_input_io_binding(use_torch)

        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )

        # no-ops if merged decoder is not used
        use_merged_no_cache = past_key_values is None and self.parent_model.use_merged
        use_merged_cache = past_key_values is not None and self.parent_model.use_merged
        use_cache_branch_tensor, past_key_values, cache_position = self.prepare_inputs_for_merged(
            input_ids, past_key_values, cache_position, use_torch=use_torch
        )

        if self.parent_model.use_io_binding:
            known_output_shapes = self.compute_past_key_values_output_shapes(
                input_ids,
                encoder_hidden_states,
                use_cache_branch=use_cache_branch_tensor.item() if use_cache_branch_tensor is not None else None,
                past_key_values=past_key_values,
            )

            outputs_to_not_bind = self.get_outputs_not_to_bind(use_merged_cache)

            # TODO: fix transformers generate to have contiguous input_ids here already
            # For an unknown reason, calling `contiguous()` here is necessary to not have errors
            # on CPU EP with batch size > 1, despite it being also called in _prepare_io_binding.g
            model_inputs = [input_ids.contiguous()]

            if "encoder_hidden_states" in self.input_names:
                model_inputs.append(encoder_hidden_states)

            if "decoder_attention_mask" in self.input_names:
                model_inputs.append(decoder_attention_mask)

            if "encoder_attention_mask" in self.input_names:
                model_inputs.append(encoder_attention_mask)

            if past_key_values is not None:
                model_inputs += past_key_values

            if "labels" in self.input_names:
                model_inputs.append(labels)
                known_output_shapes.update({"loss": []})

            if use_cache_branch_tensor is not None:
                model_inputs.append(use_cache_branch_tensor)

            if "cache_position" in self.input_names:
                model_inputs.append(cache_position)

            io_binding, output_shapes, output_buffers = self.parent_model._prepare_io_binding(
                self.session,
                *model_inputs,
                known_output_shapes=known_output_shapes,
                ordered_input_names=self._ordered_input_names,
                outputs_to_not_bind=outputs_to_not_bind,
            )

            # Set -1 for sequence_length as it could be larger than the real sequence_length
            for name, shape in output_shapes.items():
                if name in self.key_value_output_names:
                    output_shapes[name] = shape[:2] + (-1,) + shape[3:]

            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            out_past_key_values = ()
            for name in self.key_value_output_names:
                # TODO: this should be improved
                if name in self.past_key_values_cross_attention_output_names and use_merged_cache:
                    continue
                out_past_key_values += (output_buffers[name].view(output_shapes[name]),)

            logits = output_buffers["logits"].view(output_shapes["logits"])

            loss = None
            if "loss" in self.output_names:
                loss = output_buffers["loss"].view(output_shapes["loss"])

            if not self.use_past_in_outputs:
                out_past_key_values = None
            elif not self.use_past_in_inputs or use_merged_no_cache:
                out_past_key_values = tuple(
                    out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
                )
            else:
                if self.use_legacy_outputs is True:
                    msg = (
                        "For the decoder with past, using ONNX models outputting cross attention past key values"
                        " is deprecated and the support will be removed in optimum 2.0. We recommend exporting again the model"
                        " with optimum>=1.7.3."
                    )
                    warn_once(logger, msg=msg)
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                # grab the cross attention key/values from the inputs
                elif self.num_pkv == 2:
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                elif self.num_pkv == 4:
                    # despite num_pkv being 4, we did not bind the cross-attention output
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + 2] + past_key_values[2 * i + 2 : 2 * i + 4]
                        for i in range(0, len(out_past_key_values), 2)
                    )
                else:
                    raise ValueError("Unsupported num_pkv")
        else:
            model_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "decoder_attention_mask": decoder_attention_mask,
                "encoder_attention_mask": encoder_attention_mask,
                "use_cache_branch": use_cache_branch_tensor,
                "cache_position": cache_position,
                "labels": labels,
            }
            if past_key_values is not None:
                model_inputs.update(zip(self.key_value_input_names, past_key_values))

            onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

            # TODO: using a new variable out_past_key_values is memory inefficient,
            # past_key_values is not used anymore at this point
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            out_past_key_values = tuple(model_outputs[output_name] for output_name in self.key_value_output_names)

            loss = model_outputs.get("loss", None)
            logits = model_outputs["logits"]

            # TODO: this is extremely ugly and unreadable. What if cross-attention k/v change?
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to:
            # * 4 for the decoder without cache (k/v of self-attention + k/v of cross-attention)
            # * 2 for the decoder with cache (k/v of self-attention as cross-attention cache is constant)
            if not self.use_past_in_outputs:
                out_past_key_values = None
            elif not self.use_past_in_inputs or use_merged_no_cache or self.no_cross_attention_cache:
                out_past_key_values = tuple(
                    out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
                )
            else:
                if self.use_legacy_outputs is True:
                    msg = (
                        "For the decoder with past, using ONNX models outputting cross attention past key values"
                        " is deprecated and the support will be removed in optimum 2.0. We recommend exporting again the model"
                        " with optimum>=1.7.3."
                    )
                    warn_once(logger, msg=msg)
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                # grab the cross attention key/values from the inputs
                elif self.num_pkv == 2:
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                elif self.num_pkv == 4:
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + 2] + past_key_values[i + 2 : i + 4]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                else:
                    raise ValueError("Unsupported num_pkv")

        return Seq2SeqLMOutput(loss=loss, logits=logits, past_key_values=out_past_key_values)

    def prepare_inputs_for_merged(
        self,
        input_ids: Optional[Union[torch.LongTensor, np.ndarray]],
        past_key_values: Optional[Tuple[Union[torch.FloatTensor, np.ndarray]]],
        cache_position: Optional[Union[torch.Tensor, np.ndarray]],
        use_torch: bool,
    ):
        constructor = torch if use_torch is True else np

        if self.parent_model.use_merged:
            # Uses without/with branch of a merged decoder depending on whether real past key values are passed
            use_cache_branch_tensor = constructor.full((1,), past_key_values is not None)
            if use_torch and use_cache_branch_tensor is not None:
                use_cache_branch_tensor = use_cache_branch_tensor.to(self.device)
        else:
            use_cache_branch_tensor = None

        # Generate dummy past for the first forward if uses a merged decoder
        if self.parent_model.use_merged and past_key_values is None:
            batch_size = input_ids.shape[0]
            num_attention_heads = self.normalized_config.num_attention_heads
            embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads
            dtype = constructor.float16 if self.use_fp16 else constructor.float32
            shape = (batch_size, num_attention_heads, 1, embed_size_per_head)
            key_or_value = constructor.zeros(shape, dtype=dtype)

            if use_torch is True:
                key_or_value = key_or_value.to(self.device)

            past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))

        # Generate dummy position cache for the first forward if uses a merged decoder
        if self.parent_model.use_merged and cache_position is None:
            cache_position = constructor.zeros((1,), dtype=constructor.int64)
            if use_torch is True:
                cache_position = cache_position.to(self.device)

        return use_cache_branch_tensor, past_key_values, cache_position


class ORTDecoder(ORTDecoderForSeq2Seq):
    def __init__(self, *args, **kwargs):
        logger.warning(
            "The class `ORTDecoder` is deprecated and will be removed in optimum v1.15.0, please use `ORTDecoderForSeq2Seq` instead."
        )
        super().__init__(*args, **kwargs)
