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
from typing import Dict, List, Optional, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput

from onnxruntime import InferenceSession

from ..utils import NormalizedConfigManager


class ORTModelPart:
    """
    For multi-file ONNX models, such as encoder-decoder models, represents a part of the model.
    It has its own `onnxruntime.InferenceSession`, and can perform a forward pass.
    """

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

    @property
    def device(self):
        return self.parent_model.device

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
        if self.device.type == "cuda" and self.parent_model.use_io_binding:
            model_inputs = [input_ids]
            if "attention_mask" in self.input_names:
                model_inputs.append(attention_mask)
            io_binding, output_shapes, output_buffers = self.parent_model._prepare_io_binding(
                self.session, *model_inputs
            )

            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = {"input_ids": input_ids.cpu().detach().numpy()}

            # Add the attention_mask inputs when needed
            if "attention_mask" in self.input_names:
                onnx_inputs["attention_mask"] = attention_mask.cpu().detach().numpy()

            # Run inference
            outputs = self.session.run(None, onnx_inputs)
            last_hidden_state = torch.from_numpy(outputs[self.output_names["last_hidden_state"]]).to(self.device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoder(ORTModelPart):
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

        if len(self.key_value_output_names) == 0:
            raise RuntimeError("Could not find the past key values in the provided model.")

        # Attributes useful when computing the past key/values output shapes.
        self.expected_key_symbolic_shape = None
        self.expected_value_symbolic_shape = None
        for output in self.session.get_outputs():
            if ".key" in output.name:
                self.expected_key_symbolic_shape = output.shape
            elif ".value" in output.name:
                self.expected_value_symbolic_shape = output.shape
            # To handle the old case when past_key_values were following the format: past_key_values_{idx}
            elif "key_values" in output.name:
                if self.expected_key_symbolic_shape is None:
                    self.expected_key_symbolic_shape = output.shape
                else:
                    self.expected_value_symbolic_shape = output.shape
            if self.expected_key_symbolic_shape is not None and self.expected_value_symbolic_shape is not None:
                break

        self.key_sequence_length_idx = -2
        if (
            isinstance(self.expected_key_symbolic_shape[-1], str)
            and "sequence_length" in self.expected_key_symbolic_shape[-1]
        ):
            self.key_sequence_length_idx = -1

        self.value_sequence_length_idx = -2
        if (
            isinstance(self.expected_value_symbolic_shape[-1], str)
            and "sequence_length" in self.expected_value_symbolic_shape[-1]
        ):
            self.value_sequence_length_idx = -1

    def compute_past_key_values_output_shapes(
        self, input_ids: torch.Tensor, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> Dict[str, List[int]]:
        """
        Computes the outputs of the past key / value because it is not always easy to perform shape inference on them,
        which is needed for creating IO binding output buffers.

        Args:
            input_ids (`torch.Tensor`):
                The input ids that are associated with the current inputs.
            past_key_values (`Optional[Tuple[Tuple[torch.Tensor]]]`, defaults to `None`):
                The past key values associated with the current inputs.

        Returns:
            `Dict[str, List[int]]`: The dictionary mapping each past key value output name to its corresponding shape.
        """
        batch_size = input_ids.size(0)
        num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads
        sequence_length = input_ids.size(1)
        if past_key_values is not None:
            sequence_length += past_key_values[0].size(2)

        half_shape = [batch_size, num_attention_heads]
        if len(self.expected_key_symbolic_shape) == 3:
            half_shape[0] = batch_size * num_attention_heads
            half_shape.pop(1)

        key_shape = [sequence_length, embed_size_per_head]
        if self.key_sequence_length_idx == -1:
            key_shape[0], key_shape[1] = key_shape[1], key_shape[0]

        value_shape = [sequence_length, embed_size_per_head]
        if self.value_sequence_length_idx == -1:
            value_shape[0], value_shape[1] = value_shape[1], value_shape[0]

        key_shape = half_shape + key_shape
        value_shape = half_shape + value_shape

        return {name: key_shape if "key" in name else value_shape for name in self.key_value_output_names}

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = [past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer]

        if self.device.type == "cuda" and self.parent_model.use_io_binding:
            past_key_values_shapes = self.compute_past_key_values_output_shapes(
                input_ids,
                past_key_values=past_key_values,
            )

            past_key_values_inputs = past_key_values if past_key_values is not None else [None]

            model_inputs = [input_ids]

            if "attention_mask" in self.input_names:
                model_inputs.append(attention_mask)

            if past_key_values is not None:
                model_inputs += past_key_valuess

            io_binding, output_shapes, output_buffers = self.parent_model._prepare_io_binding(
                self.session,
                *model_inputs,
                known_output_shapes=past_key_values_shapes,
            )

            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer(2)
            past_key_values = tuple()
            for name in self.key_value_output_names:
                past_key_values += (output_buffers[name].view(output_shapes[name]),)

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (self-attention key and value per decoder layer)
            num_pkv = 2
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = {
                "input_ids": input_ids.cpu().detach().numpy(),
                "attention_mask": attention_mask.cpu().detach().numpy(),
            }

            if past_key_values is not None:
                # Add the past_key_values to the decoder inputs
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                    onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

            # Run inference
            outputs = self.session.run(None, onnx_inputs)

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 for the self-attention)
            past_key_values = tuple(
                torch.from_numpy(outputs[self.output_names[key]]).to(self.device)
                for key in self.key_value_output_names
            )

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # per decoder layer
            num_pkv = 2
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))
            logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self.device)

        return CausalLMOutputWithCrossAttentions(logits=logits, past_key_values=past_key_values)


class ORTDecoderForSeq2Seq(ORTDecoder):
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.
    """

    def compute_past_key_values_output_shapes(
        self, input_ids, encoder_hidden_states, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> Dict[str, int]:
        batch_size = input_ids.size(0)
        num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads

        sequence_length = input_ids.size(1)
        encoder_sequence_length = encoder_hidden_states.size(1)
        if past_key_values is not None:
            sequence_length += past_key_values[0].size(2)

        self_attn_shape = (batch_size, num_attention_heads, sequence_length, embed_size_per_head)
        cross_attn_shape = (batch_size, num_attention_heads, encoder_sequence_length, embed_size_per_head)

        past_key_values_shapes = {}
        for idx, name in enumerate(self.key_value_output_names):
            is_self_attn = idx % 4 < 2
            past_key_values_shapes[name] = self_attn_shape if is_self_attn else cross_attn_shape
        return past_key_values_shapes

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )

        if self.parent_model.device.type == "cuda" and self.parent_model.use_io_binding:
            past_key_values_shapes = self.compute_past_key_values_output_shapes(
                input_ids,
                encoder_hidden_states,
                past_key_values=past_key_values,
            )

            def filter_out_output(output_name):
                return not output_name.startswith("present") and output_name not in {"loss", "logits"}

            outputs_to_not_bind = {name for name in self.output_names if filter_out_output(name)}

            model_inputs = [input_ids]

            if "encoder_attention_mask" in self.input_names:
                model_inputs.append(encoder_attention_mask)

            if "encoder_hidden_states" in self.input_names:
                model_inputs.append(encoder_hidden_states)

            if past_key_values is not None:
                model_inputs += past_key_values

            if "labels" in self.input_names:
                model_inputs.append(labels)

            io_binding, output_shapes, output_buffers = self.parent_model._prepare_io_binding(
                self.session,
                *model_inputs,
                known_output_shapes=past_key_values_shapes,
                forward_function=self.forward,
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
            past_key_values = tuple()
            for name in self.key_value_output_names:
                past_key_values += (output_buffers[name].view(output_shapes[name]),)

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # cross-attention per decoder layer
            num_pkv = 4
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))

            logits = output_buffers["logits"].view(output_shapes["logits"])

            loss = None
            if "loss" in self.output_names:
                loss = output_buffers["loss"].view(output_shapes["loss"])
        else:
            onnx_inputs = {
                "input_ids": input_ids.cpu().detach().numpy(),
            }

            # Add the encoder_attention_mask inputs when needed
            if "encoder_attention_mask" in self.input_names:
                onnx_inputs["encoder_attention_mask"] = encoder_attention_mask.cpu().detach().numpy()

            # Add the encoder_hidden_states inputs when needed
            if "encoder_hidden_states" in self.input_names:
                onnx_inputs["encoder_hidden_states"] = encoder_hidden_states.cpu().detach().numpy()

            if past_key_values is not None:
                # Add the past_key_values to the decoder inputs
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                    onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

            if "labels" in self.input_names:
                # TODO: Any preprocessing like  `self._shift_right(labels)`?
                onnx_inputs["labels"] = labels.cpu().detach().numpy()

            # Run inference
            outputs = self.session.run(None, onnx_inputs)

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            past_key_values = tuple(
                torch.from_numpy(outputs[self.output_names[key]]).to(self.device)
                for key in self.key_value_output_names
            )

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # cross-attention per decoder layer
            num_pkv = 4
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))
            logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self.device)

            loss = None
            if "loss" in self.output_names:
                loss = torch.from_numpy(outputs[self.output_names["loss"]]).to(self.device)

        return Seq2SeqLMOutput(loss=loss, logits=logits, past_key_values=past_key_values)
