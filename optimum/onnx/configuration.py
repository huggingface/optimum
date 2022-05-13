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

import copy
from abc import ABC
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from transformers.file_utils import TensorType, is_tf_available, is_torch_available
from transformers.onnx.utils import compute_effective_axis_dimension
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import torch
from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast


logger = logging.get_logger(__name__)


class OnnxConfigWithLoss(OnnxConfig, ABC):
    """
    Wrapper for the children classes of `transformers.onnx.OnnxConfig` to export the model through the ONNX format with loss in outputs.
    """

    _tasks_to_extra_inputs = {
        "default": OrderedDict({"labels": {0: "batch"}}),
        "masked-lm": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "causal-lm": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "seq2seq-lm": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "sequence-classification": OrderedDict({"labels": {0: "batch"}}),
        "token-classification": OrderedDict({"labels": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"labels": {0: "batch"}}),
        "question-answering": OrderedDict(
            {
                "start_positions": {0: "batch"},
                "end_positions": {0: "batch"},
            }
        ),
        "image-classification": OrderedDict({"labels": {0: "batch"}}),
    }
    _tasks_to_extra_outputs = {
        "default": OrderedDict({"loss": {}}),
    }

    def __init__(self, config: OnnxConfig):
        self.__dict__ = copy.deepcopy(config.__dict__)
        self._onnx_config = config
        if self.task not in self._tasks_to_extra_inputs:
            raise ValueError(
                f"{self.task} is not a supported task, supported tasks: {self._tasks_to_extra_inputs.keys()}"
            )

    @classmethod
    def from_model_config(cls, config: OnnxConfig) -> "OnnxConfigWithLoss":
        """
        Instantiate a OnnxConfigWithLoss for a specific model
        Args:
            config: The model's configuration to use when exporting to ONNX
        Returns:
            OnnxConfigWithLoss for this model
        """
        return cls(config)

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors(including labels) to provide to the model
        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        inputs = self._onnx_config.inputs
        inputs.update(self._tasks_to_extra_inputs[self.task])
        return inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors(including loss) to provide to the model
        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        common_outputs = self._onnx_config.outputs
        extra_outputs = self._tasks_to_extra_outputs["default"]
        common_outputs.update(extra_outputs)
        for key in reversed(extra_outputs.keys()):
            common_outputs.move_to_end(key, last=False)
        return copy.deepcopy(common_outputs)

    def _generate_extra_dummy_inputs_pt(
        self,
        dummy_inputs,
        batch_size,
        seq_length,
    ) -> Mapping[str, Any]:
        import torch

        for label, input in self._tasks_to_extra_inputs[self.task].items():
            if "sequence" in input.values():
                dummy_inputs[label] = torch.zeros(batch_size, seq_length, dtype=torch.long)
            else:
                dummy_inputs[label] = torch.zeros(batch_size, dtype=torch.long)
        return dummy_inputs

    def _generate_extra_dummy_inputs_tf(
        self,
        dummy_inputs,
        batch_size,
        seq_length,
    ) -> Mapping[str, Any]:
        import tensorflow as tf

        for label, input in self._tasks_to_extra_inputs[self.task].items():
            if "sequence" in input.values():
                dummy_inputs[label] = tf.zeros(batch_size, seq_length, dtype=tf.int64)
            else:
                dummy_inputs[label] = tf.zeros(batch_size, dtype=tf.int64)
        return dummy_inputs

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework
        Args:
            preprocessor: ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):
                The preprocessor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2)
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.
        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """
        # Generate dummy labels
        dummy_inputs = super().generate_dummy_inputs(
            preprocessor,
            batch_size,
            seq_length,
            is_pair,
            framework,
            num_channels,
            image_width,
            image_height,
            tokenizer,
        )
        label_batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=self.default_fixed_batch, num_token_to_add=0
        )
        label_seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=self.default_fixed_sequence, num_token_to_add=0
        )

        if framework == TensorType.PYTORCH:
            if is_torch_available():
                return self._generate_extra_dummy_inputs_pt(dummy_inputs, label_batch_size, label_seq_length)
            else:
                raise RuntimeError(f"Could not generate dummy inputs because no PyTorch installation was found.")
        elif framework == TensorType.TENSORFLOW:
            if is_tf_available():
                return self._generate_extra_dummy_inputs_tf(dummy_inputs, label_batch_size, label_seq_length)
            else:
                raise RuntimeError(f"Could not generate dummy inputs because no TensorFlow installation was found.")
        else:
            raise ValueError(
                f"Only two frameworks are supported for ONNX export: PyTorch or TensorFlow, but {framework} was provided."
            )


class OnnxConfigWithPastAndLoss(OnnxConfigWithLoss, ABC):
    def __init__(
        self,
        config: OnnxConfigWithPast,
        use_past: bool = False,
    ):
        super().__init__(config)
        self.use_past = use_past

    @classmethod
    def with_past(cls, config: OnnxConfigWithPast) -> "OnnxConfigWithPast":
        """
        Instantiate a OnnxConfigWithPastAndLoss with `use_past` attribute set to True
        Args:
            config: The underlying model's config to use when exporting to ONNX
        Returns:
            OnnxConfigWithPastAndLoss with `.use_past = True`
        """
        return cls(config, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super().outputs
        if self.use_past:
            self._onnx_config.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:

        dummy_inputs = self._onnx_config.generate_dummy_inputs(
            tokenizer,
            batch_size,
            seq_length,
            is_pair,
            framework,
        )
        label_batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=self.default_fixed_batch, num_token_to_add=0
        )
        label_seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=self.default_fixed_sequence, num_token_to_add=0
        )

        if framework == TensorType.PYTORCH:
            if is_torch_available():
                return self._generate_extra_dummy_inputs_pt(dummy_inputs, label_batch_size, label_seq_length)
            else:
                raise RuntimeError(f"Could not generate dummy inputs because no PyTorch installation was found.")
        elif framework == TensorType.TENSORFLOW:
            if is_tf_available():
                return self._generate_extra_dummy_inputs_tf(dummy_inputs, label_batch_size, label_seq_length)
            else:
                raise RuntimeError(f"Could not generate dummy inputs because no TensorFlow installation was found.")
        else:
            raise ValueError(
                f"Only two frameworks are supported for ONNX export: PyTorch or TensorFlow, but {framework} was provided."
            )


class OnnxSeq2SeqConfigWithPastAndLoss(OnnxConfigWithPastAndLoss):
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = self._onnx_config.outputs
        extra_outputs = self._tasks_to_extra_outputs["default"]
        common_outputs.update(extra_outputs)
        for key in reversed(extra_outputs.keys()):
            common_outputs.move_to_end(key, last=False)
        return copy.deepcopy(common_outputs)


class EncoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}})


class DecoderOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "past_decoder_sequence + sequence"}),
                ("encoder_hidden_states", {0: "batch", 1: "encoder_sequence"}),
                ("encoder_attention_mask", {0: "batch", 1: "encoder_sequence"}),
            ]
        )
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = {}
        dummy_input = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )
        batch = dummy_input["input_ids"].shape[0]
        encoder_seq_length = dummy_input["input_ids"].shape[1]
        encoder_hidden_states_shape = (batch, encoder_seq_length, self._config.hidden_size)
        common_inputs["input_ids"] = dummy_input["decoder_input_ids"]
        common_inputs["encoder_hidden_states"] = torch.zeros(encoder_hidden_states_shape)
        common_inputs["encoder_attention_mask"] = dummy_input["attention_mask"]
        if "past_key_values" in dummy_input:
            common_inputs["past_key_values"] = dummy_input["past_key_values"]

        return common_inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        num_mha_per_layer = 4
        for i in range(self.num_layers[0] * num_mha_per_layer):
            inputs_or_outputs[f"past_key_values_{i}"] = {0: "batch", 2: "past_sequence + sequence"}
