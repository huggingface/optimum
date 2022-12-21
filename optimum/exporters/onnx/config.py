# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Common ONNX configuration classes that handle most of the features for building model specific configurations."""

from typing import TYPE_CHECKING, Any, List, Mapping

from ...utils import (
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    logging,
)
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...utils import DummyInputGenerator

logger = logging.get_logger(__name__)


class TextEncoderOnnxConfig(OnnxConfig):
    """
    Handles encoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)


class TextDecoderOnnxConfig(OnnxConfigWithPast):
    """
    Handles decoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {"input_ids": {0: "batch_size", 1: "sequence_length"}}
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch_size", 1: "past_sequence_length + sequence_length"}
        else:
            common_inputs["attention_mask"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs


class TextSeq2SeqOnnxConfig(OnnxSeq2SeqConfigWithPast):
    """
    Handles encoder-decoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )

    @property
    def torch_to_onnx_input_map(self) -> Mapping[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_ids"] = {0: "batch_size", 1: "encoder_sequence_length"}

        common_inputs["attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs["attention_mask"][1] = "past_encoder_sequence_length + sequence_length"
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        return common_inputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](
            self.task,
            self._normalized_config,
            **kwargs,
        )
        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2](
            self.task,
            self._normalized_config,
            encoder_sequence_length=dummy_text_input_generator.sequence_length,
            **kwargs,
        )
        dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

        return dummy_inputs_generators

    def generate_dummy_inputs_for_validation(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._behavior is ConfigBehavior.DECODER:
            reference_model_inputs["input_ids"] = reference_model_inputs.pop("decoder_input_ids")
            reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]
            reference_model_inputs["encoder_attention_mask"] = reference_model_inputs.pop("attention_mask")
        return reference_model_inputs


class VisionOnnxConfig(OnnxConfig):
    """
    Handles vision architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)


class TextAndVisionOnnxConfig(OnnxConfig):
    """
    Handles multi-modal text and vision architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyVisionInputGenerator, DummyBboxInputGenerator)


class AudioOnnxConfig(OnnxConfig):
    """
    Handles audio architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator,)


class AudioToTextOnnxConfig(OnnxSeq2SeqConfigWithPast):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyAudioInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {}

        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_features"] = {0: "batch_size", 1: "feature_size", 2: "encoder_sequence_length"}

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        return common_inputs

    @property
    def torch_to_onnx_input_map(self) -> Mapping[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}

    def generate_dummy_inputs_for_validation(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._behavior is ConfigBehavior.DECODER:
            reference_model_inputs["input_ids"] = reference_model_inputs.pop("decoder_input_ids")
            reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]

        return reference_model_inputs
