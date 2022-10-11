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
"""Model specific onnx configurations."""
from collections import OrderedDict
from typing import Any, Mapping, Optional

from ...utils import (
    DummyDecoderTextInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    NormalizedConfig,
    NormalizedSeq2SeqConfig,
)
from .config import DecoderOnnxConfig, EncoderOnnxConfig, Seq2SeqOnnxConfig


class BertOnnxConfig(EncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


class AlbertOnnxConfig(BertOnnxConfig):
    pass


class ConvBertOnnxConfig(BertOnnxConfig):
    pass


class ElectraOnnxConfig(BertOnnxConfig):
    pass


class RoFormerOnnxConfig(BertOnnxConfig):
    pass


class SqueezeBertOnnxConfig(BertOnnxConfig):
    pass


class MobileBertOnnxConfig(BertOnnxConfig):
    pass


class XLMOnnxConfig(BertOnnxConfig):
    pass


class DistilBertOnnxConfig(BertOnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )


class RobertaOnnxConfig(DistilBertOnnxConfig):
    pass


class CamembertOnnxConfig(DistilBertOnnxConfig):
    pass


class FlaubertOnnxConfig(DistilBertOnnxConfig):
    pass


class IBertOnnxConfig(DistilBertOnnxConfig):
    pass


class XLMRobertaOnnxConfig(DistilBertOnnxConfig):
    pass


# TODO: validate that
class DebertaOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            common_inputs.pop("token_type_ids")
        return common_inputs


class DebertaV2OnnxConfig(DebertaOnnxConfig):
    pass


class GPT2OnnxConfig(DecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        pad_value_override = {}
        if not getattr(self._config, "pad_token_id", None):
            pad_value_override = {"pad_token_id": 0}
        super_values_override = super().values_override
        if super_values_override:
            return {**super_values_override, **pad_value_override}
        return pad_value_override


# TODO: validate that.
class BloomOnnxConfig(GPT2OnnxConfig):
    pass


class GPTNeoOnnxConfig(DecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_attention_heads="num_heads")


class T5OnnxConfig(Seq2SeqOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        num_attention_heads="num_heads", decoder_num_layers="num_decoder_layers"
    )


class MT5OnnxConfig(T5OnnxConfig):
    pass


class LongT5OnnxConfig(Seq2SeqOnnxConfig):
    pass
    # @property
    # def inputs(self) -> Mapping[str, Mapping[int, str]]:
    #     common_inputs = {
    #         "input_ids": {0: "batch", 1: "encoder_sequence"},
    #         "attention_mask": {0: "batch", 1: "encoder_sequence"},
    #     }
    #     if self.use_past:
    #         common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
    #         common_inputs["decoder_input_ids"] = {0: "batch"}
    #         common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
    #     else:
    #         common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
    #         common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

    #     if self.use_past:
    #         self.fill_with_past_key_values_(common_inputs, direction="inputs")

    #     return common_inputs


# class BartOnnxConfig(Seq2SeqOnnxConfig):
#     DUMMY_INPUT_GENERATOR_CLASSES = (
#         DummyTextInputGenerator,
#         DummyDecoderTextInputGenerator,
#         {
#             "default": DummySeq2SeqPastKeyValuesGenerator,
#             "causal-lm": DummyPastKeyValuesGenerator,
#         }
#     )
#     def _create_dummy_input_generator_classes(self):
#         dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config)
#         dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](
#             self.task,
#             self._normalized_config,
#             batch_size=dummy_text_input_generator.batch_size,
#             sequence_length=1 if self.use_past else None,
#         )
#         task = "default" if self.task != "causal-lm" else "causal-lm"
#         dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2][task](
#             self.task,
#             self._normalized_config,
#             batch_size=dummy_text_input_generator.batch_size,
#             encoder_sequence_length=dummy_text_input_generator.sequence_length,
#         )
#         self.dummy_inputs_generators = [dummy_text_input_generator, dummy_decoder_text_input_generator, dummy_seq2seq_past_key_values_generator]
#
#     @property
#     def inputs(self) -> Mapping[str, Mapping[int, str]]:
#         if self.task in ["default", "seq2seq-lm"]:
#             common_inputs = OrderedDict(
#                 [
#                     ("input_ids", {0: "batch", 1: "encoder_sequence"}),
#                     ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
#                 ]
#             )
#
#             if self.use_past:
#                 common_inputs["decoder_input_ids"] = {0: "batch"}
#                 common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
#             else:
#                 common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
#                 common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}
#
#             if self.use_past:
#                 self.add_past_key_values(common_inputs, direction="inputs")
#         elif self.task == "causal-lm":
#             # TODO: figure this case out.
#             common_inputs = OrderedDict(
#                 [
#                     ("input_ids", {0: "batch", 1: "encoder_sequence"}),
#                     ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
#                 ]
#             )
#             if self.use_past:
#                 num_encoder_layers, _ = self.num_layers
#                 for i in range(num_encoder_layers):
#                     common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
#                     common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
#         else:
#             common_inputs = OrderedDict(
#                 [
#                     ("input_ids", {0: "batch", 1: "encoder_sequence"}),
#                     ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
#                     ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),
#                     ("decoder_attention_mask", {0: "batch", 1: "decoder_sequence"}),
#                 ]
#             )
#
#         return common_inputs
