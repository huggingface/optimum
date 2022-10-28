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
"""Model specific ONNX configurations."""
import random
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple

from packaging import version

from ...utils import (
    DummyDecoderTextInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)
from .base import OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import DecoderOnnxConfig, EncoderOnnxConfig, Seq2SeqOnnxConfig, TextAndVisionOnnxConfig, VisionOnnxConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from .base import PatchingSpec


class BertOnnxConfig(EncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "token_type_ids": dynamic_axis,
        }


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
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}


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


class BigBirdOnnxConfig(DistilBertOnnxConfig):
    pass


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
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        pad_value_override = {}
        if not getattr(self._config, "pad_token_id", None):
            pad_value_override = {"pad_token_id": 0}
        super_values_override = super().values_override
        if super_values_override:
            return {**super_values_override, **pad_value_override}
        return pad_value_override


class GPTJOnnxConfig(GPT2OnnxConfig):
    pass


class CodeGenOnnxConfig(GPT2OnnxConfig):
    pass


class GPTNeoOnnxConfig(DecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_attention_heads="num_heads")


class BloomDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt"):
        past_key_shape = (
            self.batch_size * self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.sequence_length,
        )
        past_value_shape = (
            self.batch_size * self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework),
                self.random_float_tensor(past_value_shape, framework=framework),
            )
            for _ in range(self.num_layers)
        ]


class BloomOnnxConfig(DecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (BloomDummyPastKeyValuesGenerator,) + DecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")


class T5DummySeq2SeqPastKeyValuesGenerator(DummySeq2SeqPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt"):
        encoder_shape = (
            self.batch_size,
            self.normalized_config.encoder_num_attention_heads,
            self.encoder_sequence_length,
            self.normalized_config.key_value_dim,
        )
        decoder_shape = (
            self.batch_size,
            self.normalized_config.decoder_num_attention_heads,
            self.sequence_length,
            self.normalized_config.key_value_dim,
        )
        return [
            (
                self.random_float_tensor(decoder_shape, framework=framework),
                self.random_float_tensor(decoder_shape, framework=framework),
                self.random_float_tensor(encoder_shape, framework=framework),
                self.random_float_tensor(encoder_shape, framework=framework),
            )
            for _ in range(self.normalized_config.decoder_num_layers)
        ]


class T5OnnxConfig(Seq2SeqOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    DUMMY_INPUT_GENERATOR_CLASSES = Seq2SeqOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[:-1] + (T5DummySeq2SeqPastKeyValuesGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="d_model", num_attention_heads="num_heads", encoder_num_layers="num_layers", decoder_num_layers="num_decoder_layers", key_value_dim="d_kv", allow_new=True
    )


class MT5OnnxConfig(T5OnnxConfig):
    pass


class LongT5OnnxConfig(T5OnnxConfig):
    pass


class BartDummyTextInputGenerator(DummyTextInputGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedSeq2SeqConfig,
        batch_size: int = 2,
        sequence_length: int = 16,
        num_choices: int = 4,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        force_eos_token_id_presence: bool = True,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
        )
        self.force_eos_token_id_presence = force_eos_token_id_presence
        self.eos_token_id = normalized_config.eos_token_id

    def generate(self, input_name: str, framework: str = "pt"):
        int_tensor = super().generate(input_name, framework=framework)
        # This inserts EOS_TOKEN_ID at random locations along the sequence length dimension.
        if self.force_eos_token_id_presence and "input_ids" in input_name and self.task == "sequence-classification":
            for idx in range(self.batch_size):
                random_idx = random.randint(1, self.sequence_length - 1)
                int_tensor[idx][random_idx] = self.eos_token_id

        return int_tensor


class BartOnnxConfig(Seq2SeqOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",  # Used for the causal-lm task past key values input generation.
        encoder_num_attention_heads="encoder_attention_heads",
        decoder_num_attention_heads="decoder_attention_heads",
        eos_token_id="eos_token_id",
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        BartDummyTextInputGenerator,
        DummyDecoderTextInputGenerator,
        {
            "default": DummySeq2SeqPastKeyValuesGenerator,
            "causal-lm": DummyPastKeyValuesGenerator,
        },
    )

    def create_dummy_input_generator_classes(self):
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config)
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](
            self.task,
            self._normalized_config,
            batch_size=dummy_text_input_generator.batch_size,
            sequence_length=1 if self.use_past else 16,
        )
        task = "default" if self.task != "causal-lm" else "causal-lm"
        kwargs = {}
        if self.task != "causal-lm":
            kwargs["encoder_sequence_length"] = dummy_text_input_generator.sequence_length

        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2][task](
            self.task, self._normalized_config, batch_size=dummy_text_input_generator.batch_size, **kwargs
        )
        self.dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

    @property
    def inputs_for_default_and_seq2seq_lm(self):
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
            "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
        }
        if self.use_past:
            common_inputs["decoder_input_ids"] = {0: "batch_size"}
            # common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}
            # common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        if self.use_past:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs

    @property
    def inputs_for_causal_lm(self):
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
            "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
        }
        if self.use_past:
            for i in range(self._normalized_config.decoder_num_layers):
                common_inputs[f"past_key_values.{i}.key"] = {
                    0: "batch_size",
                    2: "past_sequence_length + sequence_length",
                }
                common_inputs[f"past_key_values.{i}.value"] = {
                    0: "batch_size",
                    2: "past_sequence_length + sequence_length",
                }

        return common_inputs

    @property
    def inputs_for_other_tasks(self):
        return {
            "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
            "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
            "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
            "decoder_attention_mask": {0: "batch_size", 1: "decoder_sequence_length"},
        }

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        inputs_properties = {
            "default": self.inputs_for_default_and_seq2seq_lm,
            "seq2seq-lm": self.inputs_for_default_and_seq2seq_lm,
            "causal-lm": self.inputs_for_causal_lm,
            "other": self.inputs_for_other_tasks,
        }
        return inputs_properties.get(self.task, inputs_properties["other"])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task in ["default", "seq2seq-lm"]:
            common_outputs = super().outputs
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs
            if self.use_past:
                for i in range(self._normalized_config.encoder_num_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}
                    common_outputs[f"present.{i}.value"] = {
                        0: "batch_size",
                        2: "past_sequence_length + sequence_length",
                    }
        return common_outputs

    def generate_dummy_inputs(self, framework: str = "pt"):
        # This will handle the attention mask padding when Bart is used for causal-lm.
        if self.task == "causal-lm":
            self.PAD_ATTENTION_MASK_TO_MATCH_TOTAL_SEQUENCE_LENGTH = True

        dummy_inputs = super().generate_dummy_inputs(framework=framework)

        # if self.use_past and self.task in ["default", "seq2seq-lm"]:
        #     attention_mask_length = dummy_inputs["decoder_attention_mask"].shape[1]
        #     decoder_past_length = dummy_inputs["past_key_values"][0][0].shape[2]
        #     dummy_inputs["decoder_attention_mask"] = self.dummy_inputs_generators[0].pad_input_on_dim(
        #         dummy_inputs["decoder_attention_mask"],
        #         desired_length=decoder_past_length,
        #         dim=1,
        #         dtype=dummy_inputs["decoder_attention_mask"].dtype,
        #     )

        # Setting it back to the default version.
        self.PAD_ATTENTION_MASK_TO_MATCH_TOTAL_SEQUENCE_LENGTH = False
        return dummy_inputs

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super().flatten_past_key_values(flattened_output, name, idx, t)
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self).flatten_past_key_values(
                flattened_output, name, idx, t
            )


class MBartOnnxConfig(BartOnnxConfig):
    pass


class M2M100OnnxConfig(BartOnnxConfig):
    pass


class BlenderbotOnnxConfig(BartOnnxConfig):
    pass


class BlenderbotSmallOnnxConfig(BartOnnxConfig):
    pass


class BigBirdPegasusOnnxConfig(BartOnnxConfig):
    pass


class MarianOnnxConfig(BartOnnxConfig):
    pass


class ViTOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    MIN_TORCH_VERSION = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}


class LevitOnnxConfig(ViTOnnxConfig):
    pass


class DeiTOnnxConfig(ViTOnnxConfig):
    pass


class BeitOnnxConfig(ViTOnnxConfig):
    pass


class ConvNextOnnxConfig(ViTOnnxConfig):
    pass


class MobileViTOnnxConfig(ViTOnnxConfig):
    pass


class ResNetOnnxConfig(ViTOnnxConfig):
    pass


class DetrOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # TODO: is pixel mask needed?
        return {**super().inputs, "pixel_mask": {0: "batch_size"}}


class YolosOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 12


class SegformerOnnxConfig(YolosOnnxConfig):
    pass


class CLIPNormalizedConfig(NormalizedTextAndVisionConfig):
    TEXT_CONFIG = "text_config"
    VISION_CONFIG = "vision_config"


class CLIPOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    DEFAULT_ONNX_OPSET = 14

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return {
            "logits_per_image": {0: "batch_size"},
            "logits_per_text": {0: "batch_size"},
            "text_embeds": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        }


class GroupViTOnnxConfig(CLIPOnnxConfig):
    pass


class OwlViTOnnxConfig(CLIPOnnxConfig):
    pass


class LayoutLMOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        allow_new=True,
        MAX_2D_POSITION_EMBEDDINGS="max_2d_position_embeddings",
    )

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        }


class LayoutLMv3OnnxConfig(TextAndVisionOnnxConfig):
    MIN_TORCH_VERSION = version.parse("1.12")
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        allow_new=True,
        MAX_2D_POSITION_EMBEDDINGS="max_2d_position_embeddings",
    )
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task in ["sequence-classification", "question-answering"]:
            pixel_values_dynamic_axes = {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        else:
            pixel_values_dynamic_axes = {0: "batch_size", 1: "num_channels"}
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": pixel_values_dynamic_axes,
        }


class Data2VecTextOnnxConfig(DistilBertOnnxConfig):
    pass


class Data2VecVisionOnnxConfig(ViTOnnxConfig):
    pass


# TODO: add support when audio models are supported.
class Data2VecAudioOnnxConfig(ViTOnnxConfig):
    @property
    def inputs(self):
        raise NotImplementedError


class PerceiverDummyInputGenerator(DummyVisionInputGenerator):
    def generate(self, input_name: str, framework: str = "pt"):
        input_ = super().generate(input_name, framework)
        # if input_name == "pixel_values":
        #     input_ = input_[None, :]
        return input_


class PerceiverOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        PerceiverDummyInputGenerator,
    ) + TextAndVisionOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES

    def __init__(
        self, config: "PretrainedConfig", task: str = "default", patching_specs: Optional[List["PatchingSpec"]] = None
    ):
        super().__init__(config, task=task, patching_specs=patching_specs)
        self.is_generating_dummy_inputs = False

    @property
    def inputs_name(self):
        if self.is_generating_dummy_inputs:
            if self.task in ["masked-lm", "sequence-classification"]:
                return "input_ids"
            else:
                return "pixel_values"
        else:
            return "inputs"

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # TODO: validate that.
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            self.inputs_name: dynamic_axis,
            # TODO: should we add the attention_mask?
            # This breaks things for image-classification, suspected bug is the DummyInputGenerators not having the
            # same num_channels / sequence_length.
            # "attention_mask": dynamic_axis,
        }

    def generate_dummy_inputs(self, framework: str = "pt"):
        self.is_generating_dummy_inputs = True
        dummy_inputs = super().generate_dummy_inputs(framework=framework)
        specialized_inputs_name = self.inputs_name
        self.is_generating_dummy_inputs = True
        dummy_inputs[self.inputs_name] = dummy_inputs.pop(specialized_inputs_name)
        return dummy_inputs
