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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from packaging import version
from transformers.utils import is_tf_available

from ...utils import (
    DEFAULT_DUMMY_SHAPES,
    DummyAudioInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyVisionInputGenerator,
    NormalizedConfig,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
    logging,
)
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
    AudioOnnxConfig,
    AudioToTextOnnxConfig,
    EncoderDecoderOnnxConfig,
    TextAndVisionOnnxConfig,
    TextDecoderOnnxConfig,
    TextEncoderOnnxConfig,
    TextSeq2SeqOnnxConfig,
    VisionOnnxConfig,
)
from .model_patcher import WavLMModelPatcher


if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel

    from ...utils import DummyInputGenerator
    from .model_patcher import ModelPatcher

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

logger = logging.get_logger(__name__)


class BertOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
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


class NystromformerOnnxConfig(BertOnnxConfig):
    pass


class XLMOnnxConfig(BertOnnxConfig):
    pass


class SplinterOnnxConfig(BertOnnxConfig):
    pass


class DistilBertOnnxConfig(BertOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}


class MPNetOnnxConfig(DistilBertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12


class RobertaOnnxConfig(DistilBertOnnxConfig):
    pass


class CamembertOnnxConfig(DistilBertOnnxConfig):
    pass


class FlaubertOnnxConfig(BertOnnxConfig):
    pass


class IBertOnnxConfig(DistilBertOnnxConfig):
    pass


class XLMRobertaOnnxConfig(DistilBertOnnxConfig):
    pass


class DebertaOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            common_inputs.pop("token_type_ids")
        return common_inputs


class DebertaV2OnnxConfig(DebertaOnnxConfig):
    pass


class GPT2OnnxConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")

    @property
    def values_override(self) -> Optional[Dict[str, Any]]:
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


class ImageGPTOnnxConfig(GPT2OnnxConfig):
    pass


class GPTNeoOnnxConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_attention_heads="num_heads")


class GPTNeoXOnnxConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


class OPTOnnxConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


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


class BloomOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        BloomDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = BloomDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        """
        Refer to OnnxConfigWithPast in base.py
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {
                0: "batch_size x num_heads",
                2: decoder_sequence_name,
            }
            inputs_or_outputs[f"{name}.{i}.value"] = {
                0: "batch_size x num_heads",
                1: decoder_sequence_name,
            }


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


class T5OnnxConfig(TextSeq2SeqOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    DUMMY_INPUT_GENERATOR_CLASSES = TextSeq2SeqOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[:-1] + (
        T5DummySeq2SeqPastKeyValuesGenerator,
    )
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="d_model",
        num_attention_heads="num_heads",
        encoder_num_layers="num_layers",
        decoder_num_layers="num_decoder_layers",
        key_value_dim="d_kv",
        allow_new=True,
    )


class MT5OnnxConfig(T5OnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4


class LongT5OnnxConfig(T5OnnxConfig):
    DEFAULT_ONNX_OPSET = 14


class BartDummyTextInputGenerator(DummyTextInputGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedSeq2SeqConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        force_eos_token_id_presence: bool = True,
        **kwargs,
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
                if self.eos_token_id in int_tensor[idx]:
                    continue
                random_idx = random.randint(1, self.sequence_length - 1)
                int_tensor[idx][random_idx] = self.eos_token_id

        return int_tensor


class BartOnnxConfig(TextSeq2SeqOnnxConfig):
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
        {
            "default": DummySeq2SeqDecoderTextInputGenerator,
            "causal-lm": DummyDecoderTextInputGenerator,
        },
        {
            "default": DummySeq2SeqPastKeyValuesGenerator,
            "causal-lm": DummyPastKeyValuesGenerator,
        },
    )

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        task = "default" if self.task != "causal-lm" else "causal-lm"
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1][task](
            self.task, self._normalized_config, **kwargs
        )
        if self.task != "causal-lm":
            kwargs["encoder_sequence_length"] = dummy_text_input_generator.sequence_length

        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2][task](
            self.task, self._normalized_config, **kwargs
        )
        dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

        return dummy_inputs_generators

    @property
    def inputs_for_default_and_seq2seq_lm(self):
        return super().inputs

    @property
    def inputs_for_causal_lm(self):
        if self.use_past_in_inputs:
            common_inputs = {
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size", 1: "past_sequence_length + 1"},
            }
            for i in range(self._normalized_config.decoder_num_layers):
                common_inputs[f"past_key_values.{i}.key"] = {
                    0: "batch_size",
                    2: "past_sequence_length",
                }
                common_inputs[f"past_key_values.{i}.value"] = {
                    0: "batch_size",
                    2: "past_sequence_length",
                }
        else:
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }

        return common_inputs

    @property
    def inputs_for_other_tasks(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        inputs_properties = {
            "default": self.inputs_for_default_and_seq2seq_lm,
            "seq2seq-lm": self.inputs_for_default_and_seq2seq_lm,
            "causal-lm": self.inputs_for_causal_lm,
            "other": self.inputs_for_other_tasks,
        }
        return inputs_properties.get(self.task, inputs_properties["other"])

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.task in ["default", "seq2seq-lm"]:
            common_outputs = super().outputs
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs
            if self.task != "causal-lm":
                common_outputs["encoder_last_hidden_state"] = {0: "batch_size", 1: "sequence_length"}
            if self.use_present_in_outputs:
                for i in range(self._normalized_config.encoder_num_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}
                    common_outputs[f"present.{i}.value"] = {
                        0: "batch_size",
                        2: "past_sequence_length + sequence_length",
                    }
        return common_outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        # This will handle the attention mask padding when Bart is used for causal-lm.
        if self.task == "causal-lm":
            self.PAD_ATTENTION_MASK_TO_PAST = True

        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)

        # Setting it back to the default version.
        self.PAD_ATTENTION_MASK_TO_PAST = False
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


# big_bird and bigbird_pegasus are unsupported for now as block sparse attention is written in pure python and numpy in transformers.
# Thus, the case attention_type == "block_sparse" is unusable.
# Even with rewritting this part in pure PyTorch, torch.onnx.export is then prohibitively slow.
# References: https://github.com/pytorch/pytorch/issues/63734 & https://github.com/pytorch/pytorch/issues/94821
"""
class BigBirdOnnxConfig(DistilBertOnnxConfig):
    pass

class BigBirdPegasusOnnxConfig(BartOnnxConfig):
    def generate_dummy_inputs_for_validation(self, reference_model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._behavior is ConfigBehavior.ENCODER:
            # TODO: check why the attention mask is not present in the exported model
            reference_model_inputs.pop("attention_mask")
        return super().generate_dummy_inputs_for_validation(reference_model_inputs)
"""


class PegasusOnnxConfig(BartOnnxConfig):
    pass


class MarianOnnxConfig(BartOnnxConfig):
    pass


class ViTOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    MIN_TORCH_VERSION = version.parse("1.11")

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
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


class RegNetOnnxConfig(ViTOnnxConfig):
    # This config has the same inputs as ViTOnnxConfig
    pass


class ResNetOnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3


class DetrOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        # TODO: is pixel mask needed?
        return {**super().inputs, "pixel_mask": {0: "batch_size"}}


class YolosOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 12


class SwinOnnxConfig(ViTOnnxConfig):
    pass


class PoolFormerOnnxConfig(ViTOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    ATOL_FOR_VALIDATION = 2e-3


class SegformerOnnxConfig(YolosOnnxConfig):
    pass


class MobileNetV1OnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"pixel_values": {0: "batch_size"}}


class MobileNetV2OnnxConfig(MobileNetV1OnnxConfig):
    pass


class DonutSwinOnnxConfig(ViTOnnxConfig):
    pass


class CLIPNormalizedConfig(NormalizedTextAndVisionConfig):
    TEXT_CONFIG = "text_config"
    VISION_CONFIG = "vision_config"


class CLIPOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    DEFAULT_ONNX_OPSET = 14

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size", 1: "sequence_length"},
            "pixel_values": {0: "image_batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "logits_per_image": {0: "image_batch_size", 1: "text_batch_size"},
            "logits_per_text": {0: "text_batch_size", 1: "image_batch_size"},
            "text_embeds": {0: "text_batch_size"},
            "image_embeds": {0: "image_batch_size"},
        }


class CLIPTextOnnxConfig(TextEncoderOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        allow_new=True,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        if framework == "pt":
            import torch

            dummy_inputs["input_ids"] = dummy_inputs["input_ids"].to(dtype=torch.int32)
        return dummy_inputs


class UNetOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    # The ONNX export of a CLIPText architecture, an other Stable Diffusion component, needs the Trilu
    # operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        DummyTimestepInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "timestep": {0: "steps"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "out_sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return {
            "sample": "out_sample",
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]
        return dummy_inputs


class VaeEncoderOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-2
    # The ONNX export of a CLIPText architecture, an other Stable Diffusion component, needs the Trilu
    # operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels",
        image_size="sample_size",
        allow_new=True,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent_sample": {0: "batch_size", 1: "num_channels_latent", 2: "height_latent", 3: "width_latent"},
        }


class VaeDecoderOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    # The ONNX export of a CLIPText architecture, an other Stable Diffusion component, needs the Trilu
    # operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="latent_channels",
        allow_new=True,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent_sample": {0: "batch_size", 1: "num_channels_latent", 2: "height_latent", 3: "width_latent"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }


class GroupViTOnnxConfig(CLIPOnnxConfig):
    pass


# TODO: not supported now because of aten:broadcast_to, can be most likely patched.
# class OwlViTOnnxConfig(CLIPOnnxConfig):
#     pass


class LayoutLMOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        allow_new=True,
        MAX_2D_POSITION_EMBEDDINGS="max_2d_position_embeddings",
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
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
        image_size="input_size",
    )
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
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


class Data2VecAudioOnnxConfig(AudioOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    ATOL_FOR_VALIDATION = 1e-4


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

    def __init__(self, config: "PretrainedConfig", task: str = "default"):
        super().__init__(config, task=task)
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
    def inputs(self) -> Dict[str, Dict[int, str]]:
        # TODO: validate that.
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            self.inputs_name: dynamic_axis,
            # TODO: should we add the attention_mask?
            # This breaks things for image-classification, suspected bug is the DummyInputGenerators not having the
            # same num_channels / sequence_length.
            # "attention_mask": dynamic_axis,
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        self.is_generating_dummy_inputs = True
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        specialized_inputs_name = self.inputs_name
        self.is_generating_dummy_inputs = True
        dummy_inputs[self.inputs_name] = dummy_inputs.pop(specialized_inputs_name)
        return dummy_inputs


class HubertOnnxConfig(AudioOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig


class Wav2Vec2OnnxConfig(HubertOnnxConfig):
    pass


class Wav2Vec2ConformerOnnxConfig(HubertOnnxConfig):
    pass


class SEWOnnxConfig(HubertOnnxConfig):
    pass


class SEWDOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12


class UniSpeechOnnxConfig(HubertOnnxConfig):
    pass


class UniSpeechSATOnnxConfig(HubertOnnxConfig):
    pass


class WavLMOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    # we need to set output_attentions=True in the model input to avoid calling
    # torch.nn.functional.scaled_dot_product_attention that is not supported by the ONNX export
    # due to the op torch.nn.functional.multi_head_attention_forward used for WavLM
    def patch_model_for_export(self, model: Union["PreTrainedModel", "TFPreTrainedModel"]) -> "ModelPatcher":
        return WavLMModelPatcher(self, model)


class ASTDummyAudioInputGenerator(DummyAudioInputGenerator):
    def generate(self, input_name: str, framework: str = "pt"):
        shape = [self.batch_size, self.normalized_config.max_length, self.normalized_config.num_mel_bins]
        if input_name == "input_values":
            return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework)
        return super().generate(input_name, framework=framework)


class ASTOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_mel_bins="num_mel_bins", max_length="max_length", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (ASTDummyAudioInputGenerator,)
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"input_values": {0: "batch_size"}}


# TODO: currently disabled because an operator seems not supported by ONNX.
# class MCTCTDummyAudioInputGenerator(DummyAudioInputGenerator):
#     def generate(self, input_name: str, framework: str = "pt"):
#         shape = [self.batch_size, self.sequence_length, self.normalized_config.input_features_per_channel]
#         if input_name == "input_features":
#             return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework)
#         return super().generate(input_name, framework=framework)
#
#
# class MCTCTOnnxConfig(OnnxConfig):
#     NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(input_features_per_channel="input_feat_per_channel", allow_new=True)
#     DUMMY_INPUT_GENERATOR_CLASSES = (MCTCTDummyAudioInputGenerator,)
#     DEFAULT_ONNX_OPSET = 13
#
#     @property
#     def inputs(self) -> Dict[str, Dict[int, str]]:
#         return {"input_features": {0: "batch_size", 1: "sequence_classification"}}


class WhisperOnnxConfig(AudioToTextOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"][1] = f"{common_inputs['encoder_outputs'][1]} / 2"
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs
        if self._behavior is ConfigBehavior.ENCODER:
            # For Whisper, we need to name the second axis as encoder_sequence_length / 2 as the axis name is used for
            # dummy input generation
            common_outputs["last_hidden_state"][1] = f"{common_outputs['last_hidden_state'][1]} / 2"
        return common_outputs


class Speech2TextDummyAudioInputGenerator(DummyAudioInputGenerator):
    def generate(self, input_name: str, framework: str = "pt"):
        shape = [self.batch_size, self.sequence_length, self.normalized_config.input_features_per_channel]
        if input_name == "input_features":
            return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework)
        return super().generate(input_name, framework=framework)


class Speech2TextOnnxConfig(AudioToTextOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",
        input_features_per_channel="input_feat_per_channel",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        (Speech2TextDummyAudioInputGenerator,)
        + AudioToTextOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[1:]
        + (DummyTextInputGenerator,)
    )
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}

        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_features"] = {0: "batch_size", 1: "feature_size", 2: "encoder_sequence_length"}
            common_inputs["attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"] = {
                0: "batch_size",
                1: f"encoder_sequence_length / {( 2 * self._config.num_conv_layers)}",
            }

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs
        if self._behavior is ConfigBehavior.ENCODER:
            # for Speech2text, we need to name the second axis as
            # encoder_sequence_length / 2 * self._config.num_conv_layers as the axis name is
            # used for dummy input generation
            common_outputs["last_hidden_state"][
                1
            ] = f"{common_outputs['last_hidden_state'][1]} / {( 2 * self._config.num_conv_layers)}"
        return common_outputs


# TODO: Replace the TextSeq2SeqOnnxConfig inheritance with VisionToTextOnnxConfig when added.
# The change below however does not affect the export for the model
class TrOCROnnxConfig(TextSeq2SeqOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",
        decoder_num_attention_heads="decoder_attention_heads",
        hidden_size="cross_attention_hidden_size",
    )


class VisionEncoderDecoderOnnxConfig(EncoderDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    ATOL_FOR_VALIDATION = 1e-3

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        use_past: bool = False,
        use_past_in_inputs: Optional[bool] = None,
        use_present_in_outputs: Optional[bool] = None,
        behavior: ConfigBehavior = ConfigBehavior.MONOLITH,
    ):
        super().__init__(config, task, use_past, use_past_in_inputs, use_present_in_outputs, behavior)

        # TODO: Check modeling code to fix the issue with use_cache for trocr
        if config.decoder.model_type == "trocr":
            if self.use_past_in_inputs:
                raise ValueError("Exporting past key values is not supported with TrOCR model!")

            self.use_present_in_outputs = False

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}

        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["pixel_values"] = {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}

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
