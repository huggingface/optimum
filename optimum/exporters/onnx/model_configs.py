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
import math
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from packaging import version
from transformers.utils import is_tf_available

from ...utils import (
    DEFAULT_DUMMY_SHAPES,
    BloomDummyPastKeyValuesGenerator,
    Dinov2DummyInputGenerator,
    DummyAudioInputGenerator,
    DummyCodegenDecoderTextInputGenerator,
    DummyDecisionTransformerInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyEncodecInputGenerator,
    DummyFluxTransformerTextInputGenerator,
    DummyFluxTransformerVisionInputGenerator,
    DummyInputGenerator,
    DummyIntGenerator,
    DummyPastKeyValuesGenerator,
    DummyPatchTSTInputGenerator,
    DummyPix2StructInputGenerator,
    DummyPointsGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummySpeechT5InputGenerator,
    DummyTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyTransformerTextInputGenerator,
    DummyTransformerTimestepInputGenerator,
    DummyTransformerVisionInputGenerator,
    DummyVisionEmbeddingsGenerator,
    DummyVisionEncoderDecoderPastKeyValuesGenerator,
    DummyVisionInputGenerator,
    DummyXPathSeqInputGenerator,
    FalconDummyPastKeyValuesGenerator,
    GemmaDummyPastKeyValuesGenerator,
    GPTBigCodeDummyPastKeyValuesGenerator,
    LongformerDummyTextInputGenerator,
    MCTCTDummyAudioInputGenerator,
    MistralDummyPastKeyValuesGenerator,
    NormalizedConfig,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedTextConfigWithGQA,
    NormalizedTimeSeriesForecastingConfig,
    NormalizedVisionConfig,
    PerceiverDummyInputGenerator,
    VitPoseDummyInputGenerator,
    is_diffusers_available,
    is_diffusers_version,
    is_transformers_version,
    logging,
)
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
    AudioOnnxConfig,
    AudioToTextOnnxConfig,
    EncoderDecoderBaseOnnxConfig,
    TextAndVisionOnnxConfig,
    TextDecoderOnnxConfig,
    TextDecoderWithPositionIdsOnnxConfig,
    TextEncoderOnnxConfig,
    TextSeq2SeqOnnxConfig,
    VisionOnnxConfig,
)
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import (
    CLIPModelPatcher,
    FalconModelPatcher,
    MgpstrModelPatcher,
    MistralModelPatcher,
    MusicgenModelPatcher,
    SAMModelPatcher,
    SentenceTransformersCLIPPatcher,
    SentenceTransformersTransformerPatcher,
    SpeechT5ModelPatcher,
    VisionEncoderDecoderPatcher,
    VitPoseModelPatcher,
    WavLMModelPatcher,
)


# TODO : moved back onnx imports applied in https://github.com/huggingface/optimum/pull/2114/files after refactorization


if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel

    from .model_patcher import ModelPatcher

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin

logger = logging.get_logger(__name__)


class BertOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.

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
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class ConvBertOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class ElectraOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class RoFormerOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class SqueezeBertOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class MobileBertOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class NystromformerOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class XLMOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class SplinterOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class RemBertOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class LongformerOnnxConfig(BertOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (LongformerDummyTextInputGenerator,)
    DEFAULT_ONNX_OPSET = 14

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        inputs = super().inputs

        inputs["global_attention_mask"] = inputs["attention_mask"]

        return inputs


class MegatronBertOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class DistilBertOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for transformers>=4.46.0

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}


class ModernBertOnnxConfig(DistilBertOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.48.0")


class MPNetOnnxConfig(DistilBertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12  # For lower opsets, results in: Type 'tensor(int64)' of input parameter (/0/auto_model/encoder/Add_1_output_0) of operator (Min) in node (/0/auto_model/encoder/Min) is invalid.


class RobertaOnnxConfig(DistilBertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class CamembertOnnxConfig(DistilBertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class FlaubertOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class IBertOnnxConfig(DistilBertOnnxConfig):
    pass


class XLMRobertaOnnxConfig(DistilBertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class DebertaOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            common_inputs.pop("token_type_ids")
        return common_inputs


class MarkupLMOnnxConfig(BertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyXPathSeqInputGenerator,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        xpath_dynamic_axis = {0: "batch_size", 1: "sequence_length", 2: "max_depth"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "token_type_ids": dynamic_axis,
            "xpath_subs_seq": xpath_dynamic_axis,
            "xpath_tags_seq": xpath_dynamic_axis,
        }


class DebertaV2OnnxConfig(DebertaOnnxConfig):
    pass


class EsmOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
        }


class GPT2OnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # uses SDPA in Transformers, hence opset>=14.
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")


class GPTJOnnxConfig(GPT2OnnxConfig):
    pass


class CodeGenOnnxConfig(GPT2OnnxConfig):
    pass


class ImageGPTOnnxConfig(GPT2OnnxConfig):
    pass


class DecisionTransformerOnnxConfig(OnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyDecisionTransformerInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedConfig

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "states": {0: "batch_size", 1: "sequence_length"},
            "actions": {0: "batch_size", 1: "sequence_length"},
            "timesteps": {0: "batch_size", 1: "sequence_length"},
            "returns_to_go": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "state_preds": {0: "batch_size", 1: "sequence_length"},
            "action_preds": {0: "batch_size", 1: "sequence_length"},
            "return_preds": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }


class GPTNeoOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_attention_heads="num_heads")


class GPTNeoXOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # uses SDPA in Transformers, hence opset>=14.
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


# OPT does not take position_ids as input for transfomers < v4.46, needs it for transformers >= v4.46
if is_transformers_version(">=", "4.45.99"):

    class OPTOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
        DEFAULT_ONNX_OPSET = 14  # uses SDPA in Transformers, hence opset>=14.
        NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

else:

    class OPTOnnxConfig(TextDecoderOnnxConfig):
        DEFAULT_ONNX_OPSET = 14  # uses SDPA in Transformers, hence opset>=14.
        NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


class LlamaOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # Llama now uses F.scaled_dot_product_attention by default for torch>=2.1.1.

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


class OlmoOnnxConfig(LlamaOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4
    MIN_TRANSFORMERS_VERSION = version.parse("4.40.0")


class Olmo2OnnxConfig(OlmoOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.47.0")


class Qwen2OnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.37.0")


class GemmaOnnxConfig(LlamaOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    MIN_TRANSFORMERS_VERSION = version.parse("4.38.0")


class GraniteOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.45.0")
    MIN_TORCH_VERSION = version.parse("2.5.0")


class PhiOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # Phi now uses F.scaled_dot_product_attention by default for torch>=2.1.1.
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = version.parse("4.36.0")


class Phi3OnnxConfig(PhiOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
    MIN_TRANSFORMERS_VERSION = version.parse("4.50.0")


class MistralOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    # This is because of the patching of torch.triu in AttentionMaskConverter, that exists from transformers>=4.35
    MIN_TRANSFORMERS_VERSION = version.parse("4.34.99")

    # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MistralModelPatcher(self, model, model_kwargs=model_kwargs)


class MPTOnnxConfig(TextDecoderOnnxConfig):
    # MPT does not require position_ids input.
    DEFAULT_ONNX_OPSET = 13
    # TODO: fix inference for transformers < v4.41 for beam_search > 1
    MIN_TRANSFORMERS_VERSION = version.parse("4.41.0")
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="n_heads", hidden_size="d_model", num_layers="n_layers"
    )


class BloomOnnxConfig(TextDecoderOnnxConfig):
    # Bloom does not require position_ids input.
    DUMMY_INPUT_GENERATOR_CLASSES = (
        BloomDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = BloomDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")
    DEFAULT_ONNX_OPSET = 14  # Bloom uses aten::triu that requires opset>=14, and F.scaled_dot_product_attention

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if is_transformers_version(">=", "4.44"):
            super().add_past_key_values(inputs_or_outputs, direction)
        else:
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


class GPTBigCodeOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        GPTBigCodeDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DEFAULT_ONNX_OPSET = 14  # GPT BigCode now uses F.scaled_dot_product_attention by default for torch>=2.1.1.
    DUMMY_PKV_GENERATOR_CLASS = GPTBigCodeDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("gpt_bigcode")

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            # No dim for `n_head` when using multi-query attention
            inputs_or_outputs[f"{name}.{i}.key_value"] = {
                0: "batch_size",
                1: decoder_sequence_name,
            }

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key_value"] = t


class FalconOnnxConfig(TextDecoderOnnxConfig):
    # This is due to the cache refactoring for Falcon in 4.36
    MIN_TRANSFORMERS_VERSION = version.parse("4.35.99")

    DUMMY_INPUT_GENERATOR_CLASSES = (
        FalconDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DEFAULT_ONNX_OPSET = 14  # Falcon uses aten::triu that requires opset>=14, and F.scaled_dot_product_attention
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_PKV_GENERATOR_CLASS = FalconDummyPastKeyValuesGenerator

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        preprocessors: Optional[List[Any]] = None,
        legacy: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            preprocessors=preprocessors,
            legacy=legacy,
        )
        # For some reason Falcon config.num_kv_heads can not be trusted, see in Transformers:
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L337
        self._normalized_config.num_kv_heads = (
            self._normalized_config.num_kv_heads
            if (self._normalized_config.new_decoder_architecture or not self._normalized_config.multi_query)
            else 1
        )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs

        if not self.legacy and not self._config.alibi and self.task in ["text-generation", "feature-extraction"]:
            # When alibi is used, position_ids are not used in Falcon.
            # Reference: https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1116
            common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs

    # we need to set output_attentions=True in the model input to avoid calling
    # torch.nn.functional.scaled_dot_product_attention that is not supported by the ONNX export
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return FalconModelPatcher(self, model, model_kwargs=model_kwargs)


class T5DummySeq2SeqPastKeyValuesGenerator(DummySeq2SeqPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
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
                self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.normalized_config.decoder_num_layers)
        ]


class T5OnnxConfig(TextSeq2SeqOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # T5 uses aten::triu that requires opset>=14
    DUMMY_INPUT_GENERATOR_CLASSES = TextSeq2SeqOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[:-1] + (
        T5DummySeq2SeqPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = T5DummySeq2SeqPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="d_model",
        num_attention_heads="num_heads",
        encoder_num_layers="num_layers",
        decoder_num_layers="num_decoder_layers",
        key_value_dim="d_kv",
        allow_new=True,
    )

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if self._behavior is ConfigBehavior.DECODER:
            reference_model_inputs["input_ids"] = reference_model_inputs.pop("decoder_input_ids")

        if onnx_input_names is not None:
            if "encoder_outputs" in reference_model_inputs:
                if "encoder_hidden_states" in onnx_input_names:
                    reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]
                else:
                    reference_model_inputs.pop("encoder_outputs")
        else:
            # TODO: remove this else in optimum 2.0 and make onnx_input_names a required argument
            # T5 requires encoder_hidden_states as an input for both the without/with past models,
            # which is different than other architectures that require it only for the without past case
            reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]

        return super().generate_dummy_inputs_for_validation(reference_model_inputs)


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
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
        )
        self.force_eos_token_id_presence = force_eos_token_id_presence
        self.eos_token_id = normalized_config.eos_token_id

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        int_tensor = super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)
        # This inserts EOS_TOKEN_ID at random locations along the sequence length dimension.
        if self.force_eos_token_id_presence and "input_ids" in input_name and self.task == "text-classification":
            for idx in range(self.batch_size):
                if self.eos_token_id in int_tensor[idx]:
                    continue
                random_idx = random.randint(1, self.sequence_length - 1)
                int_tensor[idx][random_idx] = self.eos_token_id

        return int_tensor


class M2M100OnnxConfig(TextSeq2SeqOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",  # Used for the text-generation task past key values input generation.
        encoder_num_attention_heads="encoder_attention_heads",
        decoder_num_attention_heads="decoder_attention_heads",
        eos_token_id="eos_token_id",
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        BartDummyTextInputGenerator,
        {
            "feature-extraction": DummySeq2SeqDecoderTextInputGenerator,
            "text-generation": DummyDecoderTextInputGenerator,
        },
        {
            "feature-extraction": DummySeq2SeqPastKeyValuesGenerator,
            "text-generation": DummyPastKeyValuesGenerator,
        },
    )

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        task = "feature-extraction" if self.task != "text-generation" else "text-generation"
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1][task](
            self.task, self._normalized_config, **kwargs
        )
        if self.task != "text-generation":
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
                "input_ids": {0: "batch_size", 1: "sequence_length"},
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
            "feature-extraction": self.inputs_for_default_and_seq2seq_lm,
            "text2text-generation": self.inputs_for_default_and_seq2seq_lm,
            "text-generation": self.inputs_for_causal_lm,
            "other": self.inputs_for_other_tasks,
        }
        return inputs_properties.get(self.task, inputs_properties["other"])

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.task in ["feature-extraction", "text2text-generation"]:
            common_outputs = super().outputs
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs
            if self.use_past:
                # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
                for i in range(
                    self._normalized_config.encoder_num_layers
                    if self.task != "text-generation"
                    else self._normalized_config.decoder_num_layers
                ):
                    common_outputs[f"present.{i}.key"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}
                    common_outputs[f"present.{i}.value"] = {
                        0: "batch_size",
                        2: "past_sequence_length + sequence_length",
                    }
        return common_outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        # This will handle the attention mask padding when Bart is used for text-generation.
        if self.task == "text-generation":
            self.PAD_ATTENTION_MASK_TO_PAST = True

        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)

        # Setting it back to the default version.
        self.PAD_ATTENTION_MASK_TO_PAST = False
        return dummy_inputs

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        if self.task in ["feature-extraction", "text2text-generation"]:
            flattened_output = super().flatten_past_key_values(flattened_output, name, idx, t)
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self).flatten_past_key_values(
                flattened_output, name, idx, t
            )


class BartOnnxConfig(M2M100OnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # Bart now uses F.scaled_dot_product_attention by default for torch>=2.1.1.
    MIN_TORCH_VERSION = version.parse("2.1.2")


class MBartOnnxConfig(BartOnnxConfig):
    pass


class BlenderbotOnnxConfig(BartOnnxConfig):
    pass


class BlenderbotSmallOnnxConfig(BartOnnxConfig):
    pass


class BigBirdOnnxConfig(DistilBertOnnxConfig):
    pass


class BigBirdPegasusOnnxConfig(BartOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        inputs = super().inputs
        if self._config.attention_type == "block_sparse":
            # BigBirdPegasusEncoder creates its own attention_mask internally
            # https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L1875
            inputs.pop("attention_mask", None)
        return inputs


class PegasusOnnxConfig(BartOnnxConfig):
    pass


class MarianOnnxConfig(BartOnnxConfig):
    pass


class ViTOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    MIN_TORCH_VERSION = version.parse("1.11")
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs

        if self.task == "feature-extraction":
            common_outputs["last_hidden_state"] = {0: "batch_size"}

        return common_outputs


class VitPoseOnnxConfig(ViTOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (VitPoseDummyInputGenerator,)
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"pixel_values": {0: "batch_size"}}

    # Some VitPose models use multiple experts, which requires dataset_index to be provided.
    # So, we need to patch the model for export to provide the dataset_index.
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return VitPoseModelPatcher(self, model, model_kwargs=model_kwargs)


class CvTOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    ATOL_FOR_VALIDATION = 1e-2


class LevitOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class DeiTOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class BeitOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class ConvNextOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class ConvNextV2OnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class HieraOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class PvtOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class VitMAEOnnxConfig(ViTOnnxConfig):
    # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 11 is not supported.
    # Support for this operator was added in version 14, try exporting with this version.
    DEFAULT_ONNX_OPSET = 14


class VitMSNOnnxConfig(ViTOnnxConfig):
    # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 11 is not supported.
    # Support for this operator was added in version 14, try exporting with this version.
    DEFAULT_ONNX_OPSET = 14


class Dinov2OnnxConfig(ViTOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (Dinov2DummyInputGenerator,)


class MobileViTOnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4
    DEFAULT_ONNX_OPSET = 11


class RegNetOnnxConfig(ViTOnnxConfig):
    # This config has the same inputs as ViTOnnxConfig
    DEFAULT_ONNX_OPSET = 11


class ResNetOnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    DEFAULT_ONNX_OPSET = 11


class DetrOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "image-segmentation":
            return {
                "logits": {0: "batch_size", 1: "num_queries"},
                "pred_masks": {0: "batch_size", 1: "num_queries"},
            }
        else:
            return super().outputs


class TableTransformerOnnxConfig(DetrOnnxConfig):
    pass


class YolosOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class SwinOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class SwinV2OnnxConfig(SwinOnnxConfig):
    pass


class Swin2srOnnxConfig(SwinOnnxConfig):
    pass


class DptOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 14


class GlpnOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class PoolFormerOnnxConfig(ViTOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    ATOL_FOR_VALIDATION = 2e-3
    DEFAULT_ONNX_OPSET = 11


class SegformerOnnxConfig(YolosOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        outputs = super().outputs

        if self.task == "image-segmentation":
            outputs["logits"] = {0: "batch_size"}

        return outputs


class MobileNetV1OnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4
    DEFAULT_ONNX_OPSET = 11

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"pixel_values": {0: "batch_size"}}


class MobileNetV2OnnxConfig(MobileNetV1OnnxConfig):
    pass


class MaskFormerOnnxConfig(ViTOnnxConfig):
    # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::einsum' to ONNX opset version 11 is not supported.
    # Support for this operator was added in version 12, try exporting with this version.
    DEFAULT_ONNX_OPSET = 12

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "image-segmentation":
            return {
                "class_queries_logits": {0: "batch_size", 1: "num_queries"},
                "masks_queries_logits": {0: "batch_size", 1: "num_queries", 2: "height", 3: "width"},
            }
        else:
            return super().outputs

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return {
            "transformer_decoder_last_hidden_state": "last_hidden_state",
        }


class DonutSwinOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class TimmDefaultOnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    DEFAULT_ONNX_OPSET = 12

    def rename_ambiguous_inputs(self, inputs):
        #  The input name in the model signature is `x, hence the export input name is updated.
        model_inputs = {}
        model_inputs["x"] = inputs["pixel_values"]

        return model_inputs

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        return {"x": "pixel_values"}


class MgpstrOnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "char_logits": {0: "batch_size"},
            "bpe_logits": {0: "batch_size"},
            "wp_logits": {0: "batch_size"},
        }

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MgpstrModelPatcher(self, model, model_kwargs=model_kwargs)


class EfficientNetOnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs

        if self.task == "image-classification":
            common_outputs["logits"] = {0: "batch_size", 1: "num_classes"}

        return common_outputs


class SentenceTransformersTransformerOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DEFAULT_ONNX_OPSET = 14  # Some bottleneck transformers models require a specific ONNX opset to be successfully exported. We put a rather high opset here for the export to work for all architectures.

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "token_embeddings": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"},
        }

    # we need to set output_attentions=True in the model input to avoid calling
    # torch.nn.functional.scaled_dot_product_attention that is not supported by the ONNX export
    # due to the op torch.nn.functional.multi_head_attention_forward used for WavLM
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return SentenceTransformersTransformerPatcher(self, model, model_kwargs=model_kwargs)


class CLIPNormalizedConfig(NormalizedTextAndVisionConfig):
    TEXT_CONFIG = "text_config"
    VISION_CONFIG = "vision_config"


class CLIPVisionModelOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DEFAULT_ONNX_OPSET = 14  # scaled_dot_product_attention support was added in opset 14

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs
        common_outputs["last_hidden_state"] = {0: "batch_size"}
        common_outputs["pooler_output"] = {0: "batch_size"}

        return common_outputs

    def patch_model_for_export(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ModelPatcher":
        return CLIPModelPatcher(self, model, model_kwargs=model_kwargs)


class CLIPOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    DEFAULT_ONNX_OPSET = 14  # scaled_dot_product_attention support was added in opset 14

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

    def patch_model_for_export(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ModelPatcher":
        return CLIPModelPatcher(self, model, model_kwargs=model_kwargs)


class SentenceTransformersCLIPOnnxConfig(CLIPOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "text_embeds": {0: "text_batch_size"},
            "image_embeds": {0: "image_batch_size"},
        }

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return SentenceTransformersCLIPPatcher(self, model, model_kwargs=model_kwargs)


class CLIPTextWithProjectionOnnxConfig(TextEncoderOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-3
    # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        num_layers="num_hidden_layers",
        allow_new=True,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {
            "text_embeds": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }
        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs

    def patch_model_for_export(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ModelPatcher":
        return CLIPModelPatcher(self, model, model_kwargs=model_kwargs)


class CLIPTextOnnxConfig(CLIPTextWithProjectionOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }

        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs

    def patch_model_for_export(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ModelPatcher":
        return CLIPModelPatcher(self, model, model_kwargs=model_kwargs)


class SiglipNormalizedConfig(CLIPNormalizedConfig):
    pass


class SiglipOnnxConfig(CLIPOnnxConfig):
    NORMALIZED_CONFIG_CLASS = SiglipNormalizedConfig
    # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 13 is not supported.
    # Support for this operator was added in version 14, try exporting with this version.
    DEFAULT_ONNX_OPSET = 14

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size", 1: "sequence_length"},
            "pixel_values": {0: "image_batch_size", 1: "num_channels", 2: "height", 3: "width"},
            # NOTE: No attention_mask
        }


class SiglipTextWithProjectionOnnxConfig(CLIPTextWithProjectionOnnxConfig):
    pass


class SiglipTextOnnxConfig(CLIPTextOnnxConfig):
    pass


class SiglipVisionModelOnnxConfig(CLIPVisionModelOnnxConfig):
    # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 11 is not supported.
    # Support for this operator was added in version 14, try exporting with this version.
    DEFAULT_ONNX_OPSET = 14


class UNetOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4
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
        common_inputs = {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "timestep": {},  # a scalar with no dimension
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }

        # TODO : add addition_embed_type == text_image, image and image_embeds
        # https://github.com/huggingface/diffusers/blob/9366c8f84bfe47099ff047272661786ebb54721d/src/diffusers/models/unets/unet_2d_condition.py#L671
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs["text_embeds"] = {0: "batch_size"}
            common_inputs["time_ids"] = {0: "batch_size"}

        if getattr(self._normalized_config, "time_cond_proj_dim", None) is not None:
            common_inputs["timestep_cond"] = {0: "batch_size"}

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "out_sample": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return {
            "sample": "out_sample",
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            dummy_inputs["added_cond_kwargs"] = {
                "text_embeds": dummy_inputs.pop("text_embeds"),
                "time_ids": dummy_inputs.pop("time_ids"),
            }

        return dummy_inputs

    def ordered_inputs(self, model) -> Dict[str, Dict[int, str]]:
        inputs = super().ordered_inputs(model=model)
        # to fix mismatch between model forward signature and expected inputs
        # a dictionnary of additional embeddings `added_cond_kwargs` is expected depending on config.addition_embed_type
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            inputs["text_embeds"] = self.inputs["text_embeds"]
            inputs["time_ids"] = self.inputs["time_ids"]

        return inputs


class VaeEncoderOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 3e-4
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
            "sample": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent_parameters": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        }


class VaeDecoderOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4
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
            "latent_sample": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
        }


class T5EncoderOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4
    DEFAULT_ONNX_OPSET = 12  # int64 was supported since opset 12

    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self):
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }


class SD3TransformerOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 1e-4
    # The ONNX export of a CLIPText architecture, an other Stable Diffusion component, needs the Trilu
    # operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestepInputGenerator,
        DummyTransformerVisionInputGenerator,
        DummyTransformerTextInputGenerator,
    )

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        vocab_size="attention_head_dim",
        hidden_size="joint_attention_dim",
        projection_size="pooled_projection_dim",
        allow_new=True,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "hidden_states": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "step"},
        }

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "out_hidden_states": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def torch_to_onnx_output_map(self) -> Dict[str, str]:
        return {
            "sample": "out_hidden_states",
        }


class FluxTransformerOnnxConfig(SD3TransformerOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestepInputGenerator,
        DummyFluxTransformerVisionInputGenerator,
        DummyFluxTransformerTextInputGenerator,
    )

    @property
    def inputs(self):
        common_inputs = super().inputs
        common_inputs["hidden_states"] = {0: "batch_size", 1: "packed_height_width"}
        common_inputs["txt_ids"] = (
            {0: "sequence_length"} if is_diffusers_version(">=", "0.31.0") else {0: "batch_size", 1: "sequence_length"}
        )
        common_inputs["img_ids"] = (
            {0: "packed_height_width"}
            if is_diffusers_version(">=", "0.31.0")
            else {0: "batch_size", 1: "packed_height_width"}
        )

        if getattr(self._normalized_config, "guidance_embeds", False):
            common_inputs["guidance"] = {0: "batch_size"}

        return common_inputs

    @property
    def outputs(self):
        return {
            "out_hidden_states": {0: "batch_size", 1: "packed_height_width"},
        }


class GroupViTOnnxConfig(CLIPOnnxConfig):
    pass


class OwlViTOnnxConfig(CLIPOnnxConfig):
    # Sets the absolute tolerance to when validating the exported ONNX model against the
    # reference model.
    ATOL_FOR_VALIDATION = 1e-4
    MIN_TORCH_VERSION = version.parse("2.1")

    # needs einsum operator support, available since opset 12
    DEFAULT_ONNX_OPSET = 12

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        preprocessors: Optional[List[Any]] = None,
        legacy: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            legacy=legacy,
        )
        if task == "zero-shot-object-detection":
            logger.warning(
                "The batch size of this model will not be dynamic because non-maximum suppression is performed. "
                "Make sure to export the model with the same batch size as the one you will use at inference "
                "with `--batch_size N`."
            )

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        outputs = {}
        if self.task == "feature-extraction":
            outputs["logits_per_image"] = {0: "image_batch_size", 1: "text_batch_size"}
            outputs["logits_per_text"] = {0: "text_batch_size", 1: "image_batch_size"}
        elif self.task == "zero-shot-object-detection":
            outputs["logits"] = {0: "image_batch_size", 2: "num_queries"}
            outputs["pred_boxes"] = {0: "image_batch_size", 1: "num_boxes"}

        outputs["text_embeds"] = {0: "text_batch_size", 1: "max_text_queries"}
        outputs["image_embeds"] = {0: "image_batch_size"}
        return outputs


class OwlV2OnnxConfig(OwlViTOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.35.0")


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
        if self.task in ["text-classification", "question-answering"]:
            pixel_values_dynamic_axes = {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        else:
            pixel_values_dynamic_axes = {0: "batch_size", 1: "num_channels"}
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": pixel_values_dynamic_axes,
        }


class LiltOnnxConfig(TextAndVisionOnnxConfig):
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
        }


class Data2VecTextOnnxConfig(DistilBertOnnxConfig):
    pass


class Data2VecVisionOnnxConfig(ViTOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class Data2VecAudioOnnxConfig(AudioOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.
    NORMALIZED_CONFIG_CLASS = NormalizedConfig


class PerceiverOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        PerceiverDummyInputGenerator,
    ) + TextAndVisionOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        preprocessors: Optional[List[Any]] = None,
        legacy: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            legacy=legacy,
        )
        self.is_generating_dummy_inputs = False

    @property
    def inputs_name(self):
        if self.is_generating_dummy_inputs:
            if self.task in ["fill-mask", "text-classification"]:
                return "input_ids"
            else:
                return "pixel_values"
        else:
            return "inputs"

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.inputs_name in ["input_ids", "inputs"]:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
            return {
                "input_ids": dynamic_axis,
                "attention_mask": dynamic_axis,
            }
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length", 2: "width", 3: "height"}
            return {
                "pixel_values": dynamic_axis,
            }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        outputs = super().outputs

        if "logits" in outputs:
            # default is {0: "batch_size", 1: "sequence_length"} where sequence_length is dynamic axis
            # but perceiver always return the same max sequence length in the second dimension
            outputs["logits"] = {0: "batch_size"}

        return outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        self.is_generating_dummy_inputs = True
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs[self.inputs_name] = dummy_inputs.pop(self.inputs_name)
        return dummy_inputs


class HubertOnnxConfig(AudioOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class Wav2Vec2OnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class Wav2Vec2ConformerOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 11


class SEWOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class SEWDOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12


class UniSpeechOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class UniSpeechSATOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.


class WavLMOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    # we need to set output_attentions=True in the model input to avoid calling
    # torch.nn.functional.scaled_dot_product_attention that is not supported by the ONNX export
    # due to the op torch.nn.functional.multi_head_attention_forward used for WavLM
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return WavLMModelPatcher(self, model, model_kwargs=model_kwargs)


class ASTDummyAudioInputGenerator(DummyAudioInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size, self.normalized_config.max_length, self.normalized_config.num_mel_bins]
        if input_name == "input_values":
            return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework, dtype=float_dtype)
        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)


class ASTOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_mel_bins="num_mel_bins", max_length="max_length", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (ASTDummyAudioInputGenerator,)
    ATOL_FOR_VALIDATION = 1e-4
    DEFAULT_ONNX_OPSET = 14  # now uses F.scaled_dot_product_attention by default for torch>=2.1.1.

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"input_values": {0: "batch_size"}}


class MCTCTOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        input_features_per_channel="input_feat_per_channel", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (MCTCTDummyAudioInputGenerator,)
    DEFAULT_ONNX_OPSET = 13

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"input_features": {0: "batch_size", 1: "sequence_classification"}}


class MoonshineOnnxConfig(AudioToTextOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig

    # torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::triu' to ONNX opset version 11 is not supported.
    # Support for this operator was added in version 14, try exporting with this version.
    DEFAULT_ONNX_OPSET = 14

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}

        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_values"] = {0: "batch_size", 1: "num_samples"}

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
                self.add_past_key_values(common_inputs, direction="inputs")
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        return common_inputs


class WhisperOnnxConfig(AudioToTextOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # Whisper now uses F.scaled_dot_product_attention by default for torch>=2.1.1.

    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        feature_size="num_mel_bins",
        allow_new=True,
    )
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "audio-classification":
            common_inputs = {"input_features": {0: "batch_size"}}
        else:
            common_inputs = super().inputs
            if self._behavior is not ConfigBehavior.DECODER:
                common_inputs["input_features"] = {0: "batch_size"}  # Remove unnecessary dynamic axis.

            if is_transformers_version(">=", "4.43.0") and is_transformers_version("<", "4.46.0"):
                # since https://github.com/huggingface/transformers/pull/31166
                if self._behavior is not ConfigBehavior.ENCODER and self.use_past_in_inputs:
                    common_inputs["cache_position"] = {0: "decoder_sequence_length"}

            if self._behavior is ConfigBehavior.DECODER and not self.use_past_in_inputs:
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


class MusicgenOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # NOTE: Several warnings during the export are not to worry about:
    # * for i, indices in enumerate(codes): --> can be unrolled, fixed length (num_quantizers).
    # * max_pad = max(padding_left, padding_right) --> does not impact later controlflows.
    # if length <= max_pad:  --> appears to be always False for Musicgen.

    # opset>=13 needed to avoid a bug in T5 encoder SelfAttention.
    # opset>=14 needed for torch.tril export.
    DEFAULT_ONNX_OPSET = 14

    VARIANTS = {
        "text-conditional-with-past": "Exports Musicgen to ONNX to generate audio samples conditioned on a text prompt (Reference: https://huggingface.co/docs/transformers/model_doc/musicgen#text-conditional-generation). This uses the decoder KV cache. The following subcomponents are exported:\n\t\t* text_encoder.onnx: corresponds to the text encoder part in https://github.com/huggingface/transformers/blob/v4.39.1/src/transformers/models/musicgen/modeling_musicgen.py#L1457.\n\t\t* encodec_decode.onnx: corresponds to the Encodec audio encoder part in https://github.com/huggingface/transformers/blob/v4.39.1/src/transformers/models/musicgen/modeling_musicgen.py#L2472-L2480.\n\t\t* decoder_model.onnx: The Musicgen decoder, without past key values input, and computing cross attention. Not required at inference (use decoder_model_merged.onnx instead).\n\t\t* decoder_with_past_model.onnx: The Musicgen decoder, with past_key_values input (KV cache filled), not computing cross attention. Not required at inference (use decoder_model_merged.onnx instead).\n\t\t* decoder_model_merged.onnx: The two previous models fused in one, to avoid duplicating weights. A boolean input `use_cache_branch` allows to select the branch to use. In the first forward pass where the KV cache is empty, dummy past key values inputs need to be passed and are ignored with use_cache_branch=False.\n\t\t* build_delay_pattern_mask.onnx: A model taking as input `input_ids`, `pad_token_id`, `max_length`, and building a delayed pattern mask to the input_ids. Implements https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/musicgen/modeling_musicgen.py#L1054.",
    }
    # TODO: support audio-prompted generation (- audio_encoder_encode.onnx: corresponds to the audio encoder part in https://github.com/huggingface/transformers/blob/f01e1609bf4dba146d1347c1368c8c49df8636f6/src/transformers/models/musicgen/modeling_musicgen.py#L2087.\n\t)
    # With that, we have full Encodec support.
    DEFAULT_VARIANT = "text-conditional-with-past"

    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyCodegenDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummyEncodecInputGenerator,
        DummyIntGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        behavior: ConfigBehavior = ConfigBehavior.ENCODER,
        preprocessors: Optional[List[Any]] = None,
        model_part: Optional[Literal["text_encoder", "encodec_decode", "decoder", "build_delay_pattern_mask"]] = None,
        legacy: bool = False,
        variant: str = "text-conditional-with-past",
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            behavior=behavior,
            preprocessors=preprocessors,
            legacy=legacy,
        )
        if legacy:
            raise ValueError("Musicgen does not support legacy=True.")

        if (
            model_part in ["text_encoder", "encodec_decode", "build_delay_pattern_mask"]
            and behavior != ConfigBehavior.ENCODER
        ):
            raise ValueError(
                f"model_part is {model_part} and behavior is {behavior}. This is not supported, please open an issue at https://github.com/huggingface/optimum/issues."
            )

        if model_part == "decoder" and behavior != ConfigBehavior.DECODER:
            raise ValueError(
                f"model_part is {model_part} and behavior is {behavior}. This is not supported, please open an issue at https://github.com/huggingface/optimum/issues."
            )

        if behavior == ConfigBehavior.MONOLITH:
            raise ValueError(
                "Musicgen does not support behavior=ConfigBehavior.MONOLITH. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        if config.audio_encoder.model_type != "encodec":
            raise ValueError(
                f"Optimum ONNX export for Musicgen supports only Encodec as the audio encoder, got: {config.audio_encoder.model_type}. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        # Handling it would require to trace the audio_encoder.decode with torch.jit.script as we than have an unrollable loop.
        if config.audio_encoder.chunk_length_s is not None:
            raise ValueError(
                f"Musicgen ONNX export currently does not support audio_encoder.chunk_length_s not None (got {config.audio_encoder.chunk_length_s}). Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        self.model_part = model_part
        if self.model_part == "decoder":
            self.use_past = True  # without past is not supported, hard-code it here.

        self._normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS = NormalizedTextConfig(self._config.text_encoder)
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS = NormalizedConfig(self._config.decoder)
        self._normalized_config.decoder_num_layers = self._config.decoder.num_hidden_layers
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_layers = self._config.decoder.num_hidden_layers
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.encoder_num_attention_heads = (
            self._config.decoder.num_attention_heads
        )
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_attention_heads = (
            self._config.decoder.num_attention_heads
        )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        # Batched inference is not supported in Transformers.
        if self.model_part == "text_encoder":
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
                "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
            }
        elif self.model_part == "encodec_decode":
            # 0: always 1 for chunk_length_s=None, 2: num_quantizers fixed.
            common_inputs = {"audio_codes": {1: "batch_size", 3: "chunk_length"}}
        elif self.model_part == "build_delay_pattern_mask":
            common_inputs = {
                "input_ids": {0: "batch_size_x_num_codebooks"},
                "pad_token_id": {},
                "max_length": {},
            }
        elif self._behavior is ConfigBehavior.DECODER:
            # Naming it total_batch_size as in case we use guidance_scale, the dimension 0 may be larger than simply the batch_size.
            # Reference: https://github.com/huggingface/transformers/blob/31c575bcf13c2b85b65d652dd1b5b401f99be999/src/transformers/models/musicgen/modeling_musicgen.py#L1932-L1935
            common_inputs = {
                "decoder_input_ids": {0: "total_batch_size_x_num_codebooks"},
                "encoder_outputs": {0: "total_batch_size", 1: "encoder_sequence_length"},
                # MusicgenForConditionalGeneration maps attention_mask to encoder_attention_mask.
                "attention_mask": {
                    0: "batch_size",
                    1: "encoder_sequence_length",
                },
            }
            if self.use_past_in_inputs:
                # TODO: validate the axis name for attention_mask
                # common_inputs["attention_mask"][1] = "past_encoder_sequence_length + sequence_length"
                self.add_past_key_values(common_inputs, direction="inputs")
            else:
                common_inputs["decoder_input_ids"] = {
                    0: "total_batch_size_x_num_codebooks",
                    1: "decoder_sequence_length",
                }
        else:
            raise ValueError(
                "This should not happen. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {}

        if self.model_part == "text_encoder":
            common_outputs = super().outputs
        elif self.model_part == "encodec_decode":
            common_outputs["audio_values"] = {0: "batch_size", 2: "audio_length"}
        elif self.model_part == "build_delay_pattern_mask":
            common_outputs["input_ids_edited"] = {0: "total_batch_size_x_num_codebooks"}
            common_outputs["delay_pattern_mask"] = {0: "total_batch_size_x_num_codebooks", 1: "max_length"}
        elif self._behavior is ConfigBehavior.DECODER:
            common_outputs = super().outputs

            # MusicgenForConditionalGeneration output is named logits, not last_hidden_state.
            # Rename last_hidden_state -> logits while keeping the order.
            common_outputs = {
                "logits" if name == "last_hidden_state" else name: value for name, value in common_outputs.items()
            }
        else:
            raise ValueError(
                "This should not happen. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        return common_outputs

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_decoder_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_decoder_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "total_batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "total_batch_size", 2: decoder_sequence_name}

            if (
                self.is_merged is True
                or (self._behavior is ConfigBehavior.DECODER and not self.use_past_in_inputs)
                or direction == "inputs"
            ):
                # TODO: we only need to call it encoder_sequence_length_out in the merge case - but at torch.onnx.export()
                # time we have currently no case to check whether we will merge at a later step or not (self.is_merged is
                # not yet set at this time)
                inputs_or_outputs[f"{name}.{i}.encoder.key"] = {
                    0: "total_batch_size",
                    2: "encoder_sequence_length_out",
                }
                inputs_or_outputs[f"{name}.{i}.encoder.value"] = {
                    0: "total_batch_size",
                    2: "encoder_sequence_length_out",
                }

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MusicgenModelPatcher(self, model, model_kwargs=model_kwargs)

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}

    def post_process_exported_models(
        self,
        path: Path,
        models_and_onnx_configs: Dict[
            str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]
        ],
        onnx_files_subpaths: List[str],
    ):
        # Attempt to merge only if the decoder was exported without/with past, and ignore seq2seq models exported with text-generation task
        if "with-past" in self.variant:
            decoder_path = Path(path, onnx_files_subpaths[2])
            decoder_with_past_path = Path(path, onnx_files_subpaths[3])
            decoder_merged_path = Path(path, ONNX_DECODER_MERGED_NAME + ".onnx")
            try:
                from ...onnx import merge_decoders

                # The decoder with past does not output the cross attention past key values as they are constant,
                # hence the need for strict=False
                merge_decoders(
                    decoder=decoder_path,
                    decoder_with_past=decoder_with_past_path,
                    save_path=decoder_merged_path,
                    strict=False,
                )
            except Exception as e:
                raise Exception(f"Unable to merge decoders. Detailed error: {e}")

            # In order to do the validation of the two branches on the same file
            text_encoder_path = onnx_files_subpaths[0]
            encodec_decode_path = onnx_files_subpaths[1]
            build_delay_pattern_mask_path = onnx_files_subpaths[4]

            onnx_files_subpaths_new = [
                text_encoder_path,
                encodec_decode_path,
                decoder_merged_path.name,
                decoder_merged_path.name,
                build_delay_pattern_mask_path,
            ]

            # We validate the two branches of the decoder model then
            models_and_onnx_configs[ONNX_DECODER_NAME][1].is_merged = True
            models_and_onnx_configs[ONNX_DECODER_NAME][1].use_cache_branch = False

            # Past key values won't be generated by default, but added in the input
            models_and_onnx_configs[ONNX_DECODER_NAME][1].use_past_in_inputs = True

            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1].use_cache_branch = True
            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1].is_merged = True
        else:
            onnx_files_subpaths_new = onnx_files_subpaths

        return models_and_onnx_configs, onnx_files_subpaths_new

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: "DummyInputGenerator", input_name: str, framework: str, input_shapes: Dict
    ):
        if self.model_part == "build_delay_pattern_mask" and input_name == "input_ids":
            original_batch_size = dummy_input_gen.batch_size
            dummy_input_gen.batch_size = (
                original_batch_size * dummy_input_gen.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_codebooks
            )

            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )

            dummy_input_gen.batch_size = original_batch_size

        else:
            dummy_input = super().overwrite_shape_and_generate_input(
                dummy_input_gen, input_name, framework, input_shapes
            )

        return dummy_input


class SpeechT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    # TODO: Transformers batched generation for Speecht5 is BROKEN (https://github.com/huggingface/transformers/pull/25943),
    # so we won't support for now.
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="hidden_size",
        num_attention_heads="encoder_attention_heads",  # TODO: bugged in case encoder and decoder have different number of heads
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummySpeechT5InputGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator

    VARIANTS = {
        "with-past": "The export follows the Transformers implementation using the KV cache, with the following components exported:\n\t - encoder_model.onnx: corresponds to the encoding part in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2544-L2556.\n\t - decoder_model.onnx: corresponds to the decoder part in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2572-L2602.\n\t - decoder_with_past_model.onnx: same as the above, with past_key_values input (KV cache filled).\n\t - decoder_postnet_and_vocoder.onnx: Decoder speech postnet and vocoder (e.g. a SpeechT5HifiGan) to generate speech from the spectrogram, as in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2605-L2614.",
        "without-past": "The same as `with-past`, just without KV cache support. This is not a recommended export as slower than `with-past`.",
    }
    DEFAULT_VARIANT = "with-past"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        behavior: ConfigBehavior = ConfigBehavior.MONOLITH,
        preprocessors: Optional[List[Any]] = None,
        is_postnet_and_vocoder: bool = False,
        legacy: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            behavior=behavior,
            preprocessors=preprocessors,
            legacy=legacy,
        )
        if float_dtype == "fp16":
            raise ValueError(
                "The ONNX export of SpeechT5 in float16 is currently not supported due to a bug in PyTorch: https://github.com/pytorch/pytorch/pull/110078. Please open an issue in Optimum if you would like to export SpeechT5 in float16."
            )
        self.is_postnet_and_vocoder = is_postnet_and_vocoder

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}

        # Batched inference is not supported in Transformers.
        if self._behavior is ConfigBehavior.ENCODER:
            common_inputs["input_ids"] = {1: "encoder_sequence_length"}
        elif self._behavior is ConfigBehavior.DECODER:
            # NOTE: even when past is used, the decoder takes the full sequence as input as the prenet seem to require it:
            # https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2573
            common_inputs["output_sequence"] = {1: "decoder_sequence_length"}
            common_inputs["speaker_embeddings"] = {}  # No dynamic shape here.
            common_inputs["encoder_outputs"] = {1: "encoder_sequence_length"}
            common_inputs["encoder_attention_mask"] = {1: "encoder_sequence_length"}

            if self.variant == "with-past" and self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")
        elif self.is_postnet_and_vocoder:
            common_inputs["spectrogram"] = {0: "n_spectrums x reduction_factor"}
        else:
            raise ValueError(
                "self._behavior is neither encoder or decoder, and is_postnet_and_vocoder=False. This should not happen."
            )

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {}
        if self._behavior is ConfigBehavior.ENCODER:
            common_outputs["encoder_outputs"] = {1: "encoder_sequence_length"}
            common_outputs["encoder_attention_mask"] = {1: "encoder_sequence_length"}
        elif self._behavior is ConfigBehavior.DECODER:
            common_outputs["output_sequence_out"] = {1: "decoder_sequence_length + 1"}
            common_outputs["spectrum"] = {}  # No dynamic shape here.
            common_outputs["prob"] = {}  # No dynamic shape here.

            if self.variant == "with-past" and self.use_past:
                # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
                self.add_past_key_values(common_outputs, direction="outputs")
        elif self.is_postnet_and_vocoder:
            common_outputs["waveform"] = {0: "n_samples"}
        else:
            raise ValueError(
                "self._behavior is neither encoder or decoder, and is_postnet_and_vocoder=False. This should not happen."
            )

        return common_outputs

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return SpeechT5ModelPatcher(self, model, model_kwargs=model_kwargs)

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        return {"encoder_outputs": "encoder_hidden_states"}

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: "DummyInputGenerator", input_name: str, framework: str, input_shapes: Dict
    ):
        dummy_input_gen.batch_size = 1
        dummy_input = dummy_input_gen.generate(
            input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
        )
        return dummy_input

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_decoder_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_decoder_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {2: decoder_sequence_name}

            if (
                self.is_merged is True
                or (self._behavior is ConfigBehavior.DECODER and not self.use_past_in_inputs)
                or direction == "inputs"
            ):
                inputs_or_outputs[f"{name}.{i}.encoder.key"] = {2: "encoder_sequence_length_out"}
                inputs_or_outputs[f"{name}.{i}.encoder.value"] = {2: "encoder_sequence_length_out"}


class VitsOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size", 1: "sequence_length"},
            "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "waveform": {0: "text_batch_size", 1: "n_samples"},
            "spectrogram": {0: "text_batch_size", 2: "num_bins"},
        }


class Speech2TextDummyAudioInputGenerator(DummyAudioInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size, self.sequence_length, self.normalized_config.input_features_per_channel]
        if input_name == "input_features":
            return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework, dtype=float_dtype)
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
                1: f"encoder_sequence_length / {(2 * self._config.num_conv_layers)}",
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
            ] = f"{common_outputs['last_hidden_state'][1]} / {(2 * self._config.num_conv_layers)}"
        return common_outputs


# TODO: Replace the TextSeq2SeqOnnxConfig inheritance with VisionToTextOnnxConfig when added.
# The change below however does not affect the export for the model
class TrOCROnnxConfig(TextSeq2SeqOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",
        decoder_num_attention_heads="decoder_attention_heads",
        hidden_size="hidden_size",
    )


class VisionEncoderDecoderOnnxConfig(EncoderDecoderBaseOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    ATOL_FOR_VALIDATION = 1e-3
    DEFAULT_ONNX_OPSET = 14  # uses SDPA in Transformers, hence opset>=14.

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyVisionEncoderDecoderPastKeyValuesGenerator)

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

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == ConfigBehavior.ENCODER:
            # Some encoders have static sequence length so it is useful to rely on the encoder ONNX config to grab this information.
            return self._encoder_onnx_config.outputs
        else:
            # Ideally, we would want here to have self._decoder_onnx_config.outputs, which is currently not possible
            # as we hard-code the task to feature-extraction, that has the wrong output names (e.g. mbart does not support document-question-answering
            # so we can not initializer MBartONNXConfig with document-question-answering).
            return super().outputs

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return VisionEncoderDecoderPatcher(self, model, model_kwargs=model_kwargs)


class SamOnnxConfig(OnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.29.0.dev0")
    # Since ransformers 4.32.0, SAM uses repeat_interleave op that is broken in PyTorch 2.0.1: https://github.com/pytorch/pytorch/issues/100429
    MIN_TORCH_VERSION = version.parse("2.0.99")
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyPointsGenerator, DummyVisionEmbeddingsGenerator)
    DEFAULT_ONNX_OPSET = 13  # Opset 12 for repeat_interleave falls back on the opset 9 implem, that raises Unsupported: ONNX export of repeat_interleave in opset 9.
    VARIANTS = {
        "monolith": "All the SAM model components are exported as a single model.onnx.",
        "split": "The vision encoder is exported as a separate vision_encoder.onnx, and the prompt encoder and mask decoder are exported as a prompt_encoder_mask_decoder.onnx. This allows to encoder the image only once for multiple point queries.",
    }
    DEFAULT_VARIANT = "split"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        variant: str = "split",
        vision_encoder: Optional[bool] = None,
        preprocessors: Optional[List[Any]] = None,
        legacy: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            legacy=legacy,
        )
        self.variant = variant
        self.vision_encoder = vision_encoder
        self._normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig(self._config.vision_config)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.variant == "monolith":
            inputs = {
                "pixel_values": {0: "batch_size"},
                "input_points": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
                "input_labels": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
            }
        else:
            if self.vision_encoder:
                inputs = {"pixel_values": {0: "batch_size"}}
            else:
                inputs = {
                    "image_positional_embeddings": {0: "batch_size"},
                    "image_embeddings": {0: "batch_size"},
                    "input_points": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
                    "input_labels": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
                }
        return inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.variant == "split" and self.vision_encoder:
            return {"image_embeddings": {0: "batch_size"}, "image_positional_embeddings": {0: "batch_size"}}
        else:
            return {
                "iou_scores": {0: "batch_size", 1: "point_batch_size"},
                "pred_masks": {0: "batch_size", 1: "point_batch_size"},
            }

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return SAMModelPatcher(self, model, model_kwargs=model_kwargs)


class Pix2StructNormalizedConfig(NormalizedSeq2SeqConfig):
    ENCODER_NUM_LAYERS = "vision_config.num_hidden_layers"
    DECODER_NUM_LAYERS = "text_config.num_layers"
    ENCODER_NUM_ATTENTION_HEADS = "vision_config.num_attention_heads"
    DECODER_NUM_ATTENTION_HEADS = "text_config.num_heads"
    HIDDEN_SIZE = "text_config.hidden_size"  # TODO: Isn't this bug prone?
    VOCAB_SIZE = "text_config.vocab_size"


class Pix2StructOnnxConfig(OnnxSeq2SeqConfigWithPast):
    NORMALIZED_CONFIG_CLASS = Pix2StructNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummyPix2StructInputGenerator,
    )

    DEFAULT_ONNX_OPSET = 14  # use 'aten::triu' now which is opset 14

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if is_transformers_version("==", "4.46.0") and self._behavior is ConfigBehavior.DECODER:
            logger.error(
                "Found transformers v4.46.0 while trying to exporting a Pix2Struct model, this specific version of transformers is not supported. "
                "Please upgrade to v4.46.1 or higher, or downgrade your transformers version"
            )

    @property
    def inputs(self):
        common_inputs = {}
        common_inputs["attention_mask"] = {0: "batch_size"}

        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["flattened_patches"] = {0: "batch_size"}

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

        if self._behavior is ConfigBehavior.DECODER:
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

            common_inputs["encoder_outputs"] = {0: "batch_size"}

            # Contrary to other seq2seq archs as t5 and bart, Pix2Struct DO make use of the decoder_attention_mask input.
            common_inputs["decoder_attention_mask"] = {0: "batch_size", 1: "past_sequence_length + 1"}

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior is ConfigBehavior.ENCODER:
            common_outputs = {
                "last_hidden_state": {0: "batch_size"}
            }  # The last hidden state dim=1 is constant, no need for it to be dynamic.
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs

        # Renaming the outputs axes properly.
        for name, axes_names in common_outputs.items():
            if self._behavior is ConfigBehavior.ENCODER or "encoder" in name:
                sequence_name = "encoder_sequence_length"
            else:
                sequence_name = "decoder_sequence_length"

            new_axes_names = {}
            for axis_idx, axis_name in axes_names.items():
                if "sequence" in axis_name:
                    if self.use_past_in_inputs is False or self.is_merged is True:
                        new_axes_names[axis_idx] = sequence_name
                    else:
                        # Trick to force it since ONNX sometimes infer a dynamic axis where it's not.
                        new_axes_names[axis_idx] = "1"
                else:
                    new_axes_names[axis_idx] = axis_name
            common_outputs[name] = new_axes_names

        if self.use_past:
            # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
            self.add_past_key_values(common_outputs, direction="outputs")

        return common_outputs

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if self._behavior is ConfigBehavior.DECODER:
            reference_model_inputs["input_ids"] = reference_model_inputs.pop("decoder_input_ids")

        if onnx_input_names is not None:
            if "encoder_outputs" in reference_model_inputs:
                if "encoder_hidden_states" in onnx_input_names:
                    reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]
                else:
                    reference_model_inputs.pop("encoder_outputs")
        else:
            # TODO: remove this else in optimum 2.0 and make onnx_input_names a required argument
            # Pix2Struct requires encoder_hidden_states as an input for both the without/with past models,
            # which is different than other architectures that require it only for the without past case
            reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]

        return super().generate_dummy_inputs_for_validation(reference_model_inputs)

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        dummy_inputs_generators = []
        dummy_inputs_generators.append(self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config))

        if self._preprocessors is None or len(self._preprocessors) < 2:
            raise ValueError(
                f"Preprocessors for pix2struct need to be available for the ONNX export to infer input static shapes. Got: {self._preprocessors}"
            )

        encoder_sequence_length = self._preprocessors[1].image_processor.max_patches
        # A hack for DummyPix2StructInputGenerator to gain access to the preprocessors.
        # TODO: we should probably pass preprocessors to all dummy input generators.
        kwargs["preprocessors"] = self._preprocessors
        for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES[1:]:
            dummy_inputs_generators.append(
                cls_(self.task, self._normalized_config, encoder_sequence_length=encoder_sequence_length, **kwargs)
            )

        return dummy_inputs_generators

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: "DummyInputGenerator", input_name: str, framework: str, input_shapes: Dict
    ):
        if self._preprocessors is None or len(self._preprocessors) < 2:
            raise ValueError(
                f"Preprocessors for pix2struct need to be available for the ONNX export to infer input static shapes. Got: {self._preprocessors}"
            )

        # models from TextSeq2SeqOnnxConfig use decoder_input_ids as input name
        # while models from TextDecoderOnnxConfig use input_ids, hence the check for both
        if (
            self.use_past
            and self.use_past_in_inputs
            and self.use_cache_branch is not False
            and input_name in ["decoder_input_ids", "input_ids"]
        ):
            sequence_length = dummy_input_gen.sequence_length
            # Use a sequence length of 1 when the KV cache is already populated.
            dummy_input_gen.sequence_length = 1
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )
            dummy_input_gen.sequence_length = sequence_length
        elif input_name in ["encoder_outputs", "attention_mask"]:
            # pix2struct takes inputs whose so-called sequence length is **static** to max_patches, so we do NOT use
            # the passed sequence_length that behaves as a dynamic shape.
            original_seq_length = dummy_input_gen.sequence_length
            dummy_input_gen.sequence_length = self._preprocessors[1].image_processor.max_patches
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )
            dummy_input_gen.sequence_length = original_seq_length
        else:
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )

        return dummy_input


class EncoderDecoderOnnxConfig(EncoderDecoderBaseOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig

    DEFAULT_ONNX_OPSET = 14  # uses SDPA in Transformers, hence opset>=14.


class PatchTSTOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTimeSeriesForecastingConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyPatchTSTInputGenerator,)
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"past_values": {0: "batch_size", 1: "sequence_length"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "feature-extraction":
            return {"last_hidden_state": {0: "batch_size"}}
        else:
            return super().outputs


class PatchTSMixerOnnxConfig(PatchTSTOnnxConfig):
    pass


class RTDetrOnnxConfig(ViTOnnxConfig):
    # Export the operator 'aten::grid_sampler' to ONNX fails under opset 16.
    # Support for this operator was added in version 16.
    DEFAULT_ONNX_OPSET = 16
    ATOL_FOR_VALIDATION = 1e-5

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        }

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        min_image_size = int(math.ceil(self._config.num_queries / 32) * 32)
        if kwargs["height"] < min_image_size:
            warnings.warn(
                f"Exporting model with image `height={kwargs['height']}` which is less than "
                f"minimal {min_image_size}, setting `height` to {min_image_size}."
            )
            kwargs["height"] = min_image_size
        if kwargs["width"] < min_image_size:
            warnings.warn(
                f"Exporting model with image `width={kwargs['width']}` which is less than "
                f"minimal {min_image_size}, setting `width` to {min_image_size}."
            )
            kwargs["width"] = min_image_size
        return super()._create_dummy_input_generator_classes(**kwargs)


class RTDetrV2OnnxConfig(RTDetrOnnxConfig):
    pass
