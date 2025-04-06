#  Copyright 2021 The HuggingFace Team. All rights reserved.
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


from .constant import (
    CONFIG_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
    ONNX_WEIGHTS_NAME,
)
from .import_utils import (
    DIFFUSERS_MINIMUM_VERSION,
    ORT_QUANTIZE_MINIMUM_VERSION,
    TORCH_MINIMUM_VERSION,
    TRANSFORMERS_MINIMUM_VERSION,
    check_if_diffusers_greater,
    check_if_pytorch_greater,
    check_if_torch_greater,
    check_if_transformers_greater,
    is_accelerate_available,
    is_auto_gptq_available,
    is_datasets_available,
    is_diffusers_available,
    is_diffusers_version,
    is_gptqmodel_available,
    is_onnx_available,
    is_onnxruntime_available,
    is_pydantic_available,
    is_sentence_transformers_available,
    is_tf_available,
    is_timm_available,
    is_torch_available,
    is_torch_onnx_support_available,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    require_numpy_strictly_lower,
)
from .input_generators import (
    DEFAULT_DUMMY_SHAPES,
    DTYPE_MAPPER,
    BloomDummyPastKeyValuesGenerator,
    Dinov2DummyInputGenerator,
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyCodegenDecoderTextInputGenerator,
    DummyDecisionTransformerInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyEncodecInputGenerator,
    DummyFluxTransformerTextInputGenerator,
    DummyFluxTransformerVisionInputGenerator,
    DummyInputGenerator,
    DummyIntGenerator,
    DummyLabelsGenerator,
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
    MultiQueryPastKeyValuesGenerator,
    PerceiverDummyInputGenerator,
    VitPoseDummyInputGenerator,
)
from .modeling_utils import recurse_getattr, recurse_setattr
from .normalized_config import (
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedTextConfigWithGQA,
    NormalizedTimeSeriesForecastingConfig,
    NormalizedVisionConfig,
)
