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


from .import_utils import (
    ORT_QUANTIZE_MINIMUM_VERSION,
    TORCH_MINIMUM_VERSION,
    check_if_pytorch_greater,
    check_if_transformers_greater,
    is_accelerate_available,
    is_diffusers_available,
    is_onnxruntime_available,
    is_pydantic_available,
    is_torch_onnx_support_available,
    torch_version,
)
from .input_generators import (
    DEFAULT_DUMMY_SHAPES,
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyTrainingLabelsInputGenerator,
    DummyVisionInputGenerator,
)
from .normalized_config import (
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)


CONFIG_NAME = "config.json"
