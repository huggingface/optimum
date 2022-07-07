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
from enum import Enum
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


class ORTQuantizableOperator(Enum):
    # Common ops
    Gather = "Gather"
    Transpose = "Transpose"
    EmbedLayerNormalizationQuant = "EmbedLayerNormalization"

    # QLinearOps
    Conv = "Conv"
    MatMul = "MatMul"
    Add = "Add"
    Mul = "Mul"
    Relu = "Relu"
    Clip = "Clip"
    LeakyRelu = "LeakyRelu"
    Sigmoid = "Sigmoid"
    MaxPool = "MaxPool"
    GlobalAveragePool = "GlobalAveragePool"
    Split = "Split"
    Pad = "Pad"
    Reshape = "Reshape"
    Squeeze = "Squeeze"
    Unsqueeze = "Unsqueeze"
    Resize = "Resize"
    AveragePool = "AveragePool"
    Concat = "Concat"


AUTO_MINIMUM_SUPPORTED_ONNX_OPSET = None

# This value is used to indicate ORT which axis it should use to quantize an operator "per-channel"
ORT_DEFAULT_CHANNEL_FOR_OPERATORS = {"MatMul": 1}
ORT_FULLY_CONNECTED_OPERATORS = ["MatMul", "Add"]

_import_structure = {
    "configuration": ["ORTConfig"],
    "model": ["ORTModel"],
    "modeling_ort": [
        "ORTModelForCausalLM",
        "ORTModelForFeatureExtraction",
        "ORTModelForImageClassification",
        "ORTModelForQuestionAnswering",
        "ORTModelForSequenceClassification",
        "ORTModelForTokenClassification",
    ],
    "modeling_seq2seq": ["ORTModelForSeq2SeqLM"],
    "optimization": ["ORTOptimizer"],
    "quantization": ["ORTQuantizer"],
    "trainer": ["ORTTrainer"],
    "trainer_seq2seq": ["ORTSeq2SeqTrainer"],
    "utils": ["ONNX_DECODER_NAME", "ONNX_DECODER_WITH_PAST_NAME", "ONNX_ENCODER_NAME", "ONNX_WEIGHTS_NAME"],
}


# Direct imports for type-checking
if TYPE_CHECKING:
    from .configuration import ORTConfig
    from .model import ORTModel
    from .modeling_ort import (
        ORTModelForCausalLM,
        ORTModelForFeatureExtraction,
        ORTModelForImageClassification,
        ORTModelForQuestionAnswering,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )
    from .modeling_seq2seq import ORTModelForSeq2SeqLM
    from .optimization import ORTOptimizer
    from .quantization import ORTQuantizer
    from .trainer import ORTTrainer
    from .trainer_seq2seq import ORTSeq2SeqTrainer
    from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME, ONNX_WEIGHTS_NAME
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
