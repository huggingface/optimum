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
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "configuration": [
        "CalibrationConfig",
        "AutoCalibrationConfig",
        "QuantizationMode",
        "AutoQuantizationConfig",
        "OptimizationConfig",
        "AutoOptimizationConfig",
        "ORTConfig",
    ],
    "modeling_ort": [
        "ORTModel",
        "ORTModelForCustomTasks",
        "ORTModelForFeatureExtraction",
        "ORTModelForImageClassification",
        "ORTModelForMultipleChoice",
        "ORTModelForQuestionAnswering",
        "ORTModelForSemanticSegmentation",
        "ORTModelForSequenceClassification",
        "ORTModelForTokenClassification",
    ],
    "modeling_seq2seq": ["ORTModelForSeq2SeqLM", "ORTModelForSpeechSeq2Seq"],
    "modeling_decoder": ["ORTModelForCausalLM"],
    "optimization": ["ORTOptimizer"],
    "quantization": ["ORTQuantizer"],
    "trainer": ["ORTTrainer"],
    "trainer_seq2seq": ["ORTSeq2SeqTrainer"],
    "training_args": ["ORTTrainingArguments"],
    "training_args_seq2seq": ["ORTSeq2SeqTrainingArguments"],
    "utils": [
        "ONNX_DECODER_NAME",
        "ONNX_DECODER_WITH_PAST_NAME",
        "ONNX_ENCODER_NAME",
        "ONNX_WEIGHTS_NAME",
        "ORTQuantizableOperator",
    ],
}


# Direct imports for type-checking
if TYPE_CHECKING:
    from .configuration import ORTConfig
    from .modeling_decoder import ORTModelForCausalLM
    from .modeling_ort import (
        ORTModel,
        ORTModelForCustomTasks,
        ORTModelForFeatureExtraction,
        ORTModelForImageClassification,
        ORTModelForMultipleChoice,
        ORTModelForQuestionAnswering,
        ORTModelForSemanticSegmentation,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )
    from .modeling_seq2seq import ORTModelForSeq2SeqLM, ORTModelForSpeechSeq2Seq
    from .optimization import ORTOptimizer
    from .quantization import ORTQuantizer
    from .trainer import ORTTrainer
    from .trainer_seq2seq import ORTSeq2SeqTrainer
    from .training_args import ORTTrainingArguments
    from .training_args_seq2seq import ORTSeq2SeqTrainingArguments
    from .utils import (
        ONNX_DECODER_NAME,
        ONNX_DECODER_WITH_PAST_NAME,
        ONNX_ENCODER_NAME,
        ONNX_WEIGHTS_NAME,
        ORTQuantizableOperator,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
