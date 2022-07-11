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
from typing import TYPE_CHECKING, Type

from ..models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from .image_classification import ImageClassificationProcessing
from .question_answering import QuestionAnsweringProcessing
from .text_classification import TextClassificationProcessing
from .token_classification import TokenClassificationProcessing


if TYPE_CHECKING:
    from transformers import PreTrainedModel

TASK_PROCESSING_MAP = {
    "text-classification": TextClassificationProcessing,
    "token-classification": TokenClassificationProcessing,
    "question-answering": QuestionAnsweringProcessing,
    "image-classification": ImageClassificationProcessing,
}

_MODEL_MAPPING_TO_DATASET_PROCESSING = {}
for _, model_name in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
    _MODEL_MAPPING_TO_DATASET_PROCESSING[model_name] = "text-Classification"
for _, model_name in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES:
    _MODEL_MAPPING_TO_DATASET_PROCESSING[model_name] = "token-Classification"
for _, model_name in MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES:
    _MODEL_MAPPING_TO_DATASET_PROCESSING[model_name] = "question-answering"
for _, model_name in MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES:
    _MODEL_MAPPING_TO_DATASET_PROCESSING[model_name] = "image-classification"


class AutoDatasetProcessing:
    @staticmethod
    def from_model(model: "PreTrainedModel") -> Type:
        names = [model.__class__.__name__] + list(map(str, model.__class__.mro()))
        task = None
        for name in names:
            task = _MODEL_MAPPING_TO_DATASET_PROCESSING.get(name, None)
            if task is not None:
                break
        if task is None:
            raise ValueError(f"Could not infer the DataProcessing class for model class {model.__class__.__name__}")
        return TASK_PROCESSING_MAP[task]
