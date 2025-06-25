# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Model specific TensorFlow Lite configurations."""


from typing import List

from ...utils.normalized_config import NormalizedConfigManager
from ..tasks import TasksManager
from .base import QuantizationApproach
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig


register_tasks_manager_tflite = TasksManager.create_register("tflite")


COMMON_TEXT_TASKS = [
    "feature-extraction",
    "fill-mask",
    "multiple-choice",
    "question-answering",
    "text-classification",
    "token-classification",
]


@register_tasks_manager_tflite("tflite", *COMMON_TEXT_TASKS)
class BertTFLiteConfig(TextEncoderTFliteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")
    # INT8x16 not supported because of the CAST op.
    SUPPORTED_QUANTIZATION_APPROACHES = (
        QuantizationApproach.INT8_DYNAMIC,
        QuantizationApproach.INT8,
        QuantizationApproach.FP16,
    )

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


@register_tasks_manager_tflite("ablbert", *COMMON_TEXT_TASKS)
class AlbertTFLiteConfig(BertTFLiteConfig):
    pass


@register_tasks_manager_tflite("convbert", *COMMON_TEXT_TASKS)
class ConvBertTFLiteConfig(BertTFLiteConfig):
    pass


@register_tasks_manager_tflite("electra", *COMMON_TEXT_TASKS)
class ElectraTFLiteConfig(BertTFLiteConfig):
    pass


@register_tasks_manager_tflite("roformer", *COMMON_TEXT_TASKS)
class RoFormerTFLiteConfig(BertTFLiteConfig):
    # INT8x16 not supported because of the CAST and NEG ops.
    pass


@register_tasks_manager_tflite("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertTFLiteConfig(BertTFLiteConfig):
    pass


@register_tasks_manager_tflite("xlm", *COMMON_TEXT_TASKS)
class XLMTFLiteConfig(BertTFLiteConfig):
    pass


@register_tasks_manager_tflite("distilbert", *COMMON_TEXT_TASKS)
class DistilBertTFLiteConfig(BertTFLiteConfig):
    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]


@register_tasks_manager_tflite("mpnet", *COMMON_TEXT_TASKS)
class MPNetTFLiteConfig(DistilBertTFLiteConfig):
    pass


@register_tasks_manager_tflite("roberta", *COMMON_TEXT_TASKS)
class RobertaTFLiteConfig(DistilBertTFLiteConfig):
    pass


@register_tasks_manager_tflite("camembert", *COMMON_TEXT_TASKS)
class CamembertTFLiteConfig(DistilBertTFLiteConfig):
    pass


@register_tasks_manager_tflite("flaubert", *COMMON_TEXT_TASKS)
class FlaubertTFLiteConfig(BertTFLiteConfig):
    pass


@register_tasks_manager_tflite("xlm-roberta", *COMMON_TEXT_TASKS)
class XLMRobertaTFLiteConfig(DistilBertTFLiteConfig):
    SUPPORTED_QUANTIZATION_APPROACHES = {
        "default": BertTFLiteConfig.SUPPORTED_QUANTIZATION_APPROACHES,
        # INT8 quantization on question-answering is producing various errors depending on the model size and
        # calibration dataset:
        # - GATHER index out of bound
        # - (CUMSUM) failed to invoke
        # TODO => Needs to be investigated.
        "question-answering": (QuantizationApproach.INT8_DYNAMIC, QuantizationApproach.FP16),
    }


# TODO: no TensorFlow implementation, but a Jax implementation is available.
# Support the export once the Jax export to TFLite is more mature.
# class BigBirdTFLiteConfig(DistilBertTFLiteConfig):
#     pass


@register_tasks_manager_tflite(
    "deberta",
    *["feature-extraction", "fill-mask", "text-classification", "token-classification", "question-answering"],
)
class DebertaTFLiteConfig(BertTFLiteConfig):
    # INT8 quantization is producing a segfault error.
    SUPPORTED_QUANTIZATION_APPROACHES = (QuantizationApproach.INT8_DYNAMIC, QuantizationApproach.FP16)

    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            # We remove token type ids.
            common_inputs.pop(-1)
        return common_inputs


@register_tasks_manager_tflite(
    "deberta-v2",
    *["feature-extraction", "fill-mask", "text-classification", "token-classification", "question-answering"],
)
class DebertaV2TFLiteConfig(DebertaTFLiteConfig):
    pass


@register_tasks_manager_tflite("resnet", *["feature-extraction", "image-classification"])
class ResNetTFLiteConfig(VisionTFLiteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("resnet")

    @property
    def inputs(self) -> List[str]:
        return ["pixel_values"]
