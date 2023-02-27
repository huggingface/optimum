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
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig


class BertTFLiteConfig(TextEncoderTFliteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


class AlbertTFLiteConfig(BertTFLiteConfig):
    pass


class ConvBertTFLiteConfig(BertTFLiteConfig):
    pass


class ElectraTFLiteConfig(BertTFLiteConfig):
    pass


class RoFormerTFLiteConfig(BertTFLiteConfig):
    pass


class MobileBertTFLiteConfig(BertTFLiteConfig):
    pass


class XLMTFLiteConfig(BertTFLiteConfig):
    pass


class DistilBertTFLiteConfig(BertTFLiteConfig):
    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]


class MPNetTFLiteConfig(DistilBertTFLiteConfig):
    pass


class RobertaTFLiteConfig(DistilBertTFLiteConfig):
    pass


class CamembertTFLiteConfig(DistilBertTFLiteConfig):
    pass


class FlaubertTFLiteConfig(BertTFLiteConfig):
    pass


class XLMRobertaTFLiteConfig(DistilBertTFLiteConfig):
    pass


# TODO: no TensorFlow implementation, but a Jax implementation is available.
# Support the export once the Jax export to TFLite is more mature.
# class BigBirdTFLiteConfig(DistilBertTFLiteConfig):
#     pass


class DebertaTFLiteConfig(BertTFLiteConfig):
    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            # We remove token type ids.
            common_inputs.pop(-1)
        return common_inputs


class DebertaV2TFLiteConfig(DebertaTFLiteConfig):
    pass


class ResNetTFLiteConfig(VisionTFLiteConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("resnet")

    @property
    def inputs(self) -> List[str]:
        return ["pixel_values"]
