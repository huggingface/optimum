# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from unittest import TestCase

from transformers import CLIPConfig

from optimum.utils.normalized_config import NormalizedConfigManager


class NormalizedConfigTest(TestCase):
    def test_clip_text_and_vision_attributes(self):
        config = CLIPConfig()
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.vocab_size, config.text_config.vocab_size)
        self.assertEqual(normalized_config.hidden_size, config.text_config.hidden_size)
        self.assertEqual(normalized_config.num_layers, config.text_config.num_hidden_layers)
        self.assertEqual(normalized_config.num_attention_heads, config.text_config.num_attention_heads)
        self.assertEqual(normalized_config.image_size, config.vision_config.image_size)
        self.assertEqual(normalized_config.num_channels, config.vision_config.num_channels)
