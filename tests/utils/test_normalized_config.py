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

from transformers import PLBartConfig

from optimum.utils.normalized_config import NormalizedConfigManager


class NormalizedConfigManagerTest(TestCase):
    def test_plbart(self):
        config = PLBartConfig(
            d_model=64,
            encoder_attention_heads=4,
            decoder_attention_heads=8,
            encoder_layers=2,
            decoder_layers=3,
        )
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.hidden_size, 64)
        self.assertEqual(normalized_config.num_attention_heads, 4)
        self.assertEqual(normalized_config.num_layers, 2)
