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

import unittest

from transformers import LayoutLMv3Config

from optimum.utils.normalized_config import NormalizedConfigManager, NormalizedTextConfig


class NormalizedConfigManagerTest(unittest.TestCase):
    def test_layoutlmv3_is_registered(self):
        config_class = NormalizedConfigManager.get_normalized_config_class("layoutlmv3")
        self.assertIs(config_class, NormalizedTextConfig)

    def test_layoutlmv3_normalizes_expected_attributes(self):
        config = LayoutLMv3Config()
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class("layoutlmv3")
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.hidden_size, config.hidden_size)
        self.assertEqual(normalized_config.num_attention_heads, config.num_attention_heads)
        self.assertEqual(normalized_config.num_layers, config.num_hidden_layers)
