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

from transformers import Data2VecTextConfig

from optimum.utils.normalized_config import NormalizedConfigManager, NormalizedTextConfig


class NormalizedConfigManagerTest(unittest.TestCase):
    def test_data2vec_text_is_registered(self):
        config_class = NormalizedConfigManager.get_normalized_config_class("data2vec-text")
        self.assertIs(config_class, NormalizedTextConfig)

        normalized_config = config_class(
            Data2VecTextConfig(hidden_size=48, num_attention_heads=6, num_hidden_layers=3)
        )
        self.assertEqual(normalized_config.hidden_size, 48)
        self.assertEqual(normalized_config.num_attention_heads, 6)
        self.assertEqual(normalized_config.num_layers, 3)
