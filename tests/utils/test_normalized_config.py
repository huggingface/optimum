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

from transformers import DetrConfig

from optimum.utils.normalized_config import NormalizedConfigManager, NormalizedTextConfig


class NormalizedConfigManagerTest(unittest.TestCase):
    def test_detr_is_registered(self):
        config_class = NormalizedConfigManager.get_normalized_config_class("detr")
        self.assertIs(config_class, NormalizedTextConfig)

    def test_detr_normalizes_expected_attributes(self):
        # DetrConfig exposes its transformer stats under d_model / encoder_attention_heads /
        # encoder_layers, but transformers' own attribute_map on DetrConfig aliases these to
        # hidden_size / num_attention_heads / num_hidden_layers, so plain NormalizedTextConfig
        # resolves them with no custom optimum-side mapping needed.
        config = DetrConfig()
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class("detr")
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.hidden_size, config.d_model)
        self.assertEqual(normalized_config.num_attention_heads, config.encoder_attention_heads)
        self.assertEqual(normalized_config.num_layers, config.encoder_layers)
