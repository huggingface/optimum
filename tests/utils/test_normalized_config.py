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

from transformers import GroupViTConfig, Pix2StructConfig

from optimum.utils.normalized_config import NormalizedConfigManager, Pix2StructNormalizedTextConfig


class NormalizedConfigManagerTest(unittest.TestCase):
    def test_groupvit_is_registered(self):
        config_class = NormalizedConfigManager.get_normalized_config_class("groupvit")
        self.assertIs(config_class, Pix2StructNormalizedTextConfig)

    def test_groupvit_normalizes_expected_attributes(self):
        # GroupViT is a dual text/vision encoder: the stats live on the nested
        # text_config/vision_config sub-configs, not on the top-level config,
        # the same shape Pix2StructNormalizedTextConfig was already built for.
        # Unlike Pix2Struct's text_config, GroupViT's only exposes
        # num_hidden_layers, not a num_layers alias -- this is what the
        # NormalizedTextAndVisionConfig.__getattr__ fix below is needed for.
        config = GroupViTConfig()
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class("groupvit")
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.hidden_size, config.text_config.hidden_size)
        self.assertEqual(normalized_config.num_attention_heads, config.text_config.num_attention_heads)
        self.assertEqual(normalized_config.num_layers, config.text_config.num_hidden_layers)
        self.assertEqual(normalized_config.image_size, config.vision_config.image_size)
        self.assertEqual(normalized_config.num_channels, config.vision_config.num_channels)

    def test_pix2struct_num_layers_unaffected_by_the_groupvit_fix(self):
        # Regression guard: NormalizedTextAndVisionConfig.__getattr__ used to prefix
        # the *raw* accessor name onto the sub-config path instead of resolving it
        # through the NUM_LAYERS/etc. mapping first. That happened to still work for
        # Pix2Struct only because its text_config defines both num_layers and
        # num_hidden_layers as synonyms. Pin the value so a future change can't
        # silently regress it while fixing the general case.
        config = Pix2StructConfig()
        normalized_config = Pix2StructNormalizedTextConfig(config)

        self.assertEqual(normalized_config.num_layers, config.text_config.num_layers)
        self.assertEqual(normalized_config.hidden_size, config.text_config.hidden_size)
        self.assertEqual(normalized_config.num_attention_heads, config.text_config.num_attention_heads)
