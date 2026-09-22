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

from transformers import Pix2StructConfig, SiglipConfig

from optimum.utils.normalized_config import NormalizedConfigManager, Pix2StructNormalizedTextConfig


class NormalizedConfigManagerTest(unittest.TestCase):
    def test_siglip_is_registered(self):
        config_class = NormalizedConfigManager.get_normalized_config_class("siglip")
        self.assertIs(config_class, Pix2StructNormalizedTextConfig)

    def test_siglip_normalizes_expected_attributes(self):
        # Same dual text/vision encoder shape as GroupViT/OwlViT/OwlV2/Pix2Struct.
        # SigLIP's text_config only has num_hidden_layers, not a num_layers alias,
        # so this also exercises the NormalizedTextAndVisionConfig.__getattr__ fix.
        config = SiglipConfig()
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class("siglip")
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.hidden_size, config.text_config.hidden_size)
        self.assertEqual(normalized_config.num_attention_heads, config.text_config.num_attention_heads)
        self.assertEqual(normalized_config.num_layers, config.text_config.num_hidden_layers)
        self.assertEqual(normalized_config.image_size, config.vision_config.image_size)
        self.assertEqual(normalized_config.num_channels, config.vision_config.num_channels)

    def test_pix2struct_num_layers_unaffected(self):
        config = Pix2StructConfig()
        normalized_config = Pix2StructNormalizedTextConfig(config)

        self.assertEqual(normalized_config.num_layers, config.text_config.num_layers)
        self.assertEqual(normalized_config.hidden_size, config.text_config.hidden_size)
        self.assertEqual(normalized_config.num_attention_heads, config.text_config.num_attention_heads)
