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

from transformers import ConvNextV2Config

from optimum.utils.normalized_config import NormalizedConfigManager, NormalizedVisionConfig


class NormalizedConfigManagerTest(unittest.TestCase):
    def test_convnextv2_is_registered(self):
        config_class = NormalizedConfigManager.get_normalized_config_class("convnextv2")
        self.assertIs(config_class, NormalizedVisionConfig)

    def test_convnextv2_normalizes_expected_attributes(self):
        config = ConvNextV2Config()
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class("convnextv2")
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.image_size, config.image_size)
        self.assertEqual(normalized_config.num_channels, config.num_channels)
