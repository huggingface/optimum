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

from transformers import Data2VecVisionConfig

from optimum.utils.normalized_config import NormalizedConfigManager, NormalizedVisionConfig


class NormalizedConfigManagerTest(unittest.TestCase):
    def test_data2vec_vision_is_registered(self):
        config_class = NormalizedConfigManager.get_normalized_config_class("data2vec-vision")
        self.assertIs(config_class, NormalizedVisionConfig)

    def test_data2vec_vision_normalizes_expected_attributes(self):
        config = Data2VecVisionConfig()
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class("data2vec-vision")
        normalized_config = normalized_config_class(config)

        self.assertEqual(normalized_config.image_size, config.image_size)
        self.assertEqual(normalized_config.num_channels, config.num_channels)
