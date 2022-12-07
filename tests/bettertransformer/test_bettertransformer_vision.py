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

from PIL import Image
from transformers import AutoFeatureExtractor, AutoProcessor

import requests
from testing_bettertransformer_utils import BetterTransformersTestMixin


ALL_VISION_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-ViTModel",
    "hf-internal-testing/tiny-random-YolosModel",
    "hf-internal-testing/tiny-random-ViTMAEModel",
    "hf-internal-testing/tiny-random-ViTMSNModel",
    "hf-internal-testing/tiny-random-deit",
]


ALL_VISION_TEXT_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-vilt-random-vqa",
    "ybelkada/tiny-random-flava",
]


class BetterTransformersVisionTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    all_models_to_test = ALL_VISION_MODELS_TO_TEST

    def prepare_inputs_for_class(self, model_id=None):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # Use the same feature extractor for everyone
        feature_extractor = AutoFeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-ViTModel")
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs


class BetterTransformersViLTTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision and Text Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    all_models_to_test = ALL_VISION_TEXT_MODELS_TO_TEST

    def prepare_inputs_for_class(self, model_id=None):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        text = "How many cats are there?"

        # Model takes image and text as input
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor(image, text, return_tensors="pt")
        return inputs


class BetterTransformersFlavaTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision and Text Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    all_models_to_test = ["ybelkada/tiny-random-flava"]

    def prepare_inputs_for_class(self, model_id=None):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        text = "How many cats are there?"

        # Model takes image and text as input
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor(image, text, return_tensors="pt")
        return inputs


     def  test_raise_activation_fun(self):
        r"""
        A tests that checks if the conversion raises an error if the model contains an activation function
        that is not supported by `BetterTransformer`. Here we need to loop over the config files
        """
        for hf_random_config in self.all_models_to_test:
            hf_random_config.vision_config.hidden_act = "silu"

            hf_random_model = AutoModel.from_config(hf_random_config).eval()
            with self.assertRaises(ValueError):
                _ = BetterTransformer.transform(hf_random_model, keep_original_model=True)      
