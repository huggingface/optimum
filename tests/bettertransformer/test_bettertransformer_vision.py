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

import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

import requests
from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import grid_parameters
from parameterized import parameterized
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
]

ALL_ZERO_SHOT_IMAGE_CLASSIFICATION = [
    "hf-internal-testing/tiny-random-clip-zero-shot-image-classification",  # with quick_gelu
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # with gelu
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
        inputs = processor(images=image, text=text, return_tensors="pt")
        return inputs


class BetterTransformersCLIPTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision and Text Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    all_models_to_test = ALL_ZERO_SHOT_IMAGE_CLASSIFICATION

    def prepare_inputs_for_class(self, model_id, **preprocessor_kwargs):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        text = ["a photo", "a photo of dog", "a photo of two big cats"]

        # Model takes image and text as input
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor(images=image, text=text, return_tensors="pt", **preprocessor_kwargs)
        return inputs

    def compare_outputs(self, hf_hidden_states, bt_hidden_states, atol: float, model_name: str):
        # CLIP returns a 2D tensor
        self.assert_equal(
            tensor1=hf_hidden_states,
            tensor2=bt_hidden_states,
            atol=atol,
            model_name=model_name,
        )

    # run the test over all possible combinations of `model_id` and `padding`
    @parameterized.expand(
        grid_parameters(
            {
                "model_id": ALL_ZERO_SHOT_IMAGE_CLASSIFICATION,
                "padding": ["max_length", True],
            }
        )
    )
    def test_logits(self, test_name: str, model_id, padding, max_length=20):
        super().test_logits([model_id], padding=padding, max_length=max_length)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": ALL_ZERO_SHOT_IMAGE_CLASSIFICATION,
                "padding": ["max_length", True],
            }
        )
    )
    def test_raise_autocast(self, test_name: str, model_id, padding, max_length=20):
        super().test_raise_autocast([model_id], padding=padding, max_length=max_length)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": ALL_ZERO_SHOT_IMAGE_CLASSIFICATION,
                "padding": ["max_length", True],
            }
        )
    )
    def test_raise_train(self, test_name: str, model_id, padding, max_length=20):
        super().test_raise_train([model_id], padding=padding, max_length=max_length)

    @parameterized.expand([(True,), (False,)])
    def test_invert_model_logits(self, keep_original_model=True, **preprocessor_kwargs):
        r"""
        Test that the inverse converted model and hf model have the same logits
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283
        for model in self.all_models_to_test:
            hf_model = AutoModel.from_pretrained(model)
            bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)
            bt_model = BetterTransformer.reverse(bt_model)

            if model == "laion/CLIP-ViT-B-32-laion2B-s34B-b79K":
                inputs = self.prepare_inputs_for_class(model_id=model, padding=True)
            else:
                inputs = self.prepare_inputs_for_class(model_id=model, padding="max_length")

            torch.manual_seed(42)
            output_bt = bt_model(**inputs)

            torch.manual_seed(42)
            output_hf = hf_model(**inputs)

            self.assertTrue(torch.allclose(output_bt[0], output_hf[0], atol=1e-3))
