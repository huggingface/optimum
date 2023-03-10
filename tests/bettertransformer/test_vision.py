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

import requests
from parameterized import parameterized
from PIL import Image
from testing_utils import MODELS_DICT, BetterTransformersTestMixin
from transformers import AutoFeatureExtractor, AutoProcessor

from optimum.utils.testing_utils import grid_parameters, require_torch_20


class BetterTransformersVisionTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    SUPPORTED_ARCH = ["blip", "clip", "clip_text_model", "deit", "vilt", "vit", "vit_mae", "vit_msn", "yolos"]

    def prepare_inputs_for_class(self, model_id, model_type, batch_size=3, **preprocessor_kwargs):
        if model_type == "vilt":
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            text = "How many cats are there?"

            # Model takes image and text as input
            processor = AutoProcessor.from_pretrained(model_id)
            inputs = processor(images=image, text=text, return_tensors="pt")
        elif model_type in ["clip", "clip_text_model"]:
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)

            if batch_size == 1:
                text = ["a photo"]
            else:
                text = ["a photo"] + ["a photo of two big cats"] * (batch_size - 1)

            padding = preprocessor_kwargs.pop("padding", True)
            # Model takes image and text as input
            processor = AutoProcessor.from_pretrained(model_id)
            inputs = processor(images=image, text=text, padding=padding, return_tensors="pt", **preprocessor_kwargs)
        else:
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)

            # Use the same feature extractor for everyone
            feature_extractor = AutoFeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-ViTModel")
            inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

    @parameterized.expand(SUPPORTED_ARCH)
    def test_logits(self, model_type: str):
        if model_type not in self.SUPPORTED_ARCH:
            self.skipTest("useless")
        model_id = MODELS_DICT[model_type]
        self._test_logits(model_id, model_type=model_type)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        self._test_raise_autocast(model_id, model_type=model_type)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        self._test_raise_train(model_id, model_type=model_type)

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "keep_original_model": [True, False],
            }
        )
    )
    @require_torch_20
    def test_invert_modules(self, test_name: str, model_type: str, keep_original_model=False):
        model_id = MODELS_DICT[model_type]
        self._test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "keep_original_model": [True, False],
            }
        )
    )
    @require_torch_20
    def test_save_load_invertible(self, test_name: str, model_type: str, keep_original_model=False):
        model_id = MODELS_DICT[model_type]
        self._test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "keep_original_model": [True, False],
            }
        )
    )
    @require_torch_20
    def test_invert_model_logits(self, test_name: str, model_type: str, keep_original_model=False):
        model_id = MODELS_DICT[model_type]
        self._test_invert_model_logits(
            model_id=model_id, model_type=model_type, keep_original_model=keep_original_model
        )

    def compare_outputs(self, model_type, hf_hidden_states, bt_hidden_states, atol: float, model_name: str):
        # CLIP returns a 2D tensor
        if model_type in ["clip_text_model", "clip"]:
            self.assert_equal(
                tensor1=hf_hidden_states,
                tensor2=bt_hidden_states,
                atol=atol,
                model_name=model_name,
            )
