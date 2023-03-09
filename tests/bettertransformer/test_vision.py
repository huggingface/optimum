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

from optimum.utils.testing_utils import grid_parameters


class BetterTransformersVisionTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    SUPPORTED_ARCH = ["deit", "vit", "vit_mae", "vit_msn", "yolos"]

    def prepare_inputs_for_class(self, model_id=None):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # Use the same feature extractor for everyone
        feature_extractor = AutoFeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-ViTModel")
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

    @parameterized.expand(SUPPORTED_ARCH)
    def test_logits(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_logits(model_id)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_autocast(model_id)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_train(model_id)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)


class BetterTransformersViLTTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision and Text Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    SUPPORTED_ARCH = ["vilt"]

    def prepare_inputs_for_class(self, model_id=None):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        text = "How many cats are there?"

        # Model takes image and text as input
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor(images=image, text=text, return_tensors="pt")
        return inputs

    @parameterized.expand(SUPPORTED_ARCH)
    def test_logits(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_logits(model_id)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_autocast(model_id)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_train(model_id)

    # TODO: re-enable once fixed
    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)


class BetterTransformersCLIPTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Vision and Text Models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    SUPPORTED_ARCH = ["blip", "clip", "clip_text_model"]

    def prepare_inputs_for_class(self, model_id, batch_size=3, **preprocessor_kwargs):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        if batch_size == 1:
            text = ["a photo"]
        else:
            text = ["a photo"] + ["a photo of two big cats"] * (batch_size - 1)

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

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "padding": ["max_length", True],
            }
        )
    )
    def test_logits(self, test_name: str, model_type: str, padding, max_length=20):
        model_id = MODELS_DICT[model_type]
        super()._test_logits(model_id, padding=padding, max_length=max_length)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_autocast(model_id, batch_size=1)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_train(model_id, batch_size=1)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    # TODO: re-enable once fixed
    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #             "padding": ["max_length", True],
    #         }
    #     )
    # )
    # def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False, padding=True):
    #     r"""
    #     Test that the inverse converted model and hf model have the same logits
    #     """
    #     # get hf and bt model
    #     hf_model = AutoModel.from_pretrained(model_id)
    #     # get bt model and invert it
    #     bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)
    #     bt_model = BetterTransformer.reverse(bt_model)

    #     # get inputs
    #     inputs = self.prepare_inputs_for_class(model_id, padding=padding)

    #     # get outputs
    #     torch.manual_seed(42)
    #     output_bt = bt_model(**inputs)

    #     # create a new model
    #     hf_model = AutoModel.from_pretrained(model_id)

    #     torch.manual_seed(42)
    #     output_hf = hf_model(**inputs)

    #     # Assert that the outputs are the same
    #     self.assertTrue(torch.allclose(output_bt[0], output_hf[0], atol=1e-3))
