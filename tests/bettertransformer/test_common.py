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
import tempfile
import unittest
from unittest.mock import patch

import torch
import transformers
from packaging.version import parse
from parameterized import parameterized
from testing_utils import MODELS_DICT
from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer, BetterTransformerManager
from optimum.pipelines import pipeline
from optimum.utils.testing_utils import grid_parameters


class BetterTransformerIntegrationTests(unittest.TestCase):
    def _skip_on_torch_version(self, model_type: str):
        if BetterTransformerManager.requires_torch_20(model_type) and parse(torch.__version__) < parse("1.14"):
            self.skipTest(f"The model type {model_type} require PyTorch 2.0 for BetterTransformer")

    def test_raise_error_on_double_transform_call(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")

        with self.assertRaises(Exception) as cm:
            bt_model = BetterTransformer.transform(model)
            bt_model = BetterTransformer.transform(bt_model)
        self.assertTrue("was called on a model already using Better Transformer" in str(cm.exception))

    @patch("optimum.utils.import_utils.is_onnxruntime_available")
    def test_direct_pipeline_initialization_without_onnx_installed(self, mock_onnxruntime_availability):
        mock_onnxruntime_availability.return_value = False
        pipe = pipeline(
            "question-answering",
            "hf-internal-testing/tiny-random-BertModel",
            accelerator="bettertransformer",
        )
        pipe(
            question=["Is huggingface getting better?", "Will it ever stop getting better?"],
            context=["Huggingface will never stop getting better."] * 2,
            batch_size=10,
        )

    @parameterized.expand(MODELS_DICT.keys())
    def test_raise_on_save(self, model_type: str):
        r"""
        Test if the conversion properly raises an error if someone tries to save the model using `save_pretrained`.
        """
        self._skip_on_torch_version(model_type)
        model_ids = (
            MODELS_DICT[model_type] if isinstance(MODELS_DICT[model_type], tuple) else (MODELS_DICT[model_type],)
        )
        for model_id in model_ids:
            with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdirname:
                hf_model = AutoModel.from_pretrained(model_id).eval()
                bt_model = BetterTransformer.transform(hf_model, keep_original_model=False)
                bt_model.save_pretrained(tmpdirname)

    @parameterized.expand(MODELS_DICT.keys())
    def test_conversion(self, model_type: str):
        r"""
        This tests if the conversion of a slow model to its BetterTransformer version using fastpath
        has been successful.
        """
        self._skip_on_torch_version(model_type)
        model_ids = (
            MODELS_DICT[model_type] if isinstance(MODELS_DICT[model_type], tuple) else (MODELS_DICT[model_type],)
        )
        for model_id in model_ids:
            hf_random_model = AutoModel.from_pretrained(model_id)
            converted_model = BetterTransformer.transform(hf_random_model)

            self.assertTrue(
                hasattr(converted_model, "use_bettertransformer"),
                f"The model {converted_model.__class__.__name__} is not a fast model.",
            )

            self.assertTrue(isinstance(converted_model, hf_random_model.__class__))
            self.assertTrue(hasattr(converted_model, "generate"))

    @parameterized.expand(grid_parameters({"model_type": MODELS_DICT.keys(), "keep_original_model": [True, False]}))
    def test_raise_save_pretrained_error(self, test_name: str, model_type: str, keep_original_model: bool):
        r"""
        Test if the converted model raises an error when calling `save_pretrained`
        but not when the model is reverted
        """
        self._skip_on_torch_version(model_type)
        if model_type in ["wav2vec2", "hubert"] and keep_original_model is True:
            self.skipTest("These architectures do not support deepcopy")

        model_ids = (
            MODELS_DICT[model_type] if isinstance(MODELS_DICT[model_type], tuple) else (MODELS_DICT[model_type],)
        )
        for model_id in model_ids:
            # get hf and bt model
            hf_model = AutoModel.from_pretrained(model_id)
            # get bt model and invert it
            bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)

            with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdirname:
                bt_model.save_pretrained(tmpdirname)

            model = BetterTransformer.reverse(bt_model)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

    @parameterized.expand(BetterTransformerManager.MODEL_MAPPING.keys())
    def test_raise_activation_fun(self, model_type: str):
        r"""
        A tests that checks if the conversion raises an error if the model contains an activation function
        that is not supported by `BetterTransformer`. Here we need to loop over the config files
        """
        self._skip_on_torch_version(model_type)
        if BetterTransformerManager.requires_strict_validation(model_type) is False:
            self.skipTest("The architecture does not require a specific activation function")

        if model_type in ["wav2vec2", "hubert"]:
            self.skipTest("These architectures do not support deepcopy (raise unrelated error)")

        layer_classes = BetterTransformerManager.MODEL_MAPPING[model_type].keys()
        for layer_class in layer_classes:
            if isinstance(layer_class, list):
                layer_class = layer_class[0]

            if layer_class == "EncoderLayer":
                # Hardcode it for FSMT - see https://github.com/huggingface/optimum/pull/494
                class_name = "FSMT"
            elif layer_class == "TransformerBlock":
                # Hardcode it for distilbert - see https://github.com/huggingface/transformers/pull/19966
                class_name = "DistilBert"
            elif "EncoderLayer" in layer_class:
                class_name = layer_class[:-12]
            elif "Attention" in layer_class:
                class_name = layer_class[:-9]
            else:
                class_name = layer_class[:-5]

            hf_random_config = getattr(
                transformers, class_name + "Config"
            )()  # random config class for the model to test
            hf_random_config.hidden_act = "silu"

            hf_random_model = AutoModel.from_config(hf_random_config).eval()
            with self.assertRaises(ValueError) as cm:
                _ = BetterTransformer.transform(hf_random_model, keep_original_model=True)
            self.assertTrue("Activation function" in str(cm.exception))

    def test_dict_class_consistency(self):
        """
        A test to check BetterTransformerManager.MODEL_MAPPING has good names.
        """
        for _, item in BetterTransformerManager.MODEL_MAPPING.items():
            self.assertTrue(
                any(
                    [valid_name in sub_item for sub_item in item.keys()]
                    for valid_name in ["Block", "Layer", "Attention"]
                )
            )
