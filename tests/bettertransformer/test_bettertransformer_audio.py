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

import numpy as np
import torch
from parameterized import parameterized
from testing_bettertransformer_utils import BetterTransformersTestMixin
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import grid_parameters


ALL_AUDIO_MODELS_TO_TEST = [
    "openai/whisper-tiny",
    "patrickvonplaten/wav2vec2_tiny_random",
    "ybelkada/hubert-tiny-random",
]


class BetterTransformersWhisperTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Whisper - tests all the tests defined in `BetterTransformersTestMixin`
    Since `Whisper` uses slightly different inputs than other audio models, it is preferrable
    to define its own testing class.
    """
    all_models_to_test = [ALL_AUDIO_MODELS_TO_TEST[0]]

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def prepare_inputs_for_class(self, model_id):
        input_audio = self._generate_random_audio_data()

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        input_dict = {
            "input_features": feature_extractor(input_audio, return_tensors="pt").input_features,
            "decoder_input_ids": torch.LongTensor([0]),
        }
        return input_dict

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
        r"""
        Test that the inverse converted model and hf model have the same logits.
        Since `Wav2vec2` does not support `deepcopy` we cannot test the transformation
        with `keep_original_model=True` for this model.
        """
        super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
        super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_raise_save_pretrained_error(self, test_name: str, model_id, keep_original_model=False):
        super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
        super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)


class BetterTransformersAudioTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Audio models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    all_models_to_test = ALL_AUDIO_MODELS_TO_TEST[1:]

    def prepare_inputs_for_class(self, model_id):
        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16_000 * s) for s in batch_duration_in_seconds]

        feature_extractor = AutoProcessor.from_pretrained(model_id, return_attention_mask=True)

        input_dict = feature_extractor(input_features, return_tensors="pt", padding=True)
        return input_dict

    def test_logits(self):
        r"""
        This tests if the converted model produces the same logits
        than the original model.
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283

        for model_to_test in self.all_models_to_test:
            inputs = self.prepare_inputs_for_class(model_to_test)

            torch.manual_seed(0)
            hf_random_model = AutoModel.from_pretrained(model_to_test).eval()
            random_config = hf_random_model.config

            torch.manual_seed(0)
            converted_model = BetterTransformer.transform(hf_random_model)

            torch.manual_seed(0)
            hf_random_model = AutoModel.from_pretrained(model_to_test).eval()
            random_config = hf_random_model.config

            self.assertFalse(
                hasattr(hf_random_model, "use_bettertransformer"),
                f"The model {hf_random_model.__class__.__name__} has been converted to a `fast` model by mistake.",
            )

            with torch.no_grad():
                r"""
                Make sure the models are in eval mode! Make also sure that the original model
                has not been converted to a fast model. The check is done above.
                """
                torch.manual_seed(0)
                hf_hidden_states = hf_random_model(**inputs)[0]

                torch.manual_seed(0)
                bt_hidden_states = converted_model(**inputs)[0]

                if "gelu_new" in vars(random_config).values():
                    # Since `gelu_new` is a slightly modified version of `GeLU` we expect a small
                    # discrepency.
                    tol = 4e-2
                else:
                    tol = 1e-3

                self.assertTrue(
                    torch.allclose(bt_hidden_states[0][-3:], torch.zeros_like(bt_hidden_states[0][-3:])),
                    "The BetterTransformers converted model does not give null hidden states on padded tokens",
                )

                self.assertTrue(
                    torch.allclose(hf_hidden_states[-1], bt_hidden_states[-1], atol=tol),
                    "The BetterTransformers Converted model does not produce the same logits as the original model. Failed for the model {}".format(
                        hf_random_model.__class__.__name__
                    ),
                )

    def test_raise_train(self):
        r"""
        A tests that checks if the conversion raises an error if the model is run under
        `model.train()`.
        """
        for model_id in self.all_models_to_test:
            inputs = self.prepare_inputs_for_class(model_id)

            hf_random_model = AutoModel.from_pretrained(model_id).eval()
            # Check for training mode
            with self.assertRaises(ValueError):
                bt_model = BetterTransformer.transform(hf_random_model)
                bt_model.train()
                _ = bt_model(**inputs)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [False],
            }
        )
    )
    def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
        r"""
        Test that the inverse converted model and hf model have the same modules
        Since `Wav2vec2` does not support `deepcopy` we cannot test the transformation
        with `keep_original_model=True` for this model.
        """
        super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [False],
            }
        )
    )
    def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
        r"""
        Since `Wav2vec2` does not support `deepcopy` we cannot test the transformation
        with `keep_original_model=True` for this model.
        """
        super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [False],
            }
        )
    )
    def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
        r"""
        Test that the inverse converted model and hf model have the same logits.
        Since `Wav2vec2` does not support `deepcopy` we cannot test the transformation
        with `keep_original_model=True` for this model.
        """
        super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [False],
            }
        )
    )
    def test_raise_save_pretrained_error(self, test_name: str, model_id, keep_original_model=False):
        r"""
        Test if the converted model raises an error when calling `save_pretrained`
        but not when the model is reverted
        """
        super().test_raise_save_pretrained_error(model_id=model_id, keep_original_model=keep_original_model)
