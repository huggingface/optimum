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

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

from optimum.bettertransformer import BetterTransformer
from parameterized import parameterized
from testing_bettertransformer_utils import BetterTransformersTestMixin


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

    @parameterized.expand([(False,)])
    def test_invert_modules(self, keep_original_model=False):
        r"""
        Test that the inverse converted model and hf model have the same modules
        Since `Wav2vec2` does not support `deepcopy` we cannot test the transformation
        with `keep_original_model=True` for this model.
        """
        for model in self.all_models_to_test:
            # get hf and bt model
            hf_model = AutoModel.from_pretrained(model)
            # get bt model and invert it
            bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)
            bt_model = BetterTransformer.reverse(bt_model)

            # get modules:
            hf_modules = list(hf_model.modules())
            bt_modules = list(bt_model.modules())

            # Assert that the modules are the same
            self.assertEqual(len(hf_modules), len(bt_modules))
            for hf_module, bt_module in zip(hf_modules, bt_modules):
                self.assertEqual(type(hf_module), type(bt_module))
                # check the modules have the same methods
                self.assertEqual(dir(hf_module), dir(bt_module))

                # check the modules have the same attributes
                hf_module_attributes = [attr for attr in dir(hf_module) if not attr.startswith("_")]
                bt_module_attributes = [attr for attr in dir(bt_module) if not attr.startswith("_")]

                self.assertEqual(hf_module_attributes, bt_module_attributes)

    @parameterized.expand([(False,)])
    def test_save_load_invertible(self, keep_original_model=False):
        r"""
        Since `Wav2vec2` does not support `deepcopy` we cannot test the transformation
        with `keep_original_model=True` for this model.
        """
        for model in self.all_models_to_test:
            with tempfile.TemporaryDirectory() as tmpdirname:
                hf_model = AutoModel.from_pretrained(model).eval()
                bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)

                bt_model = BetterTransformer.reverse(bt_model)
                # check if no parameter is on the `meta` device

                for name, param in bt_model.named_parameters():
                    self.assertFalse(param.device.type == "meta", f"Parameter {name} is on the meta device.")

                bt_model.save_pretrained(tmpdirname)

                bt_model_from_load = AutoModel.from_pretrained(tmpdirname)

                # check if the state dict is the same
                # first check if the keys are the same
                self.assertEqual(
                    set(bt_model.state_dict().keys()),
                    set(bt_model_from_load.state_dict().keys()),
                )
                # check also with HF model
                self.assertEqual(
                    set(hf_model.state_dict().keys()),
                    set(bt_model_from_load.state_dict().keys()),
                )

                for key in bt_model.state_dict().keys():
                    self.assertTrue(
                        torch.allclose(
                            bt_model.state_dict()[key],
                            bt_model_from_load.state_dict()[key],
                        )
                    )

                    self.assertTrue(
                        torch.allclose(
                            hf_model.state_dict()[key],
                            bt_model_from_load.state_dict()[key],
                        )
                    )

    @parameterized.expand([(False,)])
    def test_invert_model_logits(self, keep_original_model=False):
        r"""
        Test that the inverse converted model and hf model have the same logits.
        Since `Wav2vec2` does not support `deepcopy` we cannot test the transformation
        with `keep_original_model=True` for this model.
        """
        for model in self.all_models_to_test:
            # get hf and bt model
            hf_model = AutoModel.from_pretrained(model)
            # get bt model and invert it
            bt_model = BetterTransformer.transform(hf_model, keep_original_model=keep_original_model)
            bt_model = BetterTransformer.reverse(bt_model)

            # get inputs
            inputs = self.prepare_inputs_for_class(model)

            # get outputs
            torch.manual_seed(42)
            output_bt = bt_model(**inputs)

            torch.manual_seed(42)
            output_hf = hf_model(**inputs)

            # Assert that the outputs are the same
            self.assertTrue(torch.allclose(output_bt[0], output_hf[0], atol=1e-3))
