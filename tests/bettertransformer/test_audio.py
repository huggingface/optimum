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
import pytest
import torch
from parameterized import parameterized
from testing_utils import MODELS_DICT, BetterTransformersTestMixin
from transformers import AutoModel, AutoProcessor, set_seed

from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import grid_parameters, require_torch_gpu


ALL_AUDIO_MODELS_TO_TEST = [
    "patrickvonplaten/wav2vec2_tiny_random",
    "ybelkada/hubert-tiny-random",
    "ybelkada/tiny-wav2vec2-stable-ln",
    "ylacombe/bark-small",
]


class TestsWhisper(unittest.TestCase):
    def test_error_message(self):
        model = AutoModel.from_pretrained("openai/whisper-tiny")

        with self.assertRaises(ValueError) as cm:
            model = BetterTransformer.transform(model)

        self.assertTrue("Transformers now supports natively BetterTransformer optimizations" in str(cm.exception))


class BetterTransformersBarkTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Bark - tests all the tests defined in `BetterTransformersTestMixin`
    Since `Bark` is a text-to-speech model, it is preferrable
    to define its own testing class.
    """
    SUPPORTED_ARCH = ["bark"]

    FULL_GRID = {
        "model_type": SUPPORTED_ARCH,
        "keep_original_model": [False],
    }

    def prepare_inputs_for_class(self, model_id, model_type, batch_size=1, **kwargs):
        if batch_size == 1:
            texts = ["a dummy input yeah!"]
        else:
            texts = ["a dummy input yeah!"] + ["and two"] * (batch_size - 1)

        processor = AutoProcessor.from_pretrained(model_id)

        input_dict = processor(texts, **kwargs)

        return input_dict

    @require_torch_gpu
    def _test_fp16_inference(
        self, model_id: str, model_type: str, automodel_class, use_to_operator=False, **preprocessor_kwargs
    ):
        r"""
        This tests if the converted model runs fine under fp16.
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283
        inputs = self.prepare_inputs_for_class(model_id=model_id, model_type=model_type, **preprocessor_kwargs).to(0)

        set_seed(0)

        if not use_to_operator:
            hf_random_model = automodel_class.from_pretrained(model_id, torch_dtype=torch.float16).to(0)
            converted_model = BetterTransformer.transform(hf_random_model, keep_original_model=False)

            hf_random_model = automodel_class.from_pretrained(model_id, torch_dtype=torch.float16).to(0)
        else:
            hf_random_model = automodel_class.from_pretrained(model_id).to(0)
            converted_model = BetterTransformer.transform(hf_random_model, keep_original_model=False)

            hf_random_model = automodel_class.from_pretrained(model_id).to(0)
            hf_random_model = hf_random_model.to(torch.float16)
            converted_model = converted_model.to(torch.float16)

        self.assertFalse(
            hasattr(hf_random_model, "use_bettertransformer"),
            f"The model {hf_random_model.__class__.__name__} has been converted to a `fast` model by mistake.",
        )

        length = 50
        rtol = 5e-2

        with torch.inference_mode():
            r"""
            Make sure the models are in eval mode! Make also sure that the original model
            has not been converted to a fast model. The check is done above.
            """
            output_hf = hf_random_model.generate(
                **inputs, fine_temperature=None, do_sample=False, semantic_max_new_tokens=length
            )

            output_bt = converted_model.generate(
                **inputs, fine_temperature=None, do_sample=False, semantic_max_new_tokens=length
            )

            self.assertTrue(
                (output_hf - output_bt).abs().mean() < rtol,
                f"Mean absolute diff: {(output_hf - output_bt).abs().mean()}",
            )

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "use_to_operator": [True, False],
                "batch_size": [1, 2],
            }
        )
    )
    @pytest.mark.fp16
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_fp16_inference(self, test_name: str, model_type: str, use_to_operator: bool, batch_size: int):
        model_id = MODELS_DICT[model_type]
        self._test_fp16_inference(
            model_id,
            model_type=model_type,
            use_to_operator=use_to_operator,
            automodel_class=AutoModel,
            batch_size=batch_size,
        )

    @parameterized.expand(grid_parameters({"model_type": SUPPORTED_ARCH, "batch_size": [1, 2]}))
    def test_generation(self, test_name: str, model_type: str, batch_size: int):
        model_id = MODELS_DICT[model_type]
        processor = AutoProcessor.from_pretrained(model_id)

        model = AutoModel.from_pretrained(model_id)

        text = ["This is me and me"]
        if batch_size > 1:
            text.append("Please continue this my dear me")
        inp = processor(text, return_tensors="pt")

        length = 50

        result_vanilla = model.generate(
            **inp, num_beams=1, fine_temperature=None, do_sample=False, semantic_max_new_tokens=length
        )

        model = BetterTransformer.transform(model)

        result_bettertransformer = model.generate(
            **inp, num_beams=1, fine_temperature=None, do_sample=False, semantic_max_new_tokens=length
        )

        self.assertTrue(
            torch.allclose(result_vanilla, result_bettertransformer),
            f" Maxdiff: {(result_vanilla - result_bettertransformer).abs().max()}",
        )

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_invert_modules(self, test_name: str, model_type: str, keep_original_model=False):
        model_id = MODELS_DICT[model_type]
        self._test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_save_load_invertible(self, test_name: str, model_type: str, keep_original_model=False):
        model_id = MODELS_DICT[model_type]
        self._test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)


class BetterTransformersAudioTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Testing suite for Audio models - tests all the tests defined in `BetterTransformersTestMixin`
    """
    SUPPORTED_ARCH = ["wav2vec2", "hubert"]

    FULL_GRID = {
        "model_type": SUPPORTED_ARCH,
        "keep_original_model": [True, False],
    }

    def prepare_inputs_for_class(self, model_id, model_type):
        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16_000 * s) for s in batch_duration_in_seconds]

        feature_extractor = AutoProcessor.from_pretrained(model_id, return_attention_mask=True)

        input_dict = feature_extractor(input_features, return_tensors="pt", padding=True)
        return input_dict

    @parameterized.expand(SUPPORTED_ARCH)
    def test_logits(self, model_type: str):
        r"""
        This tests if the converted model produces the same logits
        than the original model.
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283
        model_ids = (
            MODELS_DICT[model_type] if isinstance(MODELS_DICT[model_type], tuple) else (MODELS_DICT[model_type],)
        )
        for model_id in model_ids:
            inputs = self.prepare_inputs_for_class(model_id, model_type)

            torch.manual_seed(0)
            hf_random_model = AutoModel.from_pretrained(model_id).eval()
            random_config = hf_random_model.config

            torch.manual_seed(0)
            converted_model = BetterTransformer.transform(hf_random_model)

            torch.manual_seed(0)
            hf_random_model = AutoModel.from_pretrained(model_id).eval()
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

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_invert_modules(self, test_name: str, model_type: str, keep_original_model=False):
        if model_type in ["hubert", "wav2vec2"] and keep_original_model is True:
            self.skipTest(f"{model_type} does not support keep_original_model=True")

        model_ids = (
            MODELS_DICT[model_type] if isinstance(MODELS_DICT[model_type], tuple) else (MODELS_DICT[model_type],)
        )
        for model_id in model_ids:
            self._test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_save_load_invertible(self, test_name: str, model_type: str, keep_original_model=False):
        if model_type in ["hubert", "wav2vec2"] and keep_original_model is True:
            self.skipTest(f"{model_type} does not support keep_original_model=True")

        model_ids = (
            MODELS_DICT[model_type] if isinstance(MODELS_DICT[model_type], tuple) else (MODELS_DICT[model_type],)
        )
        for model_id in model_ids:
            self._test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_invert_model_logits(self, test_name: str, model_type: str, keep_original_model=False):
        if model_type == "hubert" and keep_original_model is True:
            self.skipTest("hubert does not support keep_original_model=True")

        model_ids = (
            MODELS_DICT[model_type] if isinstance(MODELS_DICT[model_type], tuple) else (MODELS_DICT[model_type],)
        )
        for model_id in model_ids:
            self._test_invert_model_logits(
                model_id=model_id, model_type=model_type, keep_original_model=keep_original_model
            )
