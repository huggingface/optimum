# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import gc
import unittest

import pytest
import torch
from parameterized import parameterized
from testing_utils import MODELS_DICT, BetterTransformersTestMixin
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import grid_parameters, require_torch_gpu


class BetterTransformersEncoderDecoderTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Full testing suite of the `BetterTransformers` integration into Hugging Face
    `transformers` ecosystem. Check the docstring of each test to understand the
    purpose of each test. Basically we test:
    - if the conversion dictionnary is consistent, ie if the converted model exists
    in HuggingFace `transformers` library.
    - if the converted model produces the same logits as the original model.
    - if the converted model is faster than the original model.
    """
    SUPPORTED_ARCH = [
        "bart",
        "blenderbot",
        "fsmt",
        "m2m_100",
        "marian",
        "mbart",
        "pegasus",
        "t5",
    ]

    FULL_GRID = {
        "model_type": SUPPORTED_ARCH,
        "keep_original_model": [True, False],
    }

    def tearDown(self):
        gc.collect()

    def prepare_inputs_for_class(self, model_id, model_type, **preprocessor_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        padding = preprocessor_kwargs.pop("padding", True)
        inputs = tokenizer(["a dummy input", "and two"], return_tensors="pt", padding=padding, **preprocessor_kwargs)
        inputs["decoder_input_ids"] = inputs["input_ids"]  # just a hack for m2m100
        return inputs

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "padding": ["max_length", True],
            }
        )
    )
    def test_logits_without_cache(self, test_name: str, model_type: str, padding, max_length=20):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_logits(model_id, model_type=model_type, padding=padding, max_length=max_length)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_raise_autocast(model_id, model_type=model_type)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        if model_type not in ["blenderbot", "pegasus", "t5"]:
            self._test_raise_train(model_id, model_type=model_type)
        else:
            self._test_train_decoder(model_id, model_type=model_type)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_invert_modules(self, test_name: str, model_type: str, keep_original_model=False):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_save_load_invertible(self, test_name: str, model_type: str, keep_original_model=False):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_invert_model_logits(self, test_name: str, model_type: str, keep_original_model=False):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_invert_model_logits(
            model_id=model_id, model_type=model_type, keep_original_model=keep_original_model
        )

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "use_to_operator": [True, False],
            }
        )
    )
    @pytest.mark.fp16
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_fp16_inference(self, test_name: str, model_type: str, use_to_operator: bool):
        self._skip_on_torch_version(model_type)

        # TODO: fix in transformers
        if model_type == "fsmt":
            self.skipTest("fsmt is broken is transformers when loaded through torch_dtype=torch.float16")

        model_id = MODELS_DICT[model_type]
        self._test_fp16_inference(
            model_id, model_type=model_type, use_to_operator=use_to_operator, automodel_class=AutoModelForSeq2SeqLM
        )

    @parameterized.expand(
        grid_parameters({"model_type": SUPPORTED_ARCH, "batch_size": [1, 3], "padding": [True, "max_length"]})
    )
    def test_generation(self, test_name: str, model_type: str, batch_size: int, padding: str):
        self._skip_on_torch_version(model_type)
        if batch_size == 1 and padding == "max_length":
            self.skipTest("batch_size=1 + padding='max_length' is unsupported")

        model_id = MODELS_DICT[model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text = ["This is me and me"]
        if batch_size > 1:
            text = text + ["Please"] * (batch_size - 1)
        inp = tokenizer(text, return_tensors="pt", padding=padding, max_length=25)

        length = 50
        result_vanilla = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        model = BetterTransformer.transform(model)

        result_bettertransformer = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        self.assertTrue(
            torch.allclose(result_vanilla, result_bettertransformer),
            f" Maxdiff: {(result_vanilla - result_bettertransformer).abs().max()}",
        )
