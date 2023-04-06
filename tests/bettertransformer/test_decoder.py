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
import unittest

import pytest
import torch
from parameterized import parameterized
from testing_utils import MODELS_DICT, BetterTransformersTestMixin
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.bettertransformer import BetterTransformer
from optimum.utils import DummyPastKeyValuesGenerator, NormalizedConfigManager
from optimum.utils.testing_utils import grid_parameters, require_torch_20, require_torch_gpu


class BetterTransformersDecoderTest(BetterTransformersTestMixin, unittest.TestCase):
    SUPPORTED_ARCH = ["codegen", "gpt2", "gptj", "gpt_neo", "gpt_neox", "opt"]

    FULL_GRID = {
        "model_type": SUPPORTED_ARCH,
        "keep_original_model": [True, False],
    }

    def prepare_inputs_for_class(self, model_id, model_type, batch_size=2, **preprocessor_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        padding = preprocessor_kwargs.pop("padding", True)
        if batch_size == 1:
            texts = ["a dummy input yeah!"]
        else:
            texts = ["a dummy input yeah!"] + ["and two"] * (batch_size - 1)
        inputs = tokenizer(texts, return_tensors="pt", padding=padding, max_length=20, **preprocessor_kwargs)
        return inputs

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "padding": ["max_length", True],
                "batch_size": [1, 3],
            }
        )
    )
    def test_logits_without_cache(self, test_name: str, model_type: str, padding, batch_size: int):
        self._skip_on_torch_version(model_type)
        if batch_size == 1 and padding == "max_length":
            self.skipTest("batch_size=1 + padding='max_length' is unsupported")

        model_id = MODELS_DICT[model_type]
        self._test_logits(model_id, model_type=model_type, padding=padding, batch_size=batch_size)

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
        self._skip_on_torch_version(model_type)

        model_id = MODELS_DICT[model_type]
        self._test_fp16_inference(
            model_id,
            model_type=model_type,
            use_to_operator=use_to_operator,
            automodel_class=AutoModelForCausalLM,
            batch_size=batch_size,
        )

    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "batch_size": [1, 3],
            }
        )
    )
    def test_logits_with_cache(self, test_name: str, model_type: str, batch_size: int):
        self._skip_on_torch_version(model_type)
        input_ids = torch.randint(low=1, high=10, size=(batch_size, 1))
        seq_length = 12
        attention_mask = torch.ones(batch_size, seq_length + 1, dtype=torch.int32)

        model_id = MODELS_DICT[model_type]

        model = AutoModelForCausalLM.from_pretrained(model_id)

        normalized_config = NormalizedConfigManager.get_normalized_config_class(model.config.model_type)(model.config)
        pkv_generator = DummyPastKeyValuesGenerator(
            task="", normalized_config=normalized_config, batch_size=batch_size, sequence_length=seq_length
        )
        past_key_values = pkv_generator.generate(input_name="past_key_values")

        result_vanilla = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values)

        model = BetterTransformer.transform(model)

        result_bettertransformer = model(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values
        )

        logits_vanilla = result_vanilla.logits
        logits_bettertransformer = result_bettertransformer.logits
        self.assertTrue(
            torch.allclose(logits_vanilla, logits_bettertransformer, atol=1e-5),
            f" Maxdiff: {(logits_vanilla - logits_bettertransformer).abs().max()}",
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

        model = AutoModelForCausalLM.from_pretrained(model_id)

        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text = ["This is me and me"]
        if batch_size > 1:
            text.append("Please continue this my dear me")
        inp = tokenizer(text, return_tensors="pt", padding=padding, max_length=30)

        length = 50
        result_vanilla = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        model = BetterTransformer.transform(model)

        result_bettertransformer = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        self.assertTrue(
            torch.allclose(result_vanilla, result_bettertransformer),
            f" Maxdiff: {(result_vanilla - result_bettertransformer).abs().max()}",
        )

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_raise_autocast(model_id, model_type=model_type)

    @parameterized.expand(SUPPORTED_ARCH)
    @pytest.mark.training
    def test_train(self, model_type: str):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_train_decoder(model_id, model_type=model_type)

    @parameterized.expand(grid_parameters(FULL_GRID))
    @require_torch_20
    def test_invert_modules(self, test_name: str, model_type: str, keep_original_model=False):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(grid_parameters(FULL_GRID))
    @require_torch_20
    def test_save_load_invertible(self, test_name: str, model_type: str, keep_original_model=False):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        self._test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(grid_parameters(FULL_GRID))
    @require_torch_20
    def test_invert_model_logits(self, test_name: str, model_type: str, keep_original_model=False):
        model_id = MODELS_DICT[model_type]
        self._test_invert_model_logits(
            model_id=model_id, model_type=model_type, keep_original_model=keep_original_model
        )
