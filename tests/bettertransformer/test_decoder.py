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

import torch
from parameterized import parameterized
from testing_utils import MODELS_DICT, BetterTransformersTestMixin
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import grid_parameters


class BetterTransformersDecoderTest(BetterTransformersTestMixin, unittest.TestCase):
    SUPPORTED_ARCH = ["codegen", "gpt2", "gptj", "gpt_neo", "gpt_neox"]

    def prepare_inputs_for_class(self, model_id, batch_size=2, **preprocessor_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        padding = preprocessor_kwargs.pop("padding", True)
        if batch_size == 1:
            texts = ["a dummy input yeah!"]
        else:
            texts = ["a dummy input yeah!"] + ["and two"] * (batch_size - 1)
        inputs = tokenizer(texts, return_tensors="pt", padding=padding, **preprocessor_kwargs)
        return inputs

    # run the test over all possible combinations of `model_id` and `padding`
    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "padding": ["max_length", True],
                "batch_size": [1, 2],
            }
        )
    )
    def test_logits(self, test_name: str, model_type: str, padding, batch_size: int):
        model_id = MODELS_DICT[model_type]
        super()._test_logits(model_id, padding=padding, batch_size=batch_size)

    @parameterized.expand(
        grid_parameters({"model_type": SUPPORTED_ARCH, "batch_size": [1, 2], "padding": [True, "max_length"]})
    )
    def test_generation(self, test_name: str, model_type: str, batch_size: int, padding: str):
        model_id = MODELS_DICT[model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(model_id)

        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text = ["This is me and me"]
        if batch_size > 1:
            text.append("Please continue this my dear me")
        inp = tokenizer(text, return_tensors="pt", padding=padding)

        length = 12
        result_vanilla = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        model = BetterTransformer.transform(model)

        print("\n\n\n\n\n GENERATION BT")

        result_bettertransformer = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        print("VANILLA:", result_vanilla)
        print("BT:", result_bettertransformer)

        self.assertTrue(
            torch.allclose(result_vanilla, result_bettertransformer),
            f" Maxdiff: {(result_vanilla - result_bettertransformer).abs().max()}",
        )

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_autocast(model_id)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_train(model_id)
