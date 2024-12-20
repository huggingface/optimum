# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoTokenizer
from transformers.testing_utils import (
    slow,
)

from optimum.executorchruntime import ExecuTorchModelForCausalLM


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_gemma_text_generation_with_xnnpack(self):
        # TODO: Switch to use google/gemma-2b once https://github.com/huggingface/optimum/issues/2127 is fixed
        # model_id = "google/gemma-2b"
        model_id = "weqweasdas/RM-Gemma-2B"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path=model_id,
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        EXPECTED_GENERATED_TEXT = "Hello I am doing a project for my school and I need to write a report on the history of the United States."
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Hello I am doing a project for my school",
            max_seq_len=len(tokenizer.encode(EXPECTED_GENERATED_TEXT)),
        )
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)
