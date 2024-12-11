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

from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import (
    AutoTokenizer,
)

from optimum.executorchruntime import ExecuTorchModelForCausalLM


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_MODEL_ID = "meta-llama/Llama-3.2-1B"

    def test_text_generation_with_xnnpack(self):
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path=self.TEST_MODEL_ID,
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        EXPECTED_GENERATED_TEXT = "Simply put, the theory of relativity states that the laws of physics are the same in all inertial frames of reference."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Simply put, the theory of relativity states that",
            max_seq_len=len(tokenizer.encode(EXPECTED_GENERATED_TEXT)),
        )
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)
