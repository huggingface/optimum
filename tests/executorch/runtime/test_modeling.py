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

import os
import tempfile
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
    def test_load_model_from_hub(self):
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path="NousResearch/Llama-3.2-1B",
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

    @slow
    @pytest.mark.run_slow
    def test_load_model_from_local_path(self):
        from optimum.exporters.executorch import main_export

        model_id = "NousResearch/Llama-3.2-1B"
        task = "text-generation"
        recipe = "xnnpack"

        with tempfile.TemporaryDirectory() as tempdir:
            # Export to a local dir
            main_export(
                model_name_or_path=model_id,
                task=task,
                recipe=recipe,
                output_dir=tempdir,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/model.pte"))

            # Load the exported model from a local dir
            model = ExecuTorchModelForCausalLM.from_pretrained(
                model_name_or_path=tempdir,
                export=False,
            )
            self.assertIsInstance(model, ExecuTorchModelForCausalLM)
            self.assertIsInstance(model.model, ExecuTorchModule)

    @slow
    @pytest.mark.run_slow
    def test_llama3_2_1b_text_generation_with_xnnpack(self):
        model_id = "NousResearch/Llama-3.2-1B"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path=model_id,
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        EXPECTED_GENERATED_TEXT = "Simply put, the theory of relativity states that the laws of physics are the same in all inertial frames of reference."
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Simply put, the theory of relativity states that",
            max_seq_len=len(tokenizer.encode(EXPECTED_GENERATED_TEXT)),
        )
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)

    @slow
    @pytest.mark.run_slow
    def test_llama3_2_3b_text_generation_with_xnnpack(self):
        model_id = "NousResearch/Hermes-3-Llama-3.2-3B"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path=model_id,
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        EXPECTED_GENERATED_TEXT = (
            "Simply put, the theory of relativity states that time is relative and can be affected "
            "by an object's speed. This theory was developed by Albert Einstein in the early 20th "
            "century. The theory has two parts"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Simply put, the theory of relativity states that",
            max_seq_len=len(tokenizer.encode(EXPECTED_GENERATED_TEXT)),
        )
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)

    @slow
    @pytest.mark.run_slow
    def test_qwen2_5_text_generation_with_xnnpack(self):
        model_id = "Qwen/Qwen2.5-0.5B"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path=model_id,
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        EXPECTED_GENERATED_TEXT = "My favourite condiment is iced tea. I love it with my breakfast, my lunch"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="My favourite condiment is ",
            max_seq_len=len(tokenizer.encode(EXPECTED_GENERATED_TEXT)),
        )
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)

    @slow
    @pytest.mark.run_slow
    def test_gemma2_text_generation_with_xnnpack(self):
        # model_id = "google/gemma-2-2b"
        model_id = "unsloth/gemma-2-2b-it"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path=model_id,
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        EXPECTED_GENERATED_TEXT = (
            "Hello I am doing a project for my school and I need to make sure it is a great to be creative and I can!"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Hello I am doing a project for my school",
            max_seq_len=len(tokenizer.encode(EXPECTED_GENERATED_TEXT)),
        )
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)

    @slow
    @pytest.mark.run_slow
    def test_gemma_text_generation_with_xnnpack(self):
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

    @slow
    @pytest.mark.run_slow
    def test_olmo_text_generation_with_xnnpack(self):
        model_id = "allenai/OLMo-1B-hf"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path=model_id,
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        EXPECTED_GENERATED_TEXT = (
            "Simply put, the theory of relativity states that the speed of light is the same in all directions."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Simply put, the theory of relativity states that",
            max_seq_len=len(tokenizer.encode(EXPECTED_GENERATED_TEXT)),
        )
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)
