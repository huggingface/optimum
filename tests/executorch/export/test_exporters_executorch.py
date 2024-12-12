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
import subprocess
import tempfile
import unittest

import pytest
from transformers.testing_utils import slow


class TestExportToExecuTorchCLI(unittest.TestCase):
    def test_helps_no_raise(self):
        subprocess.run(
            "optimum-cli export executorch --help",
            shell=True,
            check=True,
        )

    @slow
    @pytest.mark.run_slow
    def test_llama3_2_1b_export_to_executorch(self):
        model_id = "NousResearch/Llama-3.2-1B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    @slow
    @pytest.mark.run_slow
    def test_llama3_2_3b_export_to_executorch(self):
        model_id = "NousResearch/Hermes-3-Llama-3.2-3B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    @slow
    @pytest.mark.run_slow
    def test_qwen2_5_export_to_executorch(self):
        model_id = "Qwen/Qwen2.5-0.5B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    @slow
    @pytest.mark.run_slow
    def test_gemma2_export_to_executorch(self):
        model_id = "unsloth/gemma-2-2b-it"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    @slow
    @pytest.mark.run_slow
    def test_gemma_export_to_executorch(self):
        model_id = "weqweasdas/RM-Gemma-2B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    @slow
    @pytest.mark.run_slow
    def test_olmo_export_to_executorch(self):
        model_id = "allenai/OLMo-1B-hf"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))
