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


class TestExportToExecuTorchCLI(unittest.TestCase):
    def test_helps_no_raise(self):
        subprocess.run(
            "optimum-cli export executorch --help",
            shell=True,
            check=True,
        )

    def test_export_to_executorch_command(self):
        model_id = "meta-llama/Llama-3.2-1B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))
