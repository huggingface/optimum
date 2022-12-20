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

import subprocess
import tempfile
import unittest


class TestCLI(unittest.TestCase):
    def test_helps_no_raise(self):
        commands = [
            "optimum-cli --help",
            "optimum-cli export --help",
            "optimum-cli export onnx --help",
            "optimum-cli env --help",
        ]

        for command in commands:
            subprocess.run(command, shell=True, check=True)

    def test_basic_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            commands = [
                "optimum-cli env",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-vision_perceiver_conv --task image-classification {tempdir}",
            ]

            for command in commands:
                subprocess.run(command, shell=True, check=True)
