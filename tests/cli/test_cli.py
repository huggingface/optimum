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
            "optimum-cli onnxruntime quantize --help",
            "optimum-cli onnxruntime optimize --help",
        ]

        for command in commands:
            subprocess.run(command, shell=True, check=True)

    def test_env_commands(self):
        subprocess.run("optimum-cli env", shell=True, check=True)

    def test_export_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            command = (
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-vision_perceiver_conv --task image-classification {tempdir}",
            )
            subprocess.run(command, shell=True, check=True)

    def test_optimize_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # First export a tiny encoder, decoder only and encoder-decoder
            export_commands = [
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-BertModel {tempdir}/encoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-gpt2 {tempdir}/decoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 {tempdir}/encoder-decoder",
            ]
            optimize_commands = [
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/encoder -O1",
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/decoder -O1",
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/encoder-decoder -O1",
            ]

            for export, optimize in zip(export_commands, optimize_commands):
                subprocess.run(export, shell=True, check=True)
                subprocess.run(optimize, shell=True, check=True)

    def test_optimize_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # First export a tiny encoder, decoder only and encoder-decoder
            export_commands = [
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-BertModel {tempdir}/encoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-gpt2 {tempdir}/decoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 {tempdir}/encoder-decoder",
            ]
            optimize_commands = [
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/encoder --avx2",
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/decoder --avx2",
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/encoder-decoder --avx2",
            ]

            for export, optimize in zip(export_commands, optimize_commands):
                subprocess.run(export, shell=True, check=True)
                subprocess.run(optimize, shell=True, check=True)
