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

import inspect
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import optimum.commands


CLI_WIH_CUSTOM_COMMAND_PATH = Path(__file__).parent / "cli_with_custom_command.py"
OPTIMUM_COMMANDS_DIR = Path(inspect.getfile(optimum.commands)).parent
REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH = OPTIMUM_COMMANDS_DIR / "register" / "cli_with_custom_command.py"


class TestCLI(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink(missing_ok=True)

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
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-bart {tempdir}/encoder-decoder",
            ]
            optimize_commands = [
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/encoder -O1",
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/decoder -O1",
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/encoder-decoder -O1",
            ]

            for export, optimize in zip(export_commands, optimize_commands):
                subprocess.run(export, shell=True, check=True)
                subprocess.run(optimize, shell=True, check=True)

    def test_quantize_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # First export a tiny encoder, decoder only and encoder-decoder
            export_commands = [
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-BertModel {tempdir}/encoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-gpt2 {tempdir}/decoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 {tempdir}/encoder-decoder",
            ]
            quantize_commands = [
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/encoder --avx2",
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/decoder --avx2",
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/encoder-decoder --avx2",
            ]

            for export, quantize in zip(export_commands, quantize_commands):
                subprocess.run(export, shell=True, check=True)
                subprocess.run(quantize, shell=True, check=True)

    def _run_command_and_check_content(self, command: str, content: str) -> bool:
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        print(stdout)
        print(stderr)
        return content in stdout

    def test_register_command(self):
        custom_command = "optimum-cli blablabla"
        command_content = "If the CI can read this, it means it worked!"
        succeeded = self._run_command_and_check_content(custom_command, command_content)
        self.assertFalse(succeeded, "The command should fail here since it is not registered yet.")

        shutil.copy(CLI_WIH_CUSTOM_COMMAND_PATH, REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH)
        succeeded = self._run_command_and_check_content(custom_command, command_content)
        self.assertTrue(succeeded, "The command should succeed here since it is registered.")
        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink()

        shutil.copy(CLI_WIH_CUSTOM_COMMAND_PATH, REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH)
        os.environ["TEST_REGISTER_COMMAND_WITH_SUBCOMMAND"] = "true"
        custom_command = "optimum-cli export blablabla"
        succeeded = self._run_command_and_check_content(custom_command, command_content)
        self.assertTrue(succeeded, "The command should succeed here since it is registered.")
        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink()
