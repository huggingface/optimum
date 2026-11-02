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

import importlib
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import optimum.commands.base


CLI_WITH_CUSTOM_COMMAND_PATH = Path(__file__).parent / "cli_with_custom_command.py"
OPTIMUM_COMMANDS_DIR = Path(inspect.getfile(optimum.commands.base)).parent
REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH = OPTIMUM_COMMANDS_DIR / "register" / "cli_with_custom_command.py"


class TestCLI(unittest.TestCase):
    def test_env_commands(self):
        subprocess.run("optimum-cli env", shell=True, check=True)

    def test_export_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            onnx_export_commands = [
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-vit --task image-classification {tempdir}/vit",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-bert --task text-classification --sequence_length 128 {tempdir}/bert",
            ]
            onnxruntime_commands = [
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/vit --output {tempdir}/onnx-optimized -O1",
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/bert --output {tempdir}/onnx-quantized --avx2",
            ]
            commands = onnx_export_commands + onnxruntime_commands

            for command in commands:
                subprocess.run(command, shell=True, check=True)

    def _run_command_and_check_content(self, command: str, content: str) -> bool:
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        print("stdout:", stdout)
        print("stderr", stderr)
        return content in stdout

    # custom command registration tests
    def test_register_command(self):
        # Nothing was registered, it should fail.
        command_name = "blablabla"
        command_content = "If the CI can read this, it means it worked!"
        succeeded = self._run_command_and_check_content("optimum-cli blablabla", command_content)
        self.assertFalse(succeeded, "The command should fail here since it is not registered yet.")

        # As a "base" command in `optimum-cli`.
        shutil.copy(CLI_WITH_CUSTOM_COMMAND_PATH, REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH)
        # We check that the print_help method prints the registered command.
        succeeded = self._run_command_and_check_content("optimum-cli", command_name)
        self.assertTrue(succeeded, "The command name should appear in the help.")
        succeeded = self._run_command_and_check_content("optimum-cli blablabla", command_content)
        self.assertTrue(succeeded, "The command content should appear here since it is registered.")

        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink()

        # As a subcommand of an existing command, `optimum-cli export` here.
        shutil.copy(CLI_WITH_CUSTOM_COMMAND_PATH, REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH)
        os.environ["TEST_REGISTER_COMMAND_WITH_SUBCOMMAND"] = "true"
        # We check that the print_help method prints the registered command.
        succeeded = self._run_command_and_check_content("optimum-cli export", "blablabla")
        self.assertTrue(succeeded, "The command name should appear in the help.")
        succeeded = self._run_command_and_check_content("optimum-cli export blablabla", command_content)
        self.assertTrue(succeeded, "The command should succeed here since it is registered.")

        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink()

    def tearDown(self):
        super().tearDown()
        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink(missing_ok=True)

    def test_load_namespace_cli_commands_dedup_symlink(self):
        # Regression test for #2417: on systems where `lib64` is a symlink to `lib`,
        # `submodule_search_locations` can return two entries pointing to the same physical
        # directory. Commands from that directory must be registered only once.
        from optimum.commands.optimum_cli import load_optimum_namespace_cli_commands

        register_module_name = "cli_2417_dedup_check"
        register_file_content = (
            "from optimum.commands.base import BaseOptimumCLICommand, CommandInfo\n"
            "\n"
            "\n"
            "class Cli2417DedupCheckCommand(BaseOptimumCLICommand):\n"
            "    COMMAND = CommandInfo(name='cli-2417-dedup-check', help='dedup test')\n"
            "\n"
            "    def run(self):\n"
            "        pass\n"
            "\n"
            "\n"
            "REGISTER_COMMANDS = [Cli2417DedupCheckCommand]\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            real_register_dir = Path(tmp) / "register"
            real_register_dir.mkdir()
            (real_register_dir / f"{register_module_name}.py").write_text(register_file_content)
            symlink_register_dir = Path(tmp) / "register_symlink"
            symlink_register_dir.symlink_to(real_register_dir, target_is_directory=True)

            fake_spec = importlib.machinery.ModuleSpec("optimum.commands.register", loader=None, is_package=True)
            fake_spec.submodule_search_locations = [str(real_register_dir), str(symlink_register_dir)]
            # Provide a namespace parent whose __path__ points at our isolated directory so that
            # `importlib.import_module("optimum.commands.register.cli_2417_dedup_check")` resolves there.
            fake_parent = types.ModuleType("optimum.commands.register")
            fake_parent.__path__ = [str(real_register_dir)]
            fake_parent.__spec__ = fake_spec

            original_find_spec = importlib.util.find_spec
            original_parent = sys.modules.get("optimum.commands.register", None)
            original_submodule = sys.modules.get(f"optimum.commands.register.{register_module_name}", None)

            def patched_find_spec(name, *args, **kwargs):
                if name == "optimum.commands.register":
                    return fake_spec
                return original_find_spec(name, *args, **kwargs)

            try:
                sys.modules["optimum.commands.register"] = fake_parent
                with mock.patch("importlib.util.find_spec", side_effect=patched_find_spec):
                    commands = load_optimum_namespace_cli_commands()
            finally:
                if original_parent is None:
                    sys.modules.pop("optimum.commands.register", None)
                else:
                    sys.modules["optimum.commands.register"] = original_parent
                sys.modules.pop(f"optimum.commands.register.{register_module_name}", None)
                if original_submodule is not None:
                    sys.modules[f"optimum.commands.register.{register_module_name}"] = original_submodule

            command_names = [command.COMMAND.name for command, _ in commands]
            self.assertEqual(
                command_names.count("cli-2417-dedup-check"),
                1,
                f"Expected the command to be registered exactly once, but got: {command_names}",
            )
