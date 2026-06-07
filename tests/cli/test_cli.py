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

import importlib.util
import inspect
import os
import shutil
import subprocess
import tempfile
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

    def test_symlinked_register_paths_are_deduplicated(self):
        # Regression test: when optimum.commands.register's submodule_search_locations
        # contains a directory together with a symlink that resolves to the same
        # directory (e.g. lib64 -> lib on RHEL/CentOS-derived distros), the register
        # module must be imported only once. Deduplicating on the raw path strings
        # imports it twice and raises "argparse.ArgumentError: conflicting subparser"
        # at CLI startup.
        from optimum.commands import optimum_cli

        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp) / "register"
            real_dir.mkdir()
            (real_dir / "dummy_register.py").write_text("REGISTER_COMMANDS = []\n")

            link_dir = Path(tmp) / "register_symlink"
            try:
                link_dir.symlink_to(real_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are not supported on this platform")

            fake_spec = mock.MagicMock()
            fake_spec.submodule_search_locations = [str(real_dir), str(link_dir)]

            imported = []

            def fake_import(name, *args, **kwargs):
                imported.append(name)
                module = mock.MagicMock()
                module.REGISTER_COMMANDS = []
                return module

            with mock.patch.object(
                optimum_cli.importlib.util, "find_spec", return_value=fake_spec
            ), mock.patch.object(
                optimum_cli.importlib, "import_module", side_effect=fake_import
            ):
                optimum_cli.load_optimum_namespace_cli_commands()

            dummy_imports = [name for name in imported if name.endswith("dummy_register")]
            self.assertEqual(
                len(dummy_imports),
                1,
                f"register module should be imported exactly once, got {imported}",
            )

    def tearDown(self):
        super().tearDown()
        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink(missing_ok=True)
