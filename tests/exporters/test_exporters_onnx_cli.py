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
import os
import unittest
from tempfile import TemporaryDirectory
from typing import Dict, Optional

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_vision

from optimum.onnxruntime import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME
from parameterized import parameterized


if is_torch_available():
    from optimum.exporters.tasks import TasksManager

import subprocess

from exporters_utils import PYTORCH_EXPORT_MODELS_TINY


def _get_models_to_test(export_models_dict: Dict):
    models_to_test = []
    if is_torch_available():
        for model_type, model_names_tasks in export_models_dict.items():
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")

            if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
                tasks = list(task_config_mapping.keys())
                model_tasks = {model_names_tasks: tasks}
            else:
                n_tested_tasks = sum(len(tasks) for tasks in model_names_tasks.values())
                if n_tested_tasks != len(task_config_mapping):
                    raise ValueError(f"Not all tasks are tested for {model_type}.")
                model_tasks = model_names_tasks  # possibly, test different tasks on different models

            for model_name, tasks in model_tasks.items():
                for task in tasks:
                    models_to_test.append((f"{model_type}_{task}", model_name, task, False))

                    if any(
                        task.startswith(ort_special_task)
                        for ort_special_task in ["causal-lm", "seq2seq-lm", "speech2seq-lm"]
                    ):
                        models_to_test.append((f"{model_type}_{task}_forort", model_name, task, True))

            # TODO: segformer task can not be automatically inferred
            # TODO: xlm-roberta model auto-infers causal-lm, but we don't support it
            # TODO: perceiver auto-infers default, but we don't support it (why?)
            if model_type not in ["segformer", "xlm-roberta", "perceiver"]:
                models_to_test.append((f"{model_type}_no_task", model_name, None, False))

        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy")]


class OnnxCLIExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _onnx_export(self, test_name: str, model_name: str, task: Optional[str], for_ort: bool = False):

        with TemporaryDirectory() as tmpdir:
            for_ort = " --for-ort " if for_ort is True else " "
            if task is not None:
                subprocess.run(
                    f"python3 -m optimum.exporters.onnx --model {model_name}{for_ort}--task {task} {tmpdir}",
                    shell=True,
                    check=True,
                )
            else:
                subprocess.run(
                    f"python3 -m optimum.exporters.onnx --model {model_name}{for_ort}{tmpdir}",
                    shell=True,
                    check=True,
                )

    def test_all_models_tested(self):
        # make sure we test all models
        missing_models_set = TasksManager._SUPPORTED_CLI_MODEL_TYPE - set(PYTORCH_EXPORT_MODELS_TINY.keys())
        if len(missing_models_set) > 0:
            self.fail(f"Not testing all models. Missing models: {missing_models_set}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    def test_exporters_cli_pytorch(self, test_name: str, model_name: str, task: str, for_ort: bool):
        self._onnx_export(test_name, model_name, task, for_ort)

    @parameterized.expand([(False,), (True,)])
    def test_external_data(self, use_cache: bool):
        os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data

        with TemporaryDirectory() as tmpdirname:

            task = "seq2seq-lm"
            if use_cache:
                task += "-with-past"

            subprocess.run(
                f"python3 -m optimum.exporters.onnx --model hf-internal-testing/tiny-random-t5 --task {task} --for-ort {tmpdirname}",
                shell=True,
                check=True,
            )

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_ENCODER_NAME + "_data" in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME + "_data" in folder_contents)

            if use_cache:
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME in folder_contents)
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME + "_data" in folder_contents)

        os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    def test_trust_remote_code(self):
        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.onnx --model fxmarty/tiny-testing-gpt2-remote-code --task causal-lm --for-ort {tmpdirname}",
                shell=True,
                capture_output=True,
            )
            self.assertTrue(out.returncode, 1)
            self.assertTrue(f"requires you to execute the modeling file in that repo" in out.stderr.decode("utf-8"))

        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.onnx --trust-remote-code --model fxmarty/tiny-testing-gpt2-remote-code --task causal-lm --for-ort {tmpdirname}",
                shell=True,
                check=True,
            )
