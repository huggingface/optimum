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
from tempfile import TemporaryDirectory
from typing import Dict, Optional
from unittest import TestCase

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_vision

from parameterized import parameterized


if is_torch_available():
    from optimum.exporters.tasks import TasksManager

import subprocess

from exporters_utils import PYTORCH_EXPORT_MODELS_TINY


def _get_models_to_test(export_models_dict: Dict):
    models_to_test = []
    if is_torch_available():
        for model_type, model_name in export_models_dict.items():
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")

            for task in task_config_mapping.keys():
                models_to_test.append((f"{model_type}_{task}", model_name, task, False))

                if any(
                    task.startswith(ort_special_task)
                    for ort_special_task in ["causal-lm", "seq2seq-lm", "speech2seq-lm"]
                ):
                    models_to_test.append((f"{model_type}_{task}_forort", model_name, task, True))

            models_to_test.append((f"{model_type}_no_task", model_name, None, False))

        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy")]


class OnnxExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _onnx_export(self, test_name: str, model_name: str, task: Optional[str], for_ort: bool = False):

        with TemporaryDirectory() as tmpdir:
            for_ort = " --for-ort " if for_ort is True else " "
            try:
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
            except Exception as e:
                self.fail(f"{test_name} raised: {e}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    def test_exporters_cli_pytorch(self, test_name: str, model_name: str, task: str, for_ort: bool):

        # make sure we test all models
        missing_models_set = set(TasksManager._SUPPORTED_MODEL_TYPE.keys()) - set(PYTORCH_EXPORT_MODELS_TINY.keys())
        if len(missing_models_set) > 0:
            self.fail(f"Not testing all models. Missing models: {missing_models_set}")

        self._onnx_export(test_name, model_name, task, for_ort)
