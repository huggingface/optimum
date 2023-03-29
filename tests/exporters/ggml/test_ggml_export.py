# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import unittest
from tempfile import TemporaryDirectory
from typing import Dict

from parameterized import parameterized
from transformers.testing_utils import require_torch

from optimum.exporters.ggml import main_export
from optimum.exporters.tasks import TasksManager

from ..exporters_utils import PYTORCH_EXPORT_MODELS_TINY


class GgmlExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _get_models_to_test(export_models_dict: Dict):
        models_to_test = []
        for model_type, model_names_tasks in export_models_dict.items():
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "ggml")

            if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
                tasks = list(task_config_mapping.keys())
                model_tasks = {model_names_tasks: tasks}
            else:
                unique_tasks = set()
                for tasks in model_names_tasks.values():
                    for task in tasks:
                        unique_tasks.add(task)
                n_tested_tasks = len(unique_tasks)
                if n_tested_tasks < len(task_config_mapping):
                    raise ValueError(f"Not all tasks are tested for {model_type}.")
                model_tasks = model_names_tasks  # possibly, test different tasks on different models

            for model_name, tasks in model_tasks.items():
                for task in tasks:
                    if task in task_config_mapping:
                        models_to_test.append((f"{model_type}_{task}_fp32", model_type, model_name, task, False))
                        models_to_test.append((f"{model_type}_{task}_fp16", model_type, model_name, task, True))

        return sorted(models_to_test)

    def _ggml_export(
        self,
        model_name: str,
        task: str,
        fp16: bool = False,
    ):
        with TemporaryDirectory() as tmpdir:
            main_export(
                model_name_or_path=model_name,
                output=tmpdir,
                task=task,
                fp16=fp16,
            )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    def test_ggml_export(self, test_name: str, model_type: str, model_name: str, task: str, fp16: bool):
        self._ggml_export(model_name, task, fp16=fp16)
