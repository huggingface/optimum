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
import unittest
from tempfile import TemporaryDirectory
from typing import Dict

import pytest
from transformers import is_tf_available
from transformers.testing_utils import require_tf

from optimum.utils import DEFAULT_DUMMY_SHAPES
from parameterized import parameterized

from ..exporters_utils import PYTORCH_EXPORT_MODELS_TINY


if is_tf_available():
    from optimum.exporters.tasks import TasksManager

import subprocess


def _get_models_to_test(export_models_dict: Dict):
    models_to_test = []
    if is_tf_available():
        for model_type, model_names_tasks in export_models_dict.items():
            model_type = model_type.replace("_", "-")
            try:
                task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "tflite")
            except KeyError:
                # In this case the model is either not supported, or the contributor forgot to register the
                # TFLiteConfig in the TasksManager.
                # We check that supported model was left unregistered for a backend in the TasksManager unit tests, so
                # we can simply skip in this case here.
                continue

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
                    tflite_config_constructor = TasksManager.get_exporter_config_constructor(
                        model_type=model_type,
                        exporter="tflite",
                        task=task,
                        model_name=model_name,
                        exporter_config_kwargs=DEFAULT_DUMMY_SHAPES,
                    )

                    mandatory_axes = tflite_config_constructor.func.get_mandatory_axes_for_task(task)
                    shapes = " ".join(f"--{name}={DEFAULT_DUMMY_SHAPES[name]}" for name in mandatory_axes)

                    models_to_test.append((f"{model_type}_{task}", model_name, task, shapes))

        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy")]


class TFLiteCLIExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _tflite_export(self, test_name: str, model_name: str, task: str, shapes: str):
        with TemporaryDirectory() as tmpdir:
            if task is not None:
                subprocess.run(
                    f"python3 -m optimum.exporters.tflite --model {model_name} --task {task} {shapes} {tmpdir}",
                    shell=True,
                    check=True,
                )
            else:
                subprocess.run(
                    f"python3 -m optimum.exporters.tflite --model {model_name} {shapes} {tmpdir}",
                    shell=True,
                    check=True,
                )

    @pytest.mark.skip("Not supported yet, need to have proper list of models to export to do it")
    def test_all_models_tested(self):
        pass
        # TODO: enable later.
        # make sure we test all models
        # missing_models_set = TasksManager._SUPPORTED_CLI_MODEL_TYPE - set(PYTORCH_EXPORT_MODELS_TINY.keys())
        # if len(missing_models_set) > 0:
        #     self.fail(f"Not testing all models. Missing models: {missing_models_set}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_tf
    def test_exporters_cli_tflite(self, test_name: str, model_name: str, task: str, shapes: str):
        self._tflite_export(test_name, model_name, task, shapes)

    @pytest.mark.skip("Not supported yet since we only support the export for BERT")
    def test_trust_remote_code(self):
        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.tflite --model fxmarty/tiny-testing-gpt2-remote-code --task causal-lm {tmpdirname}",
                shell=True,
                capture_output=True,
            )
            self.assertTrue(out.returncode, 1)
            self.assertTrue(f"requires you to execute the modeling file in that repo" in out.stderr.decode("utf-8"))

        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.tflite --trust-remote-code --model fxmarty/tiny-testing-gpt2-remote-code --task causal-lm {tmpdirname}",
                shell=True,
                check=True,
            )
