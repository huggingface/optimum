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
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict
from unittest import TestCase

from transformers import AutoConfig, is_tf_available
from transformers.testing_utils import require_tf, require_vision, slow

from exporters_utils import PYTORCH_EXPORT_MODELS_TINY
from optimum.exporters.tflite import export, validate_model_outputs
from optimum.utils import DEFAULT_DUMMY_SHAPES
from parameterized import parameterized


if is_tf_available():
    from optimum.exporters.tasks import TasksManager


SEED = 42


class TFLiteConfigTestCase(TestCase):
    """
    Covers the test for models default.

    Default means no specific tasks is being enabled on the model.
    """

    pass


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

                    models_to_test.append(
                        (f"{model_type}_{task}", model_type, model_name, task, tflite_config_constructor)
                    )

        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy", "dummy", lambda x: x)]


class TFLiteExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _tflite_export(
        self, test_name: str, model_type: str, model_name: str, task: str, tflite_config_class_constructor
    ):
        model_class = TasksManager.get_model_class_for_task(task, framework="tf")
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)

        tflite_config = tflite_config_class_constructor(model.config)

        atol = tflite_config.ATOL_FOR_VALIDATION
        if isinstance(atol, dict):
            atol = atol[task.replace("-with-past", "")]

        with NamedTemporaryFile("w") as output:
            try:
                _, tflite_outputs = export(
                    model=model,
                    config=tflite_config,
                    output=Path(output.name),
                )

                validate_model_outputs(
                    config=tflite_config,
                    reference_model=model,
                    tflite_model_path=Path(output.name),
                    tflite_named_outputs=tflite_outputs,
                    atol=atol,
                )
            except (RuntimeError, ValueError) as e:
                self.fail(f"{model_type}, {task} -> {e}")

    # TODO: enable that when it makes sense.
    # def test_all_models_are_tested(self):
    #     # make sure we test all models
    #     missing_models_set = TasksManager._SUPPORTED_CLI_MODEL_TYPE - set(PYTORCH_EXPORT_MODELS_TINY.keys())
    #     if len(missing_models_set) > 0:
    #         self.fail(f"Not testing all models. Missing models: {missing_models_set}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @slow
    @require_tf
    @require_vision
    def test_tensorflow_export(self, test_name, name, model_name, task, tflite_config_class_constructor):
        self._tflite_export(test_name, name, model_name, task, tflite_config_class_constructor)
