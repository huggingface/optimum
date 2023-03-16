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
from typing import TYPE_CHECKING, Dict, Optional

import pytest
from parameterized import parameterized
from transformers import AutoConfig, is_tf_available
from transformers.testing_utils import require_tf

from optimum.utils import DEFAULT_DUMMY_SHAPES

from ...utils.test_task_processors import TASK_TO_NON_DEFAULT_DATASET
from ..exporters_utils import PYTORCH_EXPORT_MODELS_TINY


if is_tf_available():
    from optimum.exporters.tasks import TasksManager

if TYPE_CHECKING:
    from optimum.exporters.tasks import ExportConfigConstructor


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
                    default_shapes = dict(DEFAULT_DUMMY_SHAPES)
                    if task == "question-answering":
                        default_shapes["sequence_length"] = 384
                    tflite_config_constructor = TasksManager.get_exporter_config_constructor(
                        model_type=model_type,
                        exporter="tflite",
                        task=task,
                        model_name=model_name,
                        exporter_config_kwargs=default_shapes,
                    )

                    models_to_test.append((f"{model_type}_{task}", model_name, task, tflite_config_constructor))

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

    def _tflite_export(
        self,
        model_name: str,
        tflite_config_constructor: "ExportConfigConstructor",
        task: Optional[str] = None,
        quantization: Optional[str] = None,
        fallback_to_float: bool = False,
        inputs_dtype: Optional[str] = None,
        outputs_dtype: Optional[str] = None,
        calibration_dataset_name_or_path: Optional[str] = None,
        calibration_dataset_config_name: Optional[str] = None,
        num_calibration_samples: int = 200,
        calibration_split: Optional[str] = None,
        primary_key: Optional[str] = None,
        secondary_key: Optional[str] = None,
        question_key: Optional[str] = None,
        context_key: Optional[str] = None,
        image_key: Optional[str] = None,
    ):
        with TemporaryDirectory() as tmpdir:
            command = f"python3 -m optimum.exporters.tflite --model {model_name}".split()
            if task is not None:
                command.append(f"--task={task}")
            if quantization is not None:
                command.append(f"--quantize={quantization}")
            if fallback_to_float:
                command.append("--fallback_to_float")
            if inputs_dtype is not None:
                command.append(f"--inputs_type={inputs_dtype}")
            if outputs_dtype is not None:
                command.append(f"--outputs_type={outputs_dtype}")
            if calibration_dataset_name_or_path is not None:
                command.append(f"--calibration_dataset={calibration_dataset_name_or_path}")
            if calibration_dataset_config_name is not None:
                command.append(f"--calibration_dataset_config_name={calibration_dataset_config_name}")
            if calibration_split is not None:
                command.append(f"--calibration_split={calibration_split}")
            if primary_key is not None:
                command.append(f"--primary_key={primary_key}")
            if secondary_key is not None:
                command.append(f"--secondary_key={secondary_key}")
            if question_key is not None:
                command.append(f"--question_key={question_key}")
            if context_key is not None:
                command.append(f"--context_key={context_key}")
            if image_key is not None:
                command.append(f"--image_key={image_key}")
            command.append(f"--num_calibration_samples={num_calibration_samples}")

            tflite_config = tflite_config_constructor(AutoConfig.from_pretrained(model_name))
            mandatory_axes = tflite_config.get_mandatory_axes_for_task(task)
            shapes = [f"--{name}={getattr(tflite_config, name)}" for name in mandatory_axes]
            command += shapes
            command.append(tmpdir)

            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            outs, errs = proc.communicate()
            # Convenient to see the output with -s in PyTest.
            print(outs.decode("utf-8"))
            print(errs.decode("utf-8"))
            not_supported = quantization is not None and (
                not tflite_config.supports_quantization_approach(quantization) and not fallback_to_float
            )
            if not_supported:
                self.assertIn("QuantizationApproachNotSupported", errs.decode("utf-8"))
            else:
                self.assertEqual(proc.returncode, 0)

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
    def test_exporters_cli_tflite(
        self, test_name: str, model_name: str, task: str, tflite_config_constructor: "ExportConfigConstructor"
    ):
        self._tflite_export(model_name, tflite_config_constructor, task=task)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_tf
    @pytest.mark.quantization
    def test_exporters_cli_tflite_float16_quantization(
        self, test_name: str, model_name: str, task: str, tflite_config_constructor: "ExportConfigConstructor"
    ):
        self._tflite_export(model_name, tflite_config_constructor, task=task, quantization="fp16")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_tf
    @pytest.mark.quantization
    def test_exporters_cli_tflite_int8_dynamic_quantization(
        self,
        test_name: str,
        model_name: str,
        task: str,
        tflite_config_constructor: "ExportConfigConstructor",
    ):
        self._tflite_export(model_name, tflite_config_constructor, task=task, quantization="int8-dynamic")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_tf
    @pytest.mark.quantization
    def test_exporters_cli_tflite_full_int8_quantization_with_default_dataset(
        self,
        test_name: str,
        model_name: str,
        task: str,
        tflite_config_constructor: "ExportConfigConstructor",
    ):
        # TODO: currently only 4 tasks are supported.
        if task not in TASK_TO_NON_DEFAULT_DATASET:
            return

        self._tflite_export(
            model_name,
            tflite_config_constructor,
            task=task,
            quantization="int8",
            num_calibration_samples=3,
            inputs_dtype="int8",
            outputs_dtype="int8",
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_tf
    @pytest.mark.quantization
    def test_exporters_cli_tflite_int8_quantization_with_default_dataset(
        self,
        test_name: str,
        model_name: str,
        task: str,
        tflite_config_constructor: "ExportConfigConstructor",
    ):
        # TODO: currently only 4 tasks are supported.
        if task not in TASK_TO_NON_DEFAULT_DATASET:
            return
        self._tflite_export(
            model_name, tflite_config_constructor, task=task, quantization="int8", num_calibration_samples=3
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_tf
    @pytest.mark.quantization
    def test_exporters_cli_tflite_int8x16_quantization_with_default_dataset(
        self,
        test_name: str,
        model_name: str,
        task: str,
        tflite_config_constructor: "ExportConfigConstructor",
    ):
        # TODO: currently only 4 tasks are supported.
        if task not in TASK_TO_NON_DEFAULT_DATASET:
            return
        self._tflite_export(
            model_name, tflite_config_constructor, task=task, quantization="int8x16", num_calibration_samples=3
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_tf
    @pytest.mark.quantization
    def test_exporters_cli_tflite_int8_quantization_with_custom_dataset(
        self,
        test_name: str,
        model_name: str,
        task: str,
        tflite_config_constructor: "ExportConfigConstructor",
    ):
        # TODO: currently only 4 tasks are supported.
        if task not in TASK_TO_NON_DEFAULT_DATASET:
            return

        custom_dataset = TASK_TO_NON_DEFAULT_DATASET[task]["dataset_args"]
        config_name = None
        if isinstance(custom_dataset, dict):
            config_name = custom_dataset.get("name", None)
            custom_dataset = custom_dataset["path"]

        data_keys = TASK_TO_NON_DEFAULT_DATASET[task]["dataset_data_keys"]
        kwargs = {f"{key_name}_key": value for key_name, value in data_keys.items()}

        self._tflite_export(
            model_name,
            tflite_config_constructor,
            task=task,
            quantization="int8",
            calibration_dataset_name_or_path=custom_dataset,
            calibration_dataset_config_name=config_name,
            num_calibration_samples=3,
            **kwargs,
        )

    @pytest.mark.skip("Not supported yet since we only support the export for BERT")
    def test_trust_remote_code(self):
        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.tflite --model fxmarty/tiny-testing-gpt2-remote-code --task causal-lm {tmpdirname}",
                shell=True,
                capture_output=True,
            )
            self.assertTrue(out.returncode, 1)
            self.assertTrue("requires you to execute the modeling file in that repo" in out.stderr.decode("utf-8"))

        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.tflite --trust-remote-code --model fxmarty/tiny-testing-gpt2-remote-code --task causal-lm {tmpdirname}",
                shell=True,
                check=True,
            )
