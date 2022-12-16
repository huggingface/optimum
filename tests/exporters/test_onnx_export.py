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
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict
from unittest import TestCase
from unittest.mock import patch

import pytest
from transformers import AutoConfig, is_tf_available, is_torch_available
from transformers.testing_utils import require_onnx, require_tf, require_torch, require_torch_gpu, require_vision, slow

from exporters_utils import (
    PYTORCH_EXPORT_MODELS_TINY,
    TENSORFLOW_EXPORT_MODELS,
    VALIDATE_EXPORT_ON_SHAPES_FAST,
    VALIDATE_EXPORT_ON_SHAPES_SLOW,
)
from optimum.exporters.onnx import (
    OnnxConfig,
    OnnxConfigWithPast,
    export,
    export_models,
    get_decoder_models_for_export,
    get_encoder_decoder_models_for_export,
    validate_model_outputs,
    validate_models_outputs,
)
from optimum.utils.testing_utils import grid_parameters
from parameterized import parameterized


if is_torch_available() or is_tf_available():
    from optimum.exporters.tasks import TasksManager


@require_onnx
class OnnxUtilsTestCase(TestCase):
    """
    Covers all the utilities involved to export ONNX models.
    """

    def test_flatten_output_collection_property(self):
        """
        This test ensures we correctly flatten nested collection such as the one we use when returning past_keys.
        past_keys = Tuple[Tuple]

        ONNX exporter will export nested collections as ${collection_name}.${level_idx_0}.${level_idx_1}...${idx_n}
        """
        self.assertEqual(
            OnnxConfig.flatten_output_collection_property("past_key", [[0], [1], [2]]),
            {
                "past_key.0": 0,
                "past_key.1": 1,
                "past_key.2": 2,
            },
        )


class OnnxConfigTestCase(TestCase):
    """
    Covers the test for models default.

    Default means no specific tasks is being enabled on the model.
    """

    # TODO: insert relevant tests here.


class OnnxConfigWithPastTestCase(TestCase):
    """
    Cover the tests for model which have use_cache task (i.e. "with_past" for ONNX)
    """

    SUPPORTED_WITH_PAST_CONFIGS = ()

    @patch.multiple(OnnxConfigWithPast, __abstractmethods__=set())
    def test_use_past(self):
        """
        Ensures the use_past variable is correctly being set.
        """
        for name, config in OnnxConfigWithPastTestCase.SUPPORTED_WITH_PAST_CONFIGS:
            with self.subTest(name):
                self.assertFalse(
                    OnnxConfigWithPast.from_model_config(config()).use_past,
                    "OnnxConfigWithPast.from_model_config() should not use_past",
                )

                self.assertTrue(
                    OnnxConfigWithPast.with_past(config()).use_past,
                    "OnnxConfigWithPast.from_model_config() should use_past",
                )

    @patch.multiple(OnnxConfigWithPast, __abstractmethods__=set())
    def test_values_override(self):
        """
        Ensures the use_past variable correctly set the `use_cache` value in model's configuration.
        """
        for name, config in OnnxConfigWithPastTestCase.SUPPORTED_WITH_PAST_CONFIGS:
            with self.subTest(name):

                # Without past
                onnx_config_default = OnnxConfigWithPast.from_model_config(config())
                self.assertIsNotNone(onnx_config_default.values_override, "values_override should not be None")
                self.assertIn("use_cache", onnx_config_default.values_override, "use_cache should be present")
                self.assertFalse(
                    onnx_config_default.values_override["use_cache"], "use_cache should be False if not using past"
                )

                # With past
                onnx_config_default = OnnxConfigWithPast.with_past(config())
                self.assertIsNotNone(onnx_config_default.values_override, "values_override should not be None")
                self.assertIn("use_cache", onnx_config_default.values_override, "use_cache should be present")
                self.assertTrue(
                    onnx_config_default.values_override["use_cache"], "use_cache should be False if not using past"
                )


def _get_models_to_test(export_models_dict):
    models_to_test = []
    if is_torch_available() or is_tf_available():
        for model_type, model_name in export_models_dict.items():
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")

            for task in task_config_mapping.keys():

                onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                    model_type, "onnx", task=task, model_name=model_name
                )

                models_to_test.append(
                    (f"{model_type}_{task}", model_type, model_name, task, onnx_config_constructor, False)
                )

                if any(
                    task.startswith(ort_special_task)
                    for ort_special_task in ["causal-lm", "seq2seq-lm", "speech2seq-lm"]
                ):
                    models_to_test.append(
                        (f"{model_type}_{task}_for_ort", model_type, model_name, task, onnx_config_constructor, True)
                    )
        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy", "dummy", OnnxConfig.from_model_config)]


class OnnxExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _onnx_export(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
        onnx_config_class_constructor,
        shapes_to_validate: Dict,
        for_ort: bool,
        device="cpu",
    ):
        model_class = TasksManager.get_model_class_for_task(task)
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)

        # Dynamic axes aren't supported for YOLO-like models. This means they cannot be exported to ONNX on CUDA devices.
        # See: https://github.com/ultralytics/yolov5/pull/8378
        if model.__class__.__name__.startswith("Yolos") and device != "cpu":
            return

        onnx_config = onnx_config_class_constructor(model.config)

        # We need to set this to some value to be able to test the outputs values for batch size > 1.
        if (
            isinstance(onnx_config, OnnxConfigWithPast)
            and getattr(model.config, "pad_token_id", None) is None
            and task == "sequence-classification"
        ):
            model.config.pad_token_id = 0

        if is_torch_available():
            from optimum.exporters.onnx.utils import TORCH_VERSION

            if not onnx_config.is_torch_support_available:
                pytest.skip(
                    "Skipping due to incompatible PyTorch version. Minimum required is"
                    f" {onnx_config.MIN_TORCH_VERSION}, got: {TORCH_VERSION}"
                )

        atol = onnx_config.ATOL_FOR_VALIDATION
        if isinstance(atol, dict):
            atol = atol[task.replace("-with-past", "")]

        if for_ort is True and (model.config.is_encoder_decoder or task.startswith("causal-lm")):
            fn_get_models_from_config = (
                get_encoder_decoder_models_for_export
                if model.config.is_encoder_decoder
                else get_decoder_models_for_export
            )

            with TemporaryDirectory() as tmpdirname:
                try:
                    onnx_inputs, onnx_outputs = export_models(
                        model,
                        onnx_config,
                        onnx_config.DEFAULT_ONNX_OPSET,
                        output_dir=Path(tmpdirname),
                        fn_get_models_from_config=fn_get_models_from_config,
                        device=device,
                    )

                    input_shapes_iterator = grid_parameters(shapes_to_validate, yield_dict=True)
                    for input_shapes in input_shapes_iterator:
                        validate_models_outputs(
                            onnx_config,
                            model,
                            onnx_outputs,
                            atol,
                            output_dir=Path(tmpdirname),
                            fn_get_models_from_config=fn_get_models_from_config,
                            input_shapes=input_shapes,
                        )
                except (RuntimeError, ValueError) as e:
                    self.fail(f"{model_type}, {task} -> {e}")
        else:
            with NamedTemporaryFile("w") as output:
                try:
                    onnx_inputs, onnx_outputs = export(
                        model, onnx_config, onnx_config.DEFAULT_ONNX_OPSET, Path(output.name), device=device
                    )

                    input_shapes_iterator = grid_parameters(shapes_to_validate, yield_dict=True, add_test_name=False)
                    for input_shapes in input_shapes_iterator:
                        validate_model_outputs(
                            onnx_config,
                            model,
                            Path(output.name),
                            onnx_outputs,
                            atol,
                            input_shapes=input_shapes,
                        )
                except (RuntimeError, ValueError) as e:
                    self.fail(f"{model_type}, {task} -> {e}")

    def test_all_models_are_tested(self):
        # make sure we test all models
        missing_models_set = set(TasksManager._SUPPORTED_MODEL_TYPE.keys()) - set(PYTORCH_EXPORT_MODELS_TINY.keys())
        if len(missing_models_set) > 0:
            self.fail(f"Not testing all models. Missing models: {missing_models_set}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    def test_pytorch_export(
        self,
        test_name,
        name,
        model_name,
        task,
        onnx_config_class_constructor,
        for_ort: bool,
    ):
        if os.environ.get("RUN_SLOW", False):
            shapes_to_validate = VALIDATE_EXPORT_ON_SHAPES_SLOW
        else:
            shapes_to_validate = VALIDATE_EXPORT_ON_SHAPES_FAST

        self._onnx_export(
            test_name,
            name,
            model_name,
            task,
            onnx_config_class_constructor,
            shapes_to_validate=shapes_to_validate,
            for_ort=for_ort,
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    @require_torch_gpu
    def test_pytorch_export_on_cuda(
        self,
        test_name,
        name,
        model_name,
        task,
        onnx_config_class_constructor,
        for_ort: bool,
    ):
        if os.environ.get("RUN_SLOW", False):
            shapes_to_validate = VALIDATE_EXPORT_ON_SHAPES_SLOW
        else:
            shapes_to_validate = VALIDATE_EXPORT_ON_SHAPES_FAST

        self._onnx_export(
            test_name,
            name,
            model_name,
            task,
            onnx_config_class_constructor,
            device="cuda",
            shapes_to_validate=shapes_to_validate,
            for_ort=for_ort,
        )

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_MODELS))
    @slow
    @require_tf
    @require_vision
    def test_tensorflow_export(self, test_name, name, model_name, task, onnx_config_class_constructor, for_ort: bool):
        if for_ort == True:
            return 0

        self._onnx_export(test_name, name, model_name, task, onnx_config_class_constructor, for_ort=for_ort)
