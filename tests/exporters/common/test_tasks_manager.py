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
import importlib
import inspect
from typing import Optional, Set
from unittest import TestCase

import pytest
from transformers import BertConfig, Pix2StructForConditionalGeneration, VisualBertForQuestionAnswering
from transformers.testing_utils import slow

from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.tasks import TasksManager


class TasksManagerTestCase(TestCase):
    def _check_all_models_are_registered(
        self, backend: str, class_prefix: str, classes_to_ignore: Optional[Set[str]] = None
    ):
        registered_classes = set()
        for mappings in TasksManager._SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)
        for mappings in TasksManager._TIMM_SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)
        for mappings in TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)
        for mappings in TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)

        if classes_to_ignore is None:
            classes_to_ignore = set()

        module_name = f"optimum.exporters.{backend}.model_configs"

        def predicate(member):
            name = getattr(member, "__name__", "")
            module = getattr(member, "__module__", "")
            return all(
                (
                    inspect.isclass(member),
                    module == module_name,
                    name.endswith(class_prefix),
                    name not in classes_to_ignore,
                )
            )

        defined_classes = inspect.getmembers(importlib.import_module(module_name), predicate)

        # inspect.getmembers returns a list of (name, value) tuples, so we retrieve the names here.
        defined_classes = {x[0] for x in defined_classes}

        diff = defined_classes - registered_classes
        if diff:
            raise ValueError(
                f"Some models were defined for the {backend} backend, but never registered in the TasksManager: "
                f"{', '.join(diff)}."
            )

    def test_all_onnx_models_are_registered(self):
        return self._check_all_models_are_registered("onnx", "OnnxConfig")

    def test_register(self):
        # Case 1: We try to register a config that was already registered, it should not register anything.
        register_for_onnx = TasksManager.create_register("onnx")

        @register_for_onnx("bert", "text-classification")
        class BadBertOnnxConfig(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "onnx",
            model_type="bert",
            task="text-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertNotEqual(
            bert_onnx_config.__class__,
            BadBertOnnxConfig,
            "Registering an already existing config constructor should not do anything unless overwrite_existing=True.",
        )

        # Case 2: We try to register a config that was already registered, but authorize overwriting, it should register
        # the new config.
        register_for_onnx = TasksManager.create_register("onnx", overwrite_existing=True)

        @register_for_onnx("bert", "text-classification")
        class BadBertOnnxConfig2(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "onnx",
            model_type="bert",
            task="text-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertEqual(
            bert_onnx_config.__class__,
            BadBertOnnxConfig2,
            (
                "Registering an already existing config constructor with overwrite_existing=True should overwrite the "
                "old config constructor."
            ),
        )

        # Case 3: Registering an unknown task.
        with self.assertRaisesRegex(ValueError, "The TasksManager does not know the task called"):

            @register_for_onnx("bert", "this is a wrong name for a task")
            class UnknownTask(BertOnnxConfig):
                pass

        # Case 4: Registering for a new backend.
        register_for_new_backend = TasksManager.create_register("new-backend")

        @register_for_new_backend("bert", "text-classification")
        class BertNewBackendConfig(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "new-backend",
            model_type="bert",
            task="text-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertEqual(
            bert_onnx_config.__class__, BertNewBackendConfig, "Wrong config class compared to the registered one."
        )

        # Case 5: Registering a new task for a already existing backend.
        @register_for_new_backend("bert", "token-classification")
        class BertNewBackendConfigTaskSpecific(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "new-backend",
            model_type="bert",
            task="token-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertEqual(
            bert_onnx_config.__class__,
            BertNewBackendConfigTaskSpecific,
            "Wrong config class compared to the registered one.",
        )

    @slow
    @pytest.mark.run_slow
    def test_custom_class(self):
        task = TasksManager.infer_task_from_model("google/pix2struct-base")
        self.assertEqual(task, "image-to-text")

        model = TasksManager.get_model_from_task("image-to-text", "google/pix2struct-base")
        self.assertTrue(isinstance(model, Pix2StructForConditionalGeneration))

        model = TasksManager.get_model_from_task("question-answering", "uclanlp/visualbert-vqa")
        self.assertTrue(isinstance(model, VisualBertForQuestionAnswering))

    def test_library_detection(self):
        self.assertEqual(
            TasksManager.infer_library_from_model("intfloat/multilingual-e5-large"), "sentence_transformers"
        )
        self.assertEqual(
            TasksManager.infer_library_from_model("stabilityai/stable-diffusion-xl-base-1.0"), "diffusers"
        )
        self.assertEqual(TasksManager.infer_library_from_model("gpt2"), "transformers")
        self.assertEqual(TasksManager.infer_library_from_model("timm/mobilenetv3_large_100.ra_in1k"), "timm")
