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

from optimum.exporters import TasksManager


class TasksManagerTestCase(TestCase):
    def _check_all_models_are_registered(
        self, backend: str, class_prefix: str, classes_to_ignore: Optional[Set[str]] = None
    ):
        registered_classes = set()
        for mappings in TasksManager._SUPPORTED_MODEL_TYPE.values():
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
        defined_classes = set(map(lambda x: x[0], defined_classes))

        diff = defined_classes - registered_classes
        if diff:
            raise ValueError(
                f"Some models were defined for the {backend} backend, but never registered in the TasksManager: "
                f"{', '.join(diff)}."
            )

    def test_all_onnx_models_are_registered(self):
        return self._check_all_models_are_registered("onnx", "OnnxConfig")

    def test_all_tflite_models_are_registered(self):
        return self._check_all_models_are_registered("tflite", "TFLiteConfig")
