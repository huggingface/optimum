# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

import pytest
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers.testing_utils import (
    slow,
)

from optimum.executorchruntime import ExecuTorchModelForCausalLM


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_load_model_from_hub(self):
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_name_or_path="NousResearch/Llama-3.2-1B",
            export=True,
            task="text-generation",
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

    @slow
    @pytest.mark.run_slow
    def test_load_model_from_local_path(self):
        from optimum.exporters.executorch import main_export

        model_id = "NousResearch/Llama-3.2-1B"
        task = "text-generation"
        recipe = "xnnpack"

        with tempfile.TemporaryDirectory() as tempdir:
            # Export to a local dir
            main_export(
                model_name_or_path=model_id,
                task=task,
                recipe=recipe,
                output_dir=tempdir,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/model.pte"))

            # Load the exported model from a local dir
            model = ExecuTorchModelForCausalLM.from_pretrained(
                model_name_or_path=tempdir,
                export=False,
            )
            self.assertIsInstance(model, ExecuTorchModelForCausalLM)
            self.assertIsInstance(model.model, ExecuTorchModule)
