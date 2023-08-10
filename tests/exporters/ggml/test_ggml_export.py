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
import shutil
import subprocess
from tempfile import TemporaryDirectory, mkdtemp
from typing import Dict
from unittest import TestCase

import torch
from parameterized import parameterized
from transformers import AutoTokenizer, is_torch_available

from optimum.exporters import ggml
from optimum.exporters.ggml.__main__ import main_export

from ..exporters_utils import PYTORCH_EXPORT_MODELS_TINY


if is_torch_available():
    from optimum.exporters.tasks import TasksManager


SEED = 42


def _get_models_to_test(export_models_dict: Dict):
    models_to_test = []
    if is_torch_available():
        for model_type, model_names_tasks in export_models_dict.items():
            try:
                task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "ggml")

                # Filter only for supported tasks
                for task in list(task_config_mapping.keys()):
                    if task not in TasksManager._SUPPORTED_GGML_TASKS:
                        del task_config_mapping[task]

            except KeyError:
                # In this case the model is either not supported, or the contributor forgot to register the
                # GgmlConfig in the TasksManager.
                # We check that supported model was left unregistered for a backend in the TasksManager unit tests, so
                # we can simply skip in this case here.
                continue

            if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
                tasks = list(task_config_mapping.keys())
                model_tasks = {model_names_tasks: tasks}
            else:
                unique_tasks = set()
                for tasks in model_names_tasks.values():
                    for task in tasks:
                        unique_tasks.add(task)
                n_tested_tasks = len(unique_tasks)
                if n_tested_tasks != len(task_config_mapping):
                    raise ValueError(f"Not all tasks are tested for {model_type}.")
                model_tasks = model_names_tasks  # possibly, test different tasks on different models

            for model_name, tasks in model_tasks.items():
                for task in tasks:
                    models_to_test.append((model_type, model_name, task))

        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy")]


class GGMLExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = mkdtemp()
        cls.temp_dir_src = os.path.join(cls.temp_dir, "src")

        makefile_directory = os.path.join(os.path.dirname(ggml.__file__), "src")
        shutil.copytree(makefile_directory, cls.temp_dir_src)

        subprocess.run(["make", "clean"], cwd=cls.temp_dir_src)

    def tearDown(self):
        subprocess.run(["make", "clean"], cwd=self.temp_dir_src)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def _ggml_export(
        self,
        model_type: str,
        model_name: str,
        task: str,
        fp16: bool = False,
    ):
        subprocess.run(["make", model_type], cwd=self.temp_dir_src)

        prompt = "A"

        n_predict = 128
        seed = 0
        top_k = 40
        top_p = 0.9
        temperature = 1e-6  # approximately greedy decoding

        with TemporaryDirectory() as tmpdir:
            source_model = main_export(
                model_name_or_path=model_name,
                output=tmpdir,
                task=task,
                fp16=fp16,
                return_source_model=True,
            )

            assert len(os.listdir(tmpdir)) == 1
            bin_file = os.listdir(tmpdir)[0]
            assert bin_file.endswith(".bin")

            cpp_main_path = os.path.join(self.temp_dir_src, "main")
            model_file_path = os.path.join(tmpdir, bin_file)

            args = [cpp_main_path, "--model", model_file_path]
            args += ["--n_predict", n_predict]
            args += ["--seed", seed]
            args += ["--top_k", top_k]
            args += ["--top_p", top_p]
            args += ["--temp", temperature]
            args += ["--prompt", prompt]

            process = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert process.returncode == 0
            cpp_output = process.stdout.split("sampling parameters")[1].split("\n")[3:-7][0].replace(" [end of text]", "")

            source_tokenizer = AutoTokenizer.from_pretrained(model_name)
            source_input_tokens = source_tokenizer(prompt, return_tensors="pt")
            torch.manual_seed(seed)
            source_output_tokens = source_model.generate(
                **source_input_tokens,
                max_new_tokens=n_predict,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
            )
            source_output = source_tokenizer.decode(source_output_tokens[0].tolist())

            # TODO asserting the logits would be better, but requires modifications to the CPP generation
            #  This is because small models just tend to repeat the prompt
            # Check if generated texts are equal
            assert cpp_output == source_output


    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    def test_exporters_cli_pytorch_cpu(self, model_type: str, model_name: str, task: str):
        self._ggml_export(model_type, model_name, task)
