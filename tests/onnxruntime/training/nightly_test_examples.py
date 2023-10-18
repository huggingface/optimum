# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team.
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
"""Test ONNX Runtime Training Examples in Optimum."""

import subprocess
import unittest
from transformers.testing_utils import slow


@slow
class ORTTrainerExampleTest(unittest.TestCase):
    def test_text_classification(self):
        subprocess.run(
            "cp ../examples/onnxruntime/training/text-classification/run_glue.py ./",
            shell=True,
        )

        subprocess.run(
            "torchrun"
            " --nproc_per_node=1"
            " run_glue.py"
            " --model_name_or_path distilbert-base-uncased"
            " --task_name mnli"
            " --max_seq_length 64"
            " --learning_rate 3e-6"
            " --do_train"
            " --output_dir /tmp/distilbert"
            " --overwrite_output_dir"
            " --max_steps 50"
            " --logging_steps 50"
            " --per_device_train_batch_size 8"
            " --fp16 --optim adamw_ort_fused"
            " --max_train_samples 20",
            shell=True,
            check=True,
        )

    # TODO: Test all ORT training examples
    def test_token_classification(self):
        pass

    def test_translation(self):
        pass

    def test_summarization(self):
        pass

    def test_stable_diffusion_txt2img(self):
        pass

    def test_question_answering(self):
        pass

    def test_language_modeling(self):
        pass

    def test_image_classification(self):
        pass
