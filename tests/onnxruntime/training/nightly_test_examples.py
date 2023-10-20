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

import pytest
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

    def test_token_classification(self):
        subprocess.run(
            "cp ../examples/onnxruntime/training/token-classification/run_ner.py ./",
            shell=True,
        )

        subprocess.run(
            "torchrun"
            " --nproc_per_node=1"
            " run_ner.py"
            " --model_name_or_path bert-base-cased"
            " --dataset_name conll2003"
            " --do_train"
            " --output_dir /tmp/bert"
            " --overwrite_output_dir"
            " --max_steps 50"
            " --logging_steps 50"
            " --per_device_train_batch_size 8"
            " --fp16 --optim adamw_ort_fused"
            " --max_train_samples 20",
            shell=True,
            check=True,
        )

    def test_translation(self):
        subprocess.run(
            "cp ../examples/onnxruntime/training/translation/run_translation.py ./",
            shell=True,
        )

        subprocess.run(
            "torchrun"
            " --nproc_per_node=1"
            " run_translation.py"
            " --model_name_or_path t5-small"
            " --dataset_name wmt16"
            " --dataset_config ro-en"
            " --label_smoothing 0.1"
            " --predict_with_generate"
            " --source_lang en"
            " --target_lang ro"
            " --do_train"
            " --max_train_samples 30"
            " --output_dir /tmp/t5"
            " --overwrite_output_dir"
            " --max_steps 50"
            " --logging_steps 50"
            " --per_device_train_batch_size 2"
            " --fp16 --optim adamw_ort_fused",
            shell=True,
            check=True,
        )

    @pytest.mark.skip(reason="skip for now")
    def test_summarization(self):
        subprocess.run(
            "cp ../examples/onnxruntime/training/summarization/run_summarization.py ./",
            shell=True,
        )

        subprocess.run(
            "torchrun"
            " --nproc_per_node=1"
            " run_summarization.py"
            " --model_name_or_path t5-small"
            " --do_train"
            " --do_eval"
            " --dataset_name cnn_dailymail"
            ' --dataset_config "3.0.0"'
            ' --source_prefix "summarize: "'
            " --predict_with_generate"
            " --max_train_samples 30"
            " --output_dir /tmp/t5"
            " --overwrite_output_dir"
            " --max_steps 50"
            " --logging_steps 50"
            " --per_device_train_batch_size 2"
            " --per_device_eval_batch_size 2"
            " --fp16 --optim adamw_ort_fused",
            shell=True,
            check=True,
        )

    # TODO: Update the example and add the test
    def test_stable_diffusion_txt2img(self):
        pass

    @pytest.mark.skip(reason="skip for now")
    def test_question_answering(self):
        subprocess.run(
            "cp ../examples/onnxruntime/training/question-answering/run_qa.py ./",
            shell=True,
        )

        subprocess.run(
            "torchrun"
            " --nproc_per_node=1"
            " run_qa.py"
            " --model_name_or_path bert-base-uncased"
            " --do_train"
            " --do_eval"
            " --dataset_name squad"
            " --max_train_samples 30"
            " --output_dir /tmp/bert"
            " --overwrite_output_dir"
            " --max_steps 50"
            " --logging_steps 50"
            " --per_device_train_batch_size 2"
            " --per_device_eval_batch_size 2"
            " --fp16 --optim adamw_ort_fused",
            shell=True,
            check=True,
        )

    @pytest.mark.skip(reason="skip for now")
    def test_language_modeling(self):
        subprocess.run(
            "cp ../examples/onnxruntime/training/question-answering/run_qa.py ./",
            shell=True,
        )

        subprocess.run(
            "torchrun"
            " --nproc_per_node=1"
            " run_clm.py"
            " --model_name_or_path gpt2"
            " --do_train"
            " --do_eval"
            " --dataset_name wikitext"
            " --dataset_config_name wikitext-2-raw-v1"
            " --max_train_samples 30"
            " --output_dir /tmp/gpt2"
            " --overwrite_output_dir"
            " --max_steps 50"
            " --logging_steps 50"
            " --per_device_train_batch_size 2"
            " --per_device_eval_batch_size 2"
            " --fp16 --optim adamw_ort_fused",
            shell=True,
            check=True,
        )

    @pytest.mark.skip(reason="skip for now")
    def test_image_classification(self):
        subprocess.run(
            "cp ../examples/onnxruntime/training/image-classification/run_image_classification.py ./",
            shell=True,
        )

        subprocess.run(
            "torchrun"
            " --nproc_per_node=1"
            " run_image_classification.py"
            " --model_name_or_path google/vit-base-patch16-224-in21k"
            " --do_train"
            " --do_eval"
            " --dataset_name beans"
            " --max_train_samples 30"
            " --output_dir /tmp/vit"
            " --overwrite_output_dir"
            " --max_steps 50"
            " --logging_steps 50"
            " --per_device_train_batch_size 2"
            " --per_device_eval_batch_size 2"
            " --fp16 --optim adamw_ort_fused",
            shell=True,
            check=True,
        )
