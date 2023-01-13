# coding=utf-8
#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import json
import logging
import os
import sys
import unittest
from unittest.mock import patch

import torch
from transformers.file_utils import is_apex_available
from transformers.testing_utils import CaptureLogger, TestCasePlus, get_gpu_count, slow, torch_device


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-classification",
        "token-classification",
        "question-answering",
        "translation",
    ]
]
sys.path.extend(SRC_DIRS)
if SRC_DIRS is not None:
    import run_glue
    import run_ner
    import run_qa
    import run_translation


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results


def is_cuda_and_apex_available():
    is_using_cuda = torch.cuda.is_available() and torch_device == "cuda"
    return is_using_cuda and is_apex_available()


class ExamplesTests(TestCasePlus):

    # Text Classification Tests
    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_glue.py
            --model_name_or_path bert-base-uncased     
            --task_name sst2
            --do_train
            --do_eval
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --learning_rate=1e-5
            --per_device_train_batch_size=16
            --per_device_eval_batch_size=16
            """.split()

        with patch.object(sys, "argv", testargs):
            run_glue.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)

    # Token Classification Tests
    def test_run_ner(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        # with so little data distributed training needs more epochs to get the score on par with 0/1 gpu
        epochs = 7 if get_gpu_count() > 1 else 2

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_ner.py
            --model_name_or_path bert-base-uncased
            --dataset_name conll2003
            --do_train
            --do_eval
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --learning_rate=1e-5
            --per_device_train_batch_size=16
            --per_device_eval_batch_size=16
            --num_train_epochs={epochs}
        """.split()

        with patch.object(sys, "argv", testargs):
            run_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)
            self.assertLess(result["eval_loss"], 0.5)

    # Question Answering Tests
    def test_run_qa(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_qa.py
            --model_name_or_path bert-base-uncased
            --dataset_name squad
            --do_train
            --do_eval
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --learning_rate=1e-5
            --per_device_train_batch_size=16
            --per_device_eval_batch_size=16
        """.split()

        with patch.object(sys, "argv", testargs):
            run_qa.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_f1"], 30)
            self.assertGreaterEqual(result["eval_exact"], 30)

    @slow
    def test_run_translation(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_translation.py
            --model_name_or_path t5-large
            --source_lang en
            --target_lang ro
            --dataset_name wmt16
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=50
            --warmup_steps=8
            --do_train
            --learning_rate=3e-3
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --predict_with_generate
        """.split()

        with patch.object(sys, "argv", testargs):
            run_translation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_bleu"], 30)


if __name__ == "__main__":
    unittest.main()
