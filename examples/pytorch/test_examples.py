#  Copyright 2021 The HuggingFace Team. All rights reserved.
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
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


os.environ["CUDA_VISIBLE_DEVICES"] = ""


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-classification",
        "question-answering",
        "token-classification",
    ]
]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:
    import run_glue
    import run_qa
    import run_ner


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"Can't find {path}.")
    return results


class TestExamples(unittest.TestCase):

    def test_run_glue(self):
        provider = "inc"
        quantization_approach = "dynamic"

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_glue.py
                --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english
                --task_name=sst2
                --provider={provider}
                --quantize
                --quantization_approach={quantization_approach}
                --do_eval
                --per_device_eval_batch_size=8
                --verify_loading
                --output_dir={tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_glue.main()
                results = get_results(tmp_dir)
                self.assertGreaterEqual(results["eval_accuracy"], 0.85)

    def test_run_qa(self):
        provider = "inc"
        quantization_approach = "dynamic"

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_qa.py
                --model_name_or_path distilbert-base-uncased-distilled-squad
                --dataset_name=squad
                --provider={provider}
                --quantize
                --quantization_approach={quantization_approach}
                --do_eval
                --per_device_eval_batch_size=8
                --verify_loading
                --output_dir={tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_qa.main()
                results = get_results(tmp_dir)
                self.assertGreaterEqual(results["eval_f1"], 80)
                self.assertGreaterEqual(results["eval_exact_match"], 70)

    def test_run_ner(self):
        provider = "inc"
        quantization_approach = "dynamic"

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_ner.py 
                --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english
                --dataset_name=conll2003
                --provider={provider}
                --quantize
                --quantization_approach={quantization_approach}
                --do_eval
                --verify_loading
                --output_dir={tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_ner.main()
                results = get_results(tmp_dir)
                self.assertGreaterEqual(results["eval_accuracy"], 0.90)
                self.assertGreaterEqual(results["eval_f1"], 0.90)
                self.assertGreaterEqual(results["eval_precision"], 0.90)
                self.assertGreaterEqual(results["eval_recall"], 0.90)


if __name__ == "__main__":
    unittest.main()

