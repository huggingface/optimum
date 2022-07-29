import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import transformers

from optimum.onnxruntime.runs import OnnxRuntimeRun
from optimum.pytorch.runs import PyTorchRun


class TestParity(unittest.TestCase):
    def setUp(self):
        self.dir_path = tempfile.mkdtemp("test_transformers_optimum_examples_parity")

        transformers_version = transformers.__version__
        branch = ""
        if not transformers_version.endswith(".dev0"):
            branch = f"--branch v{transformers_version}"
        subprocess.run(
            f"git clone --depth 3 --filter=blob:none --sparse {branch} https://github.com/huggingface/transformers",
            shell=True,
            cwd=self.dir_path,
        )

    def tearDown(self):
        shutil.rmtree(self.dir_path)

    def test_text_classification_parity(self):
        model_name = "philschmid/tiny-bert-sst2-distilled"
        n_samples = 100

        subprocess.run(
            "git sparse-checkout set examples/pytorch/text-classification",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        subprocess.run(
            f"python3 examples/pytorch/text-classification/run_glue.py"
            f" --model_name_or_path {model_name}"
            f" --task_name sst2"
            f" --do_eval"
            f" --max_seq_length 9999999999"  # rely on tokenizer.model_max_length for max_length
            f" --output_dir {self.dir_path}/textclassification_sst2_transformers"
            f" --max_eval_samples {n_samples}",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'textclassification_sst2_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        subprocess.run(
            f"python3 examples/onnxruntime/quantization/text-classification/run_glue.py"
            f" --model_name_or_path {model_name}"
            f" --task_name sst2"
            f" --do_eval"
            f" --max_seq_length 9999999999"  # rely on tokenizer.model_max_length for max_length
            f" --output_dir {self.dir_path}/textclassification_sst2_optimum"
            f" --max_eval_samples {n_samples}"
            f" --opset 11"
            f" --quantization_approach dynamic"
            f" --overwrite_cache True",
            shell=True,
        )

        with open(os.path.join(self.dir_path, "textclassification_sst2_optimum", "eval_results.json"), "r") as f:
            optimum_results = json.load(f)

        run_config = {
            "task": "text-classification",
            "task_args": {"is_regression": False},
            "model_name_or_path": model_name,
            "dataset": {
                "path": "glue",
                "name": "sst2",
                "eval_split": "validation",
                "data_keys": {"primary": "sentence"},
                "ref_keys": ["label"],
            },
            "metrics": ["accuracy"],
            "framework": "pytorch",
            "max_eval_samples": n_samples,
        }
        run_instance = PyTorchRun(run_config)
        benchmark_results = run_instance.launch_eval()

        self.assertEqual(transformers_results["eval_accuracy"], benchmark_results["evaluation"]["others"]["accuracy"])

        run_config = {
            "task": "text-classification",
            "task_args": {"is_regression": False},
            "model_name_or_path": model_name,
            "dataset": {
                "path": "glue",
                "name": "sst2",
                "eval_split": "validation",
                "data_keys": {"primary": "sentence"},
                "ref_keys": ["label"],
            },
            "metrics": ["accuracy"],
            "quantization_approach": "dynamic",
            "operators_to_quantize": ["Add", "MatMul"],
            "node_exclusion": [],
            "per_channel": False,
            "framework": "onnxruntime",
            "framework_args": {"optimization_level": 1, "opset": 15},
            "max_eval_samples": n_samples,
        }
        run_instance = OnnxRuntimeRun(run_config)
        benchmark_results = run_instance.launch_eval()

        self.assertEqual(optimum_results["accuracy"], benchmark_results["evaluation"]["others"]["accuracy"])

    def test_token_classification_parity(self):
        model_name = "hf-internal-testing/tiny-bert-for-token-classification"
        n_samples = 200

        subprocess.run(
            "git sparse-checkout set examples/pytorch/token-classification",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        subprocess.run(
            f"python examples/pytorch/token-classification/run_ner.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name conll2003"
            f" --do_eval"
            f" --output_dir {self.dir_path}/tokenclassification_conll2003_transformers"
            f" --max_eval_samples {n_samples}"
            f" --overwrite_cache True",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'tokenclassification_conll2003_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        subprocess.run(
            f"python3 examples/onnxruntime/quantization/token-classification/run_ner.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name conll2003"
            f" --do_eval"
            f" --output_dir {self.dir_path}/tokenclassification_conll2003_optimum"
            f" --max_eval_samples {n_samples}"
            f" --opset 11"
            f" --quantization_approach dynamic"
            f" --overwrite_cache True",
            shell=True,
        )

        with open(os.path.join(self.dir_path, "tokenclassification_conll2003_optimum", "eval_results.json"), "r") as f:
            optimum_results = json.load(f)

        run_config = {
            "task": "token-classification",
            "model_name_or_path": model_name,
            "dataset": {
                "path": "conll2003",
                "eval_split": "validation",
                "data_keys": {"primary": "tokens"},
                "ref_keys": ["ner_tags"],
            },
            "metrics": ["seqeval"],
            "framework": "pytorch",
            "max_eval_samples": n_samples,
        }
        run_instance = PyTorchRun(run_config)
        benchmark_results = run_instance.launch_eval()

        self.assertEqual(
            transformers_results["eval_accuracy"], benchmark_results["evaluation"]["others"]["overall_accuracy"]
        )
        self.assertEqual(transformers_results["eval_f1"], benchmark_results["evaluation"]["others"]["overall_f1"])

        run_config = {
            "task": "token-classification",
            "model_name_or_path": model_name,
            "dataset": {
                "path": "conll2003",
                "eval_split": "validation",
                "data_keys": {"primary": "tokens"},
                "ref_keys": ["ner_tags"],
            },
            "metrics": ["seqeval"],
            "quantization_approach": "dynamic",
            "operators_to_quantize": ["Add", "MatMul"],
            "node_exclusion": [],
            "per_channel": False,
            "framework": "onnxruntime",
            "framework_args": {"optimization_level": 1, "opset": 11},
            "max_eval_samples": n_samples,
        }
        run_instance = OnnxRuntimeRun(run_config)
        benchmark_results = run_instance.launch_eval()

        self.assertEqual(optimum_results["accuracy"], benchmark_results["evaluation"]["others"]["overall_accuracy"])
        self.assertEqual(optimum_results["f1"], benchmark_results["evaluation"]["others"]["overall_f1"])

    def test_question_answering_parity(self):
        model_name = "mrm8488/bert-tiny-finetuned-squadv2"
        n_samples = 100

        subprocess.run(
            "git sparse-checkout set examples/pytorch/question-answering",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        # test squad_v1-like dataset
        subprocess.run(
            f"python examples/pytorch/question-answering/run_qa.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name squad"
            f" --do_eval"
            f" --output_dir {os.path.join(self.dir_path, 'questionanswering_squad_transformers')}"
            f" --max_eval_samples {n_samples}"
            f" --max_seq_length 384",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'questionanswering_squad_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        subprocess.run(
            f"python3 examples/onnxruntime/quantization/question-answering/run_qa.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name squad"
            f" --do_eval"
            f" --output_dir {self.dir_path}/questionanswering_squad_optimum"
            f" --max_eval_samples {n_samples}"
            f" --opset 11"
            f" --quantization_approach dynamic"
            f" --max_seq_length 384",
            shell=True,
        )

        with open(os.path.join(self.dir_path, "questionanswering_squad_optimum", "eval_results.json"), "r") as f:
            optimum_results = json.load(f)

        run_config = {
            "task": "question-answering",
            "model_name_or_path": model_name,
            "dataset": {
                "path": "squad",
                "eval_split": "validation",
                "data_keys": {"question": "question", "context": "context"},
                "ref_keys": ["answers"],
            },
            "metrics": ["squad"],
            "framework": "pytorch",
            "max_eval_samples": n_samples,
        }
        run_instance = PyTorchRun(run_config)
        benchmark_results = run_instance.launch_eval()

        self.assertEqual(transformers_results["eval_f1"], benchmark_results["evaluation"]["others"]["f1"])
        self.assertEqual(
            transformers_results["eval_exact_match"],
            benchmark_results["evaluation"]["others"]["exact_match"],
        )

        run_config = {
            "task": "question-answering",
            "model_name_or_path": model_name,
            "dataset": {
                "path": "squad",
                "eval_split": "validation",
                "data_keys": {"question": "question", "context": "context"},
                "ref_keys": ["answers"],
            },
            "metrics": ["squad"],
            "quantization_approach": "dynamic",
            "operators_to_quantize": ["Add", "MatMul"],
            "node_exclusion": [],
            "per_channel": False,
            "framework": "onnxruntime",
            "framework_args": {"optimization_level": 1, "opset": 15},
            "max_eval_samples": n_samples,
        }
        run_instance = OnnxRuntimeRun(run_config)
        benchmark_results = run_instance.launch_eval()

        self.assertEqual(optimum_results["f1"], benchmark_results["evaluation"]["others"]["f1"])
        self.assertEqual(optimum_results["exact_match"], benchmark_results["evaluation"]["others"]["exact_match"])

    def test_image_classification_parity(self):
        # dummy test until this question is solved
        # https://discuss.huggingface.co/t/why-use-val-transforms-function-in-image-classification-example-instead-of-feature-extractor/19976
        model_name = "fxmarty/resnet-tiny-beans"
        n_samples = 120

        run_config = {
            "task": "image-classification",
            "model_name_or_path": model_name,
            "dataset": {
                "path": "beans",
                "eval_split": "validation",
                "data_keys": {"primary": "image"},
                "ref_keys": ["labels"],
            },
            "metrics": ["accuracy"],
            "quantization_approach": "dynamic",
            "operators_to_quantize": ["Add", "MatMul"],
            "node_exclusion": [],
            "per_channel": False,
            "framework": "onnxruntime",
            "framework_args": {"optimization_level": 1, "opset": 15},
            "max_eval_samples": n_samples,
        }
        run_instance = OnnxRuntimeRun(run_config)
        benchmark_results = run_instance.launch_eval()

        self.assertEqual(benchmark_results["evaluation"]["others"]["accuracy"], 84 / 120)


if __name__ == "__main__":
    unittest.main()
