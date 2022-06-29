import dataclasses
import json
import os
import subprocess
import unittest
from datetime import datetime

from optimum.onnxruntime.runs import OnnxRuntimeRun
from optimum.utils.runs import RunConfig


class TestTextClassification(unittest.TestCase):
    def test_eval_transformers_examples(self):
        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_textclassification"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 3 --filter=blob:none --sparse https://github.com/huggingface/transformers", shell=True
        )

        os.chdir("transformers")
        subprocess.run("git sparse-checkout set examples/pytorch/text-classification", shell=True)

        subprocess.run(
            f"python3 examples/pytorch/text-classification/run_glue.py"
            f" --model_name_or_path howey/bert-base-uncased-sst2"
            f" --task_name sst2"
            f" --do_eval"
            f" --max_seq_length 9999999999"  # rely on tokenizer.model_max_length for max_length
            f" --output_dir {dir_path}/textclassification_sst2_transformers"
            f" --max_eval_samples 100",
            shell=True,
        )

        eval_filename = f"{dir_path}/textclassification_sst2_transformers/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        transformers_accuracy = eval_dict["eval_accuracy"]

        run_config = {
            "task": "text-classification",
            "task_args": {"is_regression": False},
            "model_name_or_path": "howey/bert-base-uncased-sst2",
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
            "batch_sizes": [8],
            "input_lengths": [128],
        }
        run_config = RunConfig(**run_config)
        run_config = dataclasses.asdict(run_config)

        run_instance = OnnxRuntimeRun(run_config)
        run_dict = run_instance.launch()

        optimum_benchmark_accuracy = run_dict["evaluation"]["others"]["baseline"]["accuracy"]

        print("Benchmark suite metrics :")
        print(run_dict["evaluation"]["others"]["baseline"])

        print("Transformers examples metrics:")
        print(eval_dict)

        self.assertEqual(transformers_accuracy, optimum_benchmark_accuracy)

    def test_eval_optimum_examples(self):
        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_textclassification"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 4 --filter=blob:none --sparse https://github.com/huggingface/optimum", shell=True
        )

        os.chdir("optimum")
        subprocess.run("git sparse-checkout set examples/onnxruntime/quantization/text-classification", shell=True)

        subprocess.run(
            f"python3 examples/onnxruntime/quantization/text-classification/run_glue.py"
            f" --model_name_or_path howey/bert-base-uncased-sst2"
            f" --task_name sst2"
            f" --do_eval"
            f" --max_seq_length 9999999999"  # rely on tokenizer.model_max_length for max_length
            f" --output_dir {dir_path}/textclassification_sst2_optimum"
            f" --max_eval_samples 100"
            f" --opset 11"
            f" --quantization_approach dynamic"
            f" --overwrite_cache True",
            shell=True,
        )

        eval_filename = f"{dir_path}/textclassification_sst2_optimum/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        optimum_accuracy = eval_dict["accuracy"]

        run_config = {
            "task": "text-classification",
            "task_args": {"is_regression": False},
            "model_name_or_path": "howey/bert-base-uncased-sst2",
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
            "framework_args": {"optimization_level": 1, "opset": 11},
            "batch_sizes": [8],
            "input_lengths": [128],
        }
        run_config = RunConfig(**run_config)
        run_config = dataclasses.asdict(run_config)

        run_instance = OnnxRuntimeRun(run_config)
        run_dict = run_instance.launch()

        optimum_benchmark_accuracy = run_dict["evaluation"]["others"]["optimized"]["accuracy"]

        print("Benchmark suite metrics :")
        print(run_dict["evaluation"]["others"]["optimized"])

        print("Optimum examples metrics:")
        print(eval_dict)

        self.assertEqual(optimum_accuracy, optimum_benchmark_accuracy)
