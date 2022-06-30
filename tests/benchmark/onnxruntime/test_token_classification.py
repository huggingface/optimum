import dataclasses
import json
import os
import subprocess
import unittest
from datetime import datetime

from optimum.onnxruntime.runs import OnnxRuntimeRun
from optimum.utils.runs import RunConfig


class TestTokenClassification(unittest.TestCase):
    def test_eval_transformers(self):
        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_tokenclassification"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 3 --filter=blob:none --sparse https://github.com/huggingface/transformers", shell=True
        )

        os.chdir("transformers")
        subprocess.run("git sparse-checkout set examples/pytorch/token-classification", shell=True)

        subprocess.run(
            f"python3 examples/pytorch/token-classification/run_ner.py"
            f" --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english"
            f" --dataset_name conll2003"
            f" --do_eval"
            f" --output_dir {dir_path}/tokenclassification_conll2003_transformers"
            f" --max_eval_samples 100",
            shell=True,
        )

        eval_filename = f"{dir_path}/tokenclassification_conll2003_transformers/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        transformers_accuracy = eval_dict["eval_accuracy"]
        transformers_f1 = eval_dict["eval_f1"]

        run_config = {
            "task": "token-classification",
            "task_args": {"is_regression": False},
            "model_name_or_path": "elastic/distilbert-base-uncased-finetuned-conll03-english",
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
            "framework_args": {"optimization_level": 1, "opset": 15},
            "batch_sizes": [8],
            "input_lengths": [128],
        }
        run_config = RunConfig(**run_config)
        run_config = dataclasses.asdict(run_config)

        run_instance = OnnxRuntimeRun(run_config)
        run_dict = run_instance.launch()

        optimum_benchmark_accuracy = run_dict["evaluation"]["others"]["baseline"]["accuracy"]
        optimum_benchmark_f1 = run_dict["evaluation"]["others"]["baseline"]["f1"]

        print("Benchmark suite metrics :")
        print(run_dict["evaluation"]["others"]["baseline"])

        print("Transformers examples metrics:")
        print(eval_dict)

        self.assertEqual(transformers_accuracy, optimum_benchmark_accuracy)
        self.assertEqual(transformers_f1, optimum_benchmark_f1)

    def test_eval_optimum(self):
        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_tokenclassification"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 4 --filter=blob:none --sparse https://github.com/huggingface/optimum", shell=True
        )

        os.chdir("optimum")
        subprocess.run("git sparse-checkout set examples/onnxruntime/quantization/token-classification", shell=True)

        subprocess.run(
            f"python3 examples/onnxruntime/quantization/token-classification/run_ner.py"
            f" --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english"
            f" --dataset_name conll2003"
            f" --do_eval"
            f" --output_dir {dir_path}/tokenclassification_conll2003_optimum"
            f" --max_eval_samples 100"
            f" --opset 11"
            f" --quantization_approach dynamic"
            f" --overwrite_cache True",
            shell=True,
        )

        eval_filename = f"{dir_path}/tokenclassification_conll2003_optimum/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        optimum_accuracy = eval_dict["accuracy"]
        optimum_f1 = eval_dict["f1"]

        run_config = {
            "task": "token-classification",
            "task_args": {"is_regression": False},
            "model_name_or_path": "elastic/distilbert-base-uncased-finetuned-conll03-english",
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
            "batch_sizes": [8],
            "input_lengths": [128],
        }
        run_config = RunConfig(**run_config)
        run_config = dataclasses.asdict(run_config)

        run_instance = OnnxRuntimeRun(run_config)
        run_dict = run_instance.launch()

        optimum_benchmark_accuracy = run_dict["evaluation"]["others"]["optimized"]["accuracy"]
        optimum_benchmark_f1 = run_dict["evaluation"]["others"]["optimized"]["f1"]

        print("Benchmark suite metrics :")
        print(run_dict["evaluation"]["others"]["optimized"])

        print("Optimum examples metrics:")
        print(eval_dict)

        self.assertEqual(optimum_accuracy, optimum_benchmark_accuracy)
        self.assertEqual(optimum_f1, optimum_benchmark_f1)
