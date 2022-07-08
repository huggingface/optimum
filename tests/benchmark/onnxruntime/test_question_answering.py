import dataclasses
import json
import os
import shutil
import subprocess
import unittest
from datetime import datetime

from optimum.onnxruntime.runs import OnnxRuntimeRun
from optimum.utils.runs import RunConfig


class TestQuestionAnswering(unittest.TestCase):
    def test_eval_transformers(self):
        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_questionanswering"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 3 --filter=blob:none --sparse https://github.com/huggingface/transformers", shell=True
        )

        os.chdir("transformers")
        subprocess.run("git sparse-checkout set examples/pytorch/question-answering", shell=True)

        subprocess.run(
            f"python3 examples/pytorch/question-answering/run_qa.py"
            f" --model_name_or_path distilbert-base-uncased-distilled-squad"
            f" --dataset_name squad"
            f" --do_eval"
            f" --output_dir {dir_path}/questionanswering_squad_transformers"
            f" --max_eval_samples 100"
            f" --max_seq_length 384",
            shell=True,
        )

        eval_filename = f"{dir_path}/questionanswering_squad_transformers/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        transformers_f1 = eval_dict["eval_f1"]

        run_config = {
            "task": "question-answering",
            "task_args": {"is_regression": False},
            "model_name_or_path": "distilbert-base-uncased-distilled-squad",
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
            "batch_sizes": [8],
            "input_lengths": [128],
            "max_eval_samples": 100,
            "time_benchmark_args": {"warmup_runs": 0, "duration": 0},
        }
        run_config = RunConfig(**run_config)
        run_config = dataclasses.asdict(run_config)

        run_instance = OnnxRuntimeRun(run_config)
        run_dict = run_instance.launch()

        optimum_benchmark_f1 = run_dict["evaluation"]["others"]["baseline"]["f1"]

        print("Benchmark suite metrics :")
        print(run_dict["evaluation"]["others"]["baseline"])

        print("Transformers examples metrics:")
        print(eval_dict)

        self.assertEqual(transformers_f1, optimum_benchmark_f1)

    def test_eval_optimum(self):
        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_questionanswering"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 4 --filter=blob:none --sparse https://github.com/huggingface/optimum", shell=True
        )

        os.chdir("optimum")
        subprocess.run("git sparse-checkout set examples/onnxruntime/quantization/question-answering", shell=True)

        subprocess.run(
            f"python3 examples/onnxruntime/quantization/question-answering/run_qa.py"
            f" --model_name_or_path distilbert-base-uncased-distilled-squad"
            f" --dataset_name squad"
            f" --do_eval"
            f" --output_dir {dir_path}/questionanswering_squad_optimum"
            f" --max_eval_samples 100"
            f" --opset 11"
            f" --quantization_approach dynamic"
            f" --max_seq_length 384",
            shell=True,
        )

        eval_filename = f"{dir_path}/questionanswering_squad_optimum/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        optimum_f1 = eval_dict["f1"]

        print("dir_path", dir_path)

        print("Optimum examples metrics:")
        print(eval_dict)

        run_config = {
            "task": "question-answering",
            "task_args": {"is_regression": False},
            "model_name_or_path": "distilbert-base-uncased-distilled-squad",
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
            "framework_args": {"optimization_level": 1, "opset": 11},
            "batch_sizes": [8],
            "input_lengths": [128],
            "max_eval_samples": 100,
            "time_benchmark_args": {"warmup_runs": 0, "duration": 0},
        }
        run_config = RunConfig(**run_config)
        run_config = dataclasses.asdict(run_config)

        run_instance = OnnxRuntimeRun(run_config)
        run_dict = run_instance.launch()

        optimum_benchmark_f1 = run_dict["evaluation"]["others"]["optimized"]["f1"]

        print("Benchmark suite metrics :")
        print(run_dict["evaluation"]["others"]["optimized"])

        print("Optimum examples metrics:")
        print(eval_dict)

        self.assertEqual(optimum_f1, optimum_benchmark_f1)
