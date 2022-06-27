import dataclasses
import json
import os
import subprocess
import unittest
from datetime import datetime

from optimum.onnxruntime.runs import OnnxRuntimeRun
from optimum.utils.runs import RunConfig


class TestImageClassification(unittest.TestCase):
    def test_eval_transformers_examples(self):
        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_imageclassification"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 3 --filter=blob:none --sparse https://github.com/huggingface/transformers", shell=True
        )

        os.chdir("transformers")
        subprocess.run("git sparse-checkout set examples/pytorch/image-classification", shell=True)

        subprocess.run(
            f"python3 examples/pytorch/image-classification/run_image_classification.py"
            #f"python3 /tmp/2022-06-27T11:20:17.689617_imageclassification/transformers/examples/pytorch/image-classification/run_image_classification.py"
            f" --model_name_or_path nateraw/vit-base-beans"
            f" --dataset_name beans"
            f" --do_eval"
            f" --remove_unused_columns False"
            f" --seed 42"
            f" --output_dir {dir_path}/imageclassification_beans_transformers"
            f" --max_eval_samples 100",
            shell=True,
        )

        eval_filename = f"{dir_path}/imageclassification_beans_transformers/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        transformers_accuracy = eval_dict["eval_accuracy"]

        run_config = {
            "task": "image-classification",
            "model_name_or_path": "nateraw/vit-base-beans",
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
