import dataclasses
import os
import tempfile
import unittest

from optimum.onnxruntime.runs import OnnxRuntimeRun
from optimum.utils.runs import RunConfig


class TestRunBase(unittest.TestCase):
    def test_save(self):
        run_config = {
            "model_name_or_path": "fxmarty/resnet-tiny-beans",
            "dataset": {
                "path": "beans",
                "eval_split": "validation",
                "data_keys": {"primary": "image"},
                "ref_keys": ["labels"],
            },
            "metrics": ["accuracy"],
            "task": "image-classification",
            "quantization_approach": "dynamic",
            "framework": "onnxruntime",
            "framework_args": {"opset": 12},
            "operators_to_quantize": ["Add", "MatMul"],
            "time_benchmark_args": {"duration": 0, "warmup_runs": 0},
            "max_eval_samples": 100,
        }

        cfg = RunConfig(**run_config)
        cfg = dataclasses.asdict(cfg)

        eval_run = OnnxRuntimeRun(cfg)

        save_directory = tempfile.mkdtemp()

        subdirs = set(next(os.walk(save_directory))[1])  # get non-recursive subdirectories
        _ = eval_run.launch(save=True, save_directory=save_directory)
        subdirs_after = set(next(os.walk(save_directory))[1])

        self.assertEqual(len(subdirs_after - subdirs), 1)
        result_dir = list(subdirs_after - subdirs)[0]

        result_path = os.path.join(save_directory, result_dir)

        self.assertTrue(os.path.isfile(os.path.join(result_path, "quantized_model.onnx")))
        self.assertTrue(os.path.isfile(os.path.join(result_path, "results.json")))
        self.assertTrue(os.path.isfile(os.path.join(result_path, "ort_config.json")))
