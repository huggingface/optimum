import os
import subprocess
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Set

import numpy as np
import torch
import transformers
from datasets import Dataset
from tqdm import trange

import optuna

from . import version as optimum_version
from .utils.preprocessing import (
    ImageClassificationProcessing,
    QuestionAnsweringProcessing,
    TextClassificationProcessing,
    TokenClassificationProcessing,
)
from .utils.runs import RunConfig, cpu_info_command


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_autoclass_name(task):
    if task in ["text-classification", "audio-classification"]:
        autoclass_name = "sequence-classification"
    else:
        autoclass_name = task
    return autoclass_name


class Calibrator:
    def __init__(
        self, calibration_dataset: Dataset, quantizer, model_path, qconfig, calibration_params, node_exclusion
    ):
        self.calibration_dataset = calibration_dataset
        self.quantizer = quantizer
        self.model_path = model_path
        self.qconfig = qconfig
        self.calibration_params = calibration_params
        self.node_exclusion = node_exclusion

    def fit(self):
        raise NotImplementedError()


class Run:
    def __init__(self, run_config: dict):
        """Initialize the Run class holding methods to perform inference and evaluation given a config.

        A run compares a transformers model and an optimized model on latency/throughput, model size, and provided metrics.

        Args:
            run_config (dict): Parameters to use for the run. See [`~utils.runs.RunConfig`] for the expected keys.
        """
        RunConfig(**run_config)  # validate the data (useful if used as standalone)

        self.task = run_config["task"]

        if run_config["quantization_approach"] == "static":
            self.static_quantization = True
        else:
            self.static_quantization = False

        search_space = {"batch_size": run_config["batch_sizes"], "input_length": run_config["input_lengths"]}

        self.study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.GridSampler(search_space),
        )

        cpu_info = subprocess.check_output([cpu_info_command()], shell=True).decode("utf-8")

        optimum_hash = None
        if "dev" in optimum_version.__version__:
            optimum_hash = subprocess.check_output(
                "git ls-remote https://github.com/huggingface/optimum.git HEAD | awk '{ print $1}'", shell=True
            )
            optimum_hash = optimum_hash.decode("utf-8").strip("\n")

        self.return_body = {
            "model_name_or_path": run_config["model_name_or_path"],
            "task": self.task,
            "task_args": run_config["task_args"],
            "dataset": run_config["dataset"],
            "quantization_approach": run_config["quantization_approach"],
            "operators_to_quantize": run_config["operators_to_quantize"],
            "node_exclusion": run_config["node_exclusion"],
            "aware_training": run_config["aware_training"],
            "per_channel": run_config["per_channel"],
            "calibration": run_config["calibration"],
            "framework": run_config["framework"],
            "framework_args": run_config["framework_args"],
            "hardware": cpu_info,  # is this ok?
            "versions": {
                "transformers": transformers.__version__,
                "optimum": optimum_version.__version__,
                "optimum_hash": optimum_hash,
            },
            "evaluation": {
                "time": [],
                "others": {"baseline": {}, "optimized": {}},
            },
            "max_eval_samples": run_config["max_eval_samples"],
            "time_benchmark_args": run_config["time_benchmark_args"],
        }

    def launch(self):
        """Launch inference to compare metrics between the original and optimized model.

        These metrics are latency, throughput, model size, and user provided metrics.

        Returns:
            `dict`: Finalized run data with metrics stored in the "evaluation" key.
        """
        try:
            self.study.optimize(self._launch_time)
            self.launch_eval()
        finally:
            self.finalize()
            print("Finished run.")

        return self.return_body

    def _launch_time(self, trial):
        """Optuna objective function to measure latency/throughput.

        Populate the `["evaluation"]["time"]` list of the run for various batch size and input length.

        Returns:
            Dummy data.
        """
        raise NotImplementedError()

    def launch_eval(self):
        """
        Run evaluation on the original and optimized model.

        Populate the `["evaluation"]["others"]` subdictionary of the run.
        """
        raise NotImplementedError()

    def load_datasets(self):
        """Load evaluation dataset, and if needed, calibration dataset for static quantization."""
        datasets_dict = self.task_processor.load_datasets()

        self._eval_dataset = datasets_dict["eval"]
        if self.static_quantization:
            self._calibration_dataset = datasets_dict["calibration"]

    def get_calibration_dataset(self):
        """Get calibration dataset. The dataset needs to be loaded first with [`~optimum.runs_base.Run.load_datasets`].

        Returns:
            `datasets.Dataset`: Calibration dataset.
        """
        if not hasattr(self, "_calibration_dataset"):
            raise KeyError("No calibration dataset defined for this run.")
        return self._calibration_dataset

    def get_eval_dataset(self):
        """
        Get evaluation dataset.  The dataset needs to be loaded first with [`~optimum.runs_base.Run.load_datasets`].

        Returns:
            `datasets.Dataset`: Evaluation dataset.
        """
        if not hasattr(self, "_eval_dataset"):
            raise KeyError("No evaluation dataset defined for this run.")
        return self._eval_dataset

    def finalize(self):
        """Cleanup intermediary files."""
        raise NotImplementedError()


SEC_TO_NS_SCALE = 1000000000
NS_TO_MS_SCALE = 1e6


def ns_to_ms(ns_time):
    return ns_time / NS_TO_MS_SCALE


class TimeBenchmark:
    def __init__(
        self, model, batch_size: int, input_length: int, model_input_names: Set[str], warmup_runs: int, duration: float
    ):
        self.batch_size = batch_size
        self.input_length = input_length
        self.model = model

        # in seconds
        self.warmup_runs = warmup_runs
        self.benchmark_duration = duration

        self.latencies = []
        self.throughput = float("-inf")

        self.model_input_names = model_input_names

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @contextmanager
    def track(self):
        start = perf_counter_ns()
        yield
        end = perf_counter_ns()

        # Append the time to the buffer
        self.latencies.append(end - start)

        print(f"Tracked function took: {(end - start)}ns ({(end - start) / 1e6:.3f}ms)")

    def finalize(self, duration_ns: int):
        self.throughput = round((len(self.latencies) / duration_ns) * SEC_TO_NS_SCALE, 2)

    def to_dict(self):
        # Compute stats, beware latencies are stored as ms
        benchmarks_stats = {
            "nb_forwards": len(self.latencies),
            "throughput": self.throughput,
            "latency_mean": ns_to_ms(np.mean(self.latencies)),
            "latency_std": ns_to_ms(np.std(self.latencies)),
            "latency_50": ns_to_ms(np.quantile(self.latencies, 0.5)),
            "latency_90": ns_to_ms(np.quantile(self.latencies, 0.9)),
            "latency_95": ns_to_ms(np.quantile(self.latencies, 0.95)),
            "latency_99": ns_to_ms(np.quantile(self.latencies, 0.99)),
            "latency_999": ns_to_ms(np.quantile(self.latencies, 0.999)),
        }

        return benchmarks_stats

    def execute(self):
        inputs = {}

        checked_inputs = {"input_ids", "attention_mask", "token_type_ids", "pixel_values"}
        if "input_ids" in self.model_input_names:
            inputs["input_ids"] = torch.randint(high=1000, size=(self.batch_size, self.input_length))
        if "attention_mask" in self.model_input_names:
            inputs["attention_mask"] = torch.ones(self.batch_size, self.input_length, dtype=torch.int64)
        if "token_type_ids" in self.model_input_names:
            inputs["token_type_ids"] = torch.ones(self.batch_size, self.input_length, dtype=torch.int64)
        if "pixel_values" in self.model_input_names:
            # TODO support grayscale?
            inputs["pixel_values"] = torch.rand(
                self.batch_size, 3, self.model.config.image_size, self.model.config.image_size, dtype=torch.float32
            )

        if np.any([k not in checked_inputs for k in self.model_input_names]):
            raise NotImplementedError(
                f"At least an input in {self.model_input_names} has no dummy generation for time benchmark."
            )

        # Warmup
        for _ in trange(self.warmup_runs, desc="Warming up"):
            self.model.forward(**inputs)

        if self.benchmark_duration != 0:
            benchmark_duration_ns = self.benchmark_duration * SEC_TO_NS_SCALE
            print(f"Running time tracking in {self.benchmark_duration:.1f}s.")
            while sum(self.latencies) < benchmark_duration_ns:
                # TODO not trak GPU/CPU <--> numpy/torch, need to change the implementation of forward
                with self.track():
                    self.model.forward(**inputs)

            self.finalize(benchmark_duration_ns)

            return self.to_dict()
        else:
            benchmarks_stats = {
                "nb_forwards": 0,
                "throughput": -1,
                "latency_mean": -1,
            }
            return benchmarks_stats


task_processing_map = {
    "text-classification": TextClassificationProcessing,
    "token-classification": TokenClassificationProcessing,
    "question-answering": QuestionAnsweringProcessing,
    "image-classification": ImageClassificationProcessing,
}
