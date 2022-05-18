import os
from contextlib import contextmanager
from time import perf_counter_ns

import numpy as np
import torch
from datasets import Dataset
from tqdm import trange

import optuna

from .utils.preprocessing import (
    QuestionAnsweringProcessing,
    TextClassificationProcessing,
    TokenClassificationProcessing,
)
from .utils.runs import RunConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_autoclass_name(task):
    if task in ["text-classification", "audio-classification"]:
        autoclass_name = "sequence-classification"
    else:
        autoclass_name = task
    return autoclass_name


class Calibrator:
    def __init__(self, calibration_dataset: Dataset, quantizer, model_path, qconfig, calibration_params):
        self.calibration_dataset = calibration_dataset
        self.quantizer = quantizer
        self.model_path = model_path
        self.qconfig = qconfig
        self.calibration_params = calibration_params

    def calibrate(self):
        raise NotImplementedError()


class Run:
    def __init__(self, run_config: dict):
        """Initialize the Run class holding methods to perform inference and evaluation given a query.

        A run compares a transformers model and an optimized model on latency/throughput, model size, and provided metrics.

        Args:
            run_config (dict): Parameters to use for the run. TODO: See BaseModel doc for the expected arguments.
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

    def launch(self):
        """Launch inference to compare metrics between the original and optimized model.

        These metrics are latency, throughput, model size, and user provided metrics.

        Returns:
            dict: Finalized run data with metrics stored in the "evaluation" key
        """
        try:
            self.study.optimize(self._launch_time, n_trials=100, timeout=600)
            self.launch_eval()
        finally:
            self.finalize()

        return self.return_body

    def _launch_time(self, trial):
        """Optuna objective function to measure latency/throughput.

        Populate the ["evaluation"]["time"] list of the run for various batch size and input length.

        Returns:
            Dummy data.
        """
        raise NotImplementedError()

    def launch_eval(self):
        """
        Run evaluation on the original and optimized model.

        Populate the ["evaluation"]["others"] subdictionary of the run.
        """
        raise NotImplementedError()

    def load_datasets(self):
        """Load evaluation dataset, and if needed, calibration dataset."""
        raise NotImplementedError()

    def get_calibration_dataset(self):
        """Failsafe get calibration dataset."""
        raise NotImplementedError()

    def get_eval_dataset(self):
        """Failsafe get evaluation dataset."""
        raise NotImplementedError()

    def finalize(self):
        """Cleanup intermediary files."""
        raise NotImplementedError()


SEC_TO_NS_SCALE = 1000000000
NS_TO_MS_SCALE = 1e6


def ns_to_ms(ns_time):
    return ns_time / NS_TO_MS_SCALE


class TimeBenchmark:
    def __init__(self, model, batch_size: int, input_length: int, has_token_type_ids: bool):
        self.batch_size = batch_size
        self.input_length = input_length
        self.has_token_type_ids = has_token_type_ids
        self.model = model

        # TODO parametrize
        self.warmup_runs = 2
        self.benchmark_duration = 2

        self.latencies = []
        self.throughput = float("-inf")

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
        inputs = {
            "input_ids": torch.randint(high=1000, size=(self.batch_size, self.input_length)),
            "attention_mask": torch.ones(self.batch_size, self.input_length, dtype=torch.int64),
        }
        if self.has_token_type_ids:
            inputs["token_type_ids"] = torch.ones(self.batch_size, self.input_length, dtype=torch.int64)

        # Warmup
        outputs = []
        for _ in trange(self.warmup_runs, desc="Warming up"):
            output = self.model.forward(**inputs)
            outputs.append(output[0])

        benchmark_duration_ns = self.benchmark_duration * SEC_TO_NS_SCALE
        while sum(self.latencies) < benchmark_duration_ns:
            # TODO not trak GPU/CPU <--> numpy/torch, need to change the implementation of forward
            with self.track():
                self.model.forward(**inputs)

        self.finalize(benchmark_duration_ns)

        return self.to_dict()


task_processing_map = {
    "text-classification": TextClassificationProcessing,
    "token-classification": TokenClassificationProcessing,
    "question-answering": QuestionAnsweringProcessing,
}
