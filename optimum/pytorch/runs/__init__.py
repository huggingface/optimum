import json
import os
from typing import Optional, Union

import torch
from transformers import pipeline as _transformers_pipeline
from transformers.onnx import FeaturesManager

from ...runs_base import Run, TimeBenchmark, get_autoclass_name, task_processing_map


class PyTorchRun(Run):
    def __init__(self, run_config):
        run_config = super().__init__(run_config)
        self.run_config = run_config

        self.time_benchmark_args = run_config["time_benchmark_args"]

        # pytorch benchmark
        model_class = FeaturesManager.get_model_class_for_feature(get_autoclass_name(self.task))
        self.torch_model = model_class.from_pretrained(run_config["model_name_or_path"])

        processing_class = task_processing_map[self.task]
        self.task_processor = processing_class(
            dataset_path=run_config["dataset"]["path"],
            dataset_name=run_config["dataset"]["name"],
            calibration_split=run_config["dataset"]["calibration_split"],
            eval_split=run_config["dataset"]["eval_split"],
            preprocessor=self.preprocessor,
            data_keys=run_config["dataset"]["data_keys"],
            ref_keys=run_config["dataset"]["ref_keys"],
            task_args=run_config["task_args"],
            static_quantization=self.static_quantization,
            num_calibration_samples=run_config["calibration"]["num_calibration_samples"]
            if self.static_quantization
            else None,
            config=self.torch_model.config,
            max_eval_samples=run_config["max_eval_samples"],
        )

        self.metric_names = run_config["metrics"]

        self.load_datasets()

        self.return_body[
            "model_type"
        ] = self.torch_model.config.model_type  # return_body is initialized in parent class

    def _launch_time(self, trial):
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        input_length = trial.suggest_categorical("input_length", self.input_lengths)

        model_input_names = set(self.preprocessor.model_input_names)

        print("Running PyTorch time benchmark.")
        time_benchmark = TimeBenchmark(
            self.torch_model,
            input_length=input_length,
            batch_size=batch_size,
            model_input_names=model_input_names,
            warmup_runs=self.time_benchmark_args["warmup_runs"],
            duration=self.time_benchmark_args["duration"],
        )
        with torch.no_grad():
            time_metrics = time_benchmark.execute()

        time_evaluation = {
            "batch_size": batch_size,
            "input_length": input_length,
        }
        time_evaluation.update(time_metrics)

        self.return_body["evaluation"]["time"].append(time_evaluation)

        return 0, 0

    def launch_eval(
        self, save: bool = False, save_directory: Union[str, os.PathLike] = None, run_name: Optional[str] = None
    ):
        try:
            kwargs = self.task_processor.get_pipeline_kwargs()

            # transformers pipelines are smart enought to detect whether the tokenizer or feature_extractor is needed
            pipeline = _transformers_pipeline(
                task=self.task,
                model=self.torch_model,
                tokenizer=self.preprocessor,
                feature_extractor=self.preprocessor,
                **kwargs,
            )

            eval_dataset = self.get_eval_dataset()

            print("Running evaluation...")
            metrics_dict = self.task_processor.run_evaluation(eval_dataset, pipeline, self.metric_names)

            self.return_body["evaluation"]["others"].update(metrics_dict)

            if save:
                self.save(save_directory, run_name)
        finally:
            self.finalize()

        return self.return_body

    def save(self, save_directory: Union[str, os.PathLike], run_name: str):
        save_directory = super().save(save_directory, run_name)

        # save run config and evaluation results
        with open(os.path.join(save_directory, "results.json"), "w") as f:
            json.dump(self.return_body, f, indent=4)
