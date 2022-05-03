import copy
import os
import subprocess

import transformers
from datasets import load_metric
from transformers import pipeline as _transformers_pipeline
from transformers.onnx import FeaturesManager

from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum import version as optimum_version
from optimum.pipelines import pipeline as _optimum_pipeline
from optimum.runs_base import Run, TimeBenchmark, get_autoclass_name, task_processing_map

from .. import ORTQuantizer
from ..configuration import QuantizationConfig
from ..modeling_ort import ORTModel
from ..preprocessors import QuantizationPreprocessor
from .calibrator import OnnxRuntimeCalibrator
from .utils import task_ortmodel_map


class OnnxRuntimeRun(Run):
    def __init__(self, run_query):
        super().__init__(run_query)

        # Create the quantization configuration containing all the quantization parameters
        qconfig = QuantizationConfig(
            is_static=self.static_quantization,
            format=QuantFormat.QDQ if self.static_quantization else QuantFormat.QOperator,
            mode=QuantizationMode.QLinearOps if self.static_quantization else QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QInt8 if self.static_quantization else QuantType.QUInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=run_query["per_channel"],
            reduce_range=False,
            operators_to_quantize=run_query["operators_to_quantize"],
        )

        quantizer = ORTQuantizer.from_pretrained(
            run_query["model_name_or_path"],
            feature=get_autoclass_name(self.task),
            opset=run_query["framework_args"]["opset"],
        )

        self.tokenizer = copy.deepcopy(quantizer.tokenizer)

        self.batch_sizes = run_query["batch_sizes"]
        self.input_lengths = run_query["input_lengths"]

        self.model_path = "model.onnx"
        self.quantized_model_path = "quantized_model.onnx"

        processing_class = task_processing_map[self.task]
        self.processor = processing_class(
            dataset_path=run_query["dataset"]["path"],
            dataset_name=run_query["dataset"]["name"],
            calibration_split=run_query["dataset"]["calibration_split"],
            eval_split=run_query["dataset"]["eval_split"],
            tokenizer=self.tokenizer,
            max_seq_length=run_query["dataset"]["max_seq_length"],  # not needed for some tasks?
            data_keys=run_query["dataset"]["data_keys"],
            ref_keys=run_query["dataset"]["ref_keys"],
            static_quantization=self.static_quantization,
            num_calibration_samples=run_query["calibration"]["num_calibration_samples"]
            if self.static_quantization
            else None,
            config=quantizer.model.config,
        )

        self.metric_names = run_query["metrics"]

        self.load_datasets()

        # quantization_preprocessor = QuantizationPreprocessor()
        quantization_preprocessor = None
        ranges = None
        if self.static_quantization:
            calibration_dataset = self.get_calibration_dataset()
            calibrator = OnnxRuntimeCalibrator(calibration_dataset, quantizer, self.model_path, qconfig)
            ranges = calibrator.calibrate()

        # Export the quantized model
        quantizer.export(
            onnx_model_path=self.model_path,
            onnx_quantized_model_output_path=self.quantized_model_path,
            calibration_tensors_range=ranges,
            quantization_config=qconfig,
            preprocessor=quantization_preprocessor,  # TODO
        )

        # onnxruntime benchmark
        ort_session = ORTModel.load_model(self.quantized_model_path)

        # necessary to pass the config for the pipeline not to complain later
        self.ort_model = task_ortmodel_map[self.task](ort_session, config=quantizer.model.config)

        # pytorch benchmark
        model_class = FeaturesManager.get_model_class_for_feature(get_autoclass_name(self.task))
        self.torch_model = model_class.from_pretrained(run_query["model_name_or_path"])

        cpu_info = subprocess.check_output(["lscpu"]).decode("utf-8")

        self.return_body = {
            "model_name_or_path": run_query["model_name_or_path"],
            "model_type": self.torch_model.config.model_type,
            "task": self.task,
            "dataset": run_query["dataset"],
            "quantization_approach": run_query["quantization_approach"],
            "operators_to_quantize": run_query["operators_to_quantize"],
            "node_exclusion": run_query["node_exclusion"],
            "aware_training": run_query["aware_training"],
            "per_channel": run_query["per_channel"],
            "calibration": run_query["calibration"],
            "framework": run_query["framework"],
            "framework_args": run_query["framework_args"],
            "hardware": cpu_info,  # is this ok?
            "versions": {
                "transformers": transformers.__version__,
                "optimum": optimum_version.__version__,
            },
            "evaluation": {
                "time": [],
                "others": {"raw": {}, "optimized": {}},
            },
            "metrics": run_query["metrics"],
        }

    def _launch_time(self, trial):
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        input_length = trial.suggest_categorical("input_length", self.input_lengths)

        has_token_type_ids = "token_type_ids" in self.tokenizer.model_input_names

        # onnxruntime benchmark
        ort_benchmark = TimeBenchmark(self.ort_model, input_length, batch_size, has_token_type_ids=has_token_type_ids)
        optimized_time_metrics = ort_benchmark.execute()

        # pytorch benchmark
        torch_benchmark = TimeBenchmark(
            self.torch_model, input_length, batch_size, has_token_type_ids=has_token_type_ids
        )
        raw_time_metrics = torch_benchmark.execute()

        time_evaluation = {
            "batch_size": batch_size,
            "input_length": input_length,
            "raw": raw_time_metrics,
            "optimized": optimized_time_metrics,
        }

        self.return_body["evaluation"]["time"].append(time_evaluation)

        return 0, 0

    def launch_eval(self):
        kwargs = self.processor.get_pipeline_kwargs()

        ort_pipeline = _optimum_pipeline(
            task=self.task,
            model=self.ort_model,
            tokenizer=self.tokenizer,
            feature_extractor=None,
            accelerator="ort",
            **kwargs,
        )

        transformers_pipeline = _transformers_pipeline(
            task=self.task,
            model=self.torch_model,
            tokenizer=self.tokenizer,
            feature_extractor=None,
            **kwargs,
        )

        eval_dataset = self.get_eval_dataset()

        # may be better to avoid to get labels twice
        all_labels, all_preds_raw = self.processor.run_inference(eval_dataset, transformers_pipeline)
        _, all_preds_optimized = self.processor.run_inference(eval_dataset, ort_pipeline)

        raw_metrics_dict = {}
        optimized_metrics_dict = {}

        for metric_name in self.metric_names:
            metric = load_metric(metric_name)
            raw_metrics_dict.update(
                self.processor.get_metrics(predictions=all_preds_raw, references=all_labels, metric=metric)
            )
            optimized_metrics_dict.update(
                self.processor.get_metrics(predictions=all_preds_optimized, references=all_labels, metric=metric)
            )

        self.return_body["evaluation"]["others"]["raw"].update(raw_metrics_dict)
        self.return_body["evaluation"]["others"]["optimized"].update(optimized_metrics_dict)

    def load_datasets(self):
        datasets_dict = self.processor.load_datasets()

        self._eval_dataset = datasets_dict["eval"]
        if self.static_quantization:
            self._calibration_dataset = datasets_dict["calibration"]

    def get_calibration_dataset(self):
        if not hasattr(self, "_calibration_dataset"):
            raise KeyError("No calibration dataset defined for this run.")
        return self._calibration_dataset

    def get_eval_dataset(self):
        if not hasattr(self, "_eval_dataset"):
            raise KeyError("No evaluation dataset defined for this run.")
        return self._eval_dataset

    def finalize(self):
        if os.path.isfile(self.quantized_model_path):
            os.remove(self.quantized_model_path)
        if os.path.isfile(self.model_path):
            os.remove(self.model_path)
