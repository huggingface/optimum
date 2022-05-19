import copy
import os

from datasets import load_metric
from transformers import pipeline as _transformers_pipeline
from transformers.onnx import FeaturesManager

from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.pipelines import pipeline as _optimum_pipeline
from optimum.runs_base import Run, TimeBenchmark, get_autoclass_name, task_processing_map

from .. import ORTQuantizer
from ..configuration import QuantizationConfig
from ..modeling_ort import ORTModel
from ..preprocessors import QuantizationPreprocessor
from .calibrator import OnnxRuntimeCalibrator
from .utils import task_ortmodel_map


class OnnxRuntimeRun(Run):
    def __init__(self, run_config):
        super().__init__(run_config)

        # Create the quantization configuration containing all the quantization parameters
        qconfig = QuantizationConfig(
            is_static=self.static_quantization,
            format=QuantFormat.QDQ if self.static_quantization else QuantFormat.QOperator,
            mode=QuantizationMode.QLinearOps if self.static_quantization else QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QInt8 if self.static_quantization else QuantType.QUInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=run_config["per_channel"],
            reduce_range=False,
            operators_to_quantize=run_config["operators_to_quantize"],
        )

        quantizer = ORTQuantizer.from_pretrained(
            run_config["model_name_or_path"],
            feature=get_autoclass_name(self.task),
            opset=run_config["framework_args"]["opset"],
        )

        self.tokenizer = copy.deepcopy(quantizer.tokenizer)

        self.batch_sizes = run_config["batch_sizes"]
        self.input_lengths = run_config["input_lengths"]

        self.model_path = "model.onnx"
        self.quantized_model_path = "quantized_model.onnx"

        processing_class = task_processing_map[self.task]
        self.processor = processing_class(
            dataset_path=run_config["dataset"]["path"],
            dataset_name=run_config["dataset"]["name"],
            calibration_split=run_config["dataset"]["calibration_split"],
            eval_split=run_config["dataset"]["eval_split"],
            tokenizer=self.tokenizer,
            max_seq_length=run_config["dataset"]["max_seq_length"],  # not needed for some tasks?
            data_keys=run_config["dataset"]["data_keys"],
            ref_keys=run_config["dataset"]["ref_keys"],
            static_quantization=self.static_quantization,
            num_calibration_samples=run_config["calibration"]["num_calibration_samples"]
            if self.static_quantization
            else None,
            config=quantizer.model.config,
        )

        self.metric_names = run_config["metrics"]

        self.load_datasets()

        # quantization_preprocessor = QuantizationPreprocessor()
        quantization_preprocessor = None
        ranges = None
        if self.static_quantization:
            calibration_dataset = self.get_calibration_dataset()
            calibrator = OnnxRuntimeCalibrator(
                calibration_dataset, quantizer, self.model_path, qconfig, calibration_params=run_config["calibration"]
            )
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
        self.torch_model = model_class.from_pretrained(run_config["model_name_or_path"])

        self.return_body[
            "model_type"
        ] = self.torch_model.config.model_type  # return_body is initialized in parent class

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
        baseline_time_metrics = torch_benchmark.execute()

        time_evaluation = {
            "batch_size": batch_size,
            "input_length": input_length,
            "baseline": baseline_time_metrics,
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
        all_labels, all_preds_baseline = self.processor.run_inference(eval_dataset, transformers_pipeline)
        _, all_preds_optimized = self.processor.run_inference(eval_dataset, ort_pipeline)

        baseline_metrics_dict = {}
        optimized_metrics_dict = {}

        for metric_name in self.metric_names:
            metric = load_metric(metric_name)
            baseline_metrics_dict.update(
                self.processor.get_metrics(predictions=all_preds_baseline, references=all_labels, metric=metric)
            )
            optimized_metrics_dict.update(
                self.processor.get_metrics(predictions=all_preds_optimized, references=all_labels, metric=metric)
            )

        self.return_body["evaluation"]["others"]["baseline"].update(baseline_metrics_dict)
        self.return_body["evaluation"]["others"]["optimized"].update(optimized_metrics_dict)

    def finalize(self):
        if os.path.isfile(self.quantized_model_path):
            os.remove(self.quantized_model_path)
        if os.path.isfile(self.model_path):
            os.remove(self.model_path)
