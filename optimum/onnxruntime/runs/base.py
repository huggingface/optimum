from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.modeling_ort import ORTModel
from optimum.onnxruntime.configuration import AutoCalibrationConfig, ORTConfig, QuantizationConfig
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor

from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType

from optimum.pipelines import pipeline
from transformers import pipeline as transformers_pipeline
from transformers import AutoModel

import torch
from torch import nn

from .benchmark import TimeBenchmark

from optimum import version as optimum_version

from typing import Union


class Run:
    def __init__(self, run_query):
        self.run_query = run_query

        if run_query["framework_args"]["opset"] == "latest":
            self.opset = 17

        if run_query["quantization_approach"] == "static":
            self.apply_static_quantization = True
        else:
            self.apply_static_quantization = False

        # Create the quantization configuration containing all the quantization parameters
        self.qconfig = QuantizationConfig(
            is_static=apply_static_quantization,
            format=QuantFormat.QDQ if apply_static_quantization else QuantFormat.QOperator,
            mode=QuantizationMode.QLinearOps if apply_static_quantization else QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QInt8 if apply_static_quantization else QuantType.QUInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=run_query["per_channel"],
            reduce_range=False,
            operators_to_quantize=run_query["operators_to_quantize"],
        )

        self.quantizer = ORTQuantizer.from_pretrained(
            run_query["model_name_or_path"],
            feature=run_query["task"],
            opset=self.opset,
        )

        self.tokenizer = self.quantizer.tokenizer

        self.warmup_runs = 100
        self.benchmark_duration = 20
        self.batch_size = 8
        self.input_length = 256

        self.model_path = "model.onnx"
        self.quantized_model_path = "quantized_model.onnx"

        self.quantization_preprocessor = QuantizationPreprocessor(self.model_path)

    def launch(self):
        ranges = None
        if self.apply_static_quantization:
            ranges = self.calibrate()

        # Export the quantized model
        self.quantizer.export(
            onnx_model_path=self.model_path,
            onnx_quantized_model_output_path=self.quantized_model_path,
            calibration_tensors_range=ranges,
            quantization_config=self.qconfig,
            preprocessor=self.quantization_preprocessor,  # TODO
        )

        model = ORTModel.load_model(self.quantized_model_path)

        torch_model = AutoModel.load_model(self.run_query["model_name_or_path"])

        optimized_time_metrics = self.evaluate_time(model, input_length=self.input_length, batch_size=self.batch_size)
        raw_time_metrics = self.evaluate_time(torch_model, input_length=self.input_length, batch_size=self.batch_size)
        # TODO clean intermediary savings at the end

        evaluation = {
            "batch_size": self.batch_size,
            "input_length": self.input_length,
            "raw": raw_time_metrics,
            "optimized": optimized_time_metrics,
        }

        body = {
            "model_type": model.config.model_type,
            "task": self.run_query["task"],
            "dataset": self.run_query["dataset"],
            "quantization_approach": self.run_query["dynamic"],
            "operators_to_quantize": self.run_query["operators_to_quantize"],
            "node_exclusion": self.run_query["node_exclusion"],
            "model_size": 42,  #TODO to compute
            "aware_training": self.run_query["aware_training"],
            "per_channel": self.run_query["per_channel"],
            "calibration": self.run_query["calibration"],
            "framework": self.run_query["framework"],
            "framework_args": self.run_query["framework_args"],
            "hardware": "to be defined",  #TODO define hardware
            "versions": {
                "transformers": transformers.__version__,
                "optimum": optimum_version.__version__,
            },
            "evaluation": evaluation
        }

        return res

    def evaluate_time(self, model: Union[ORTModel, nn.Module], input_length, batch_size):

        benchmark = Benchmark()

        input_ids = torch.rand(batch_size, input_length)
        attention_mask = torch.ones(batch_size, input_length)

        # Warmup
        outputs = []
        for _ in trange(self.warmup_runs, desc="Warming up"):
            output = model.forward(input_ids, attention_mask)
            outputs.append(output[0])

        benchmark_duration_ns = config.benchmark_duration * SEC_TO_NS_SCALE
        while sum(benchmark.latencies) < benchmark_duration_ns:
            # TODO do not track the movement from GPU/CPU, numpy/torch
            with benchmark.track():
                model.forward(input_ids, attention_mask)

        benchmark.finalize(benchmark_duration_ns)

        return benchmark.to_dict()

    """
    def preprocessing(self, pipe):
        preprocess_params, _, _ = pipe._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
    """

    def calibrate(self):
        raise NotImplementedError()
        # pipe = transformers_pipeline(model=run_query["model_name_or_path"], tokenizer=self.tokenizer)
        # preprocess_batch = self.preprocessing(pipe)

        # TODO how to handle dataset?
        # Create the calibration dataset used for the calibration step
        calibration_dataset = preprocessed_datasets["train"]

        # Remove the unnecessary columns of the calibration dataset before the calibration step
        calibration_dataset = self.quantizer.clean_calibration_dataset(calibration_dataset)

        # Create the calibration configuration given the selected calibration method
        if self.calibration_method["calibration"]["calibration_method"] == "entropy":
            calibration_config = AutoCalibrationConfig.entropy(calibration_dataset)
        elif self.calibration_method["calibration"]["calibration_method"] == "percentile":
            calibration_config = AutoCalibrationConfig.percentiles(
                calibration_dataset,
                percentile=self.calibration_method["calibration"]["calibration_histogram_percentile"],
            )
        else:
            calibration_config = AutoCalibrationConfig.minmax(
                calibration_dataset,
                self.calibration_method["calibration"]["calibration_moving_average"],
                self.calibration_method["calibration"]["calibration_moving_average_constant"],
            )

        # TODO estimate memory needed for entropy/percentile to autochoose number of shards
        num_calibration_shards = 4
        if not 1 <= num_calibration_shards <= len(calibration_dataset):
            raise ValueError(
                f"Invalid value of number of shards {num_calibration_shards} chosen to split the calibration"
                f" dataset, should be higher than 0 and lower or equal to the number of samples "
                f"{len(calibration_dataset)}."
            )

        for i in range(num_calibration_shards):
            shard = calibration_dataset.shard(num_calibration_shards, i)
            self.quantizer.partial_fit(
                dataset=shard,
                calibration_config=calibration_config,
                onnx_model_path=self.model_path,
                operators_to_quantize=self.qconfig.operators_to_quantize,
                batch_size=8,  # TODO set as arg?
                use_external_data_format=False,
            )
        ranges = self.quantizer.compute_ranges()

        return ranges
