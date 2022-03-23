#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import os
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.file_utils import TensorType
from transformers.onnx import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager

from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTModel, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, OptimizationConfig, ORTConfig, QuantizationConfig


class TestORTOptimizer(unittest.TestCase):
    def test_optimize(self):
        model_names = {"bert-base-cased", "distilbert-base-uncased", "facebook/bart-base", "gpt2", "roberta-base"}
        optimization_config = OptimizationConfig(optimization_level=99, optimize_with_onnxruntime_only=False)
        ort_config = ORTConfig(optimization_config=optimization_config)
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    optim_model_path = output_dir.joinpath("model-optimized.onnx")
                    optimizer = ORTOptimizer(ort_config, optimization_config)
                    optimizer.fit(model_name, output_dir, feature="sequence-classification")
                    optimizer.get_optimize_details()
                    validate_model_outputs(
                        optimizer.onnx_config,
                        optimizer.tokenizer,
                        optimizer.model,
                        optim_model_path,
                        list(optimizer.onnx_config.outputs.keys()),
                        atol=1e-4,
                    )
                    gc.collect()


class TestORTQuantizer(unittest.TestCase):
    def test_dynamic_quantization(self):
        model_names = {"bert-base-cased", "distilbert-base-uncased", "facebook/bart-base", "gpt2", "roberta-base"}

        qconfig = QuantizationConfig(
            is_static=False,
            format=QuantFormat.QOperator,
            mode=QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QUInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul"],
        )

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    model_path = output_dir.joinpath("model.onnx")
                    q8_model_path = output_dir.joinpath("model-quantized.onnx")
                    quantizer = ORTQuantizer.from_pretrained(model_name, feature="sequence-classification")
                    quantizer.export(
                        onnx_model_path=model_path,
                        onnx_quantized_model_output_path=q8_model_path,
                        calibration_tensors_range=None,
                        quantization_config=qconfig,
                    )
                    validate_model_outputs(
                        quantizer._onnx_config,
                        quantizer.tokenizer,
                        quantizer.model,
                        q8_model_path,
                        list(quantizer._onnx_config.outputs.keys()),
                        atol=8e-1,
                    )
                    gc.collect()

    def test_static_quantization(self):
        model_names = {"bert-base-cased"}

        qconfig = QuantizationConfig(
            is_static=True,
            format=QuantFormat.QDQ,
            mode=QuantizationMode.QLinearOps,
            activations_dtype=QuantType.QInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul"],
        )

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    model_path = output_dir.joinpath("model.onnx")
                    q8_model_path = output_dir.joinpath("model-quantized.onnx")
                    quantizer = ORTQuantizer.from_pretrained(model_name, feature="sequence-classification")
                    calibration_dataset = quantizer.get_calibration_dataset(
                        "glue",
                        dataset_config_name="sst2",
                        preprocess_function=partial(preprocess_function, tokenizer=quantizer.tokenizer),
                        num_samples=40,
                        dataset_split="train",
                    )
                    calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
                    ranges = quantizer.fit(
                        dataset=calibration_dataset,
                        calibration_config=calibration_config,
                        onnx_model_path=model_path,
                    )
                    quantizer.export(
                        onnx_model_path=model_path,
                        onnx_quantized_model_output_path=q8_model_path,
                        calibration_tensors_range=ranges,
                        quantization_config=qconfig,
                    )
                    validate_model_outputs(
                        quantizer._onnx_config,
                        quantizer.tokenizer,
                        quantizer.model,
                        q8_model_path,
                        list(quantizer._onnx_config.outputs.keys()),
                        atol=5e-1,
                    )
                    gc.collect()


class TestORTModel(unittest.TestCase):
    def test_evaluation(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _, onnx_config_factory = FeaturesManager.check_supported_model_or_raise(
            model, feature="sequence-classification"
        )
        onnx_config = onnx_config_factory(model.config)
        model_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)
        outputs = model(**model_inputs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, model_path)
            ort_model = ORTModel(model_path, onnx_config)
            ort_model_inputs = Dataset.from_dict(model_inputs)
            ort_outputs = ort_model.evaluation_loop(ort_model_inputs)
            self.assertTrue(np.allclose(outputs.logits.detach().numpy(), ort_outputs.predictions, atol=1e-4))
            gc.collect()


if __name__ == "__main__":
    unittest.main()
