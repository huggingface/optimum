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

from transformers import AutoTokenizer
from transformers.onnx import validate_model_outputs

from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTConfig, ORTOptimizer, ORTQuantizableOperator, ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoQuantizationConfig,
    OptimizationConfig,
    QuantizationConfig,
)


class TestORTConfig(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            optimization_config = OptimizationConfig(optimization_level=2)
            ort_config = ORTConfig(opset=11, quantization=quantization_config, optimization=optimization_config)
            ort_config.save_pretrained(tmp_dir)
            loaded_ort_config = ORTConfig.from_pretrained(tmp_dir)
            self.assertEqual(ort_config.to_dict(), loaded_ort_config.to_dict())


class TestORTOptimizer(unittest.TestCase):
    def test_optimize(self):
        model_names = {
            "bert-base-cased",
            "distilbert-base-uncased",
            "facebook/bart-base",
            "gpt2",
            "roberta-base",
            "google/electra-small-discriminator",
            "microsoft/deberta-base",
            "microsoft/deberta-v2-xlarge",
        }
        optimization_config = OptimizationConfig(optimization_level=99, optimize_with_onnxruntime_only=False)
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    model_path = output_dir.joinpath("model.onnx")
                    optimized_model_path = output_dir.joinpath("model-optimized.onnx")
                    optimizer = ORTOptimizer.from_pretrained(model_name, feature="sequence-classification")
                    optimizer.export(
                        onnx_model_path=model_path,
                        onnx_optimized_model_output_path=optimized_model_path,
                        optimization_config=optimization_config,
                    )
                    validate_model_outputs(
                        optimizer._onnx_config,
                        optimizer.tokenizer,
                        optimizer.model,
                        optimized_model_path,
                        list(optimizer._onnx_config.outputs.keys()),
                        atol=1e-4,
                    )
                    gc.collect()

    def test_optimization_details(self):
        model_name = "bert-base-cased"
        optimization_config = OptimizationConfig(optimization_level=0, optimize_with_onnxruntime_only=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            optimized_model_path = output_dir.joinpath("model-optimized.onnx")
            optimizer = ORTOptimizer.from_pretrained(model_name, feature="sequence-classification")
            optimizer.export(
                onnx_model_path=model_path,
                onnx_optimized_model_output_path=optimized_model_path,
                optimization_config=optimization_config,
            )
            difference_nodes_number = optimizer.get_nodes_number_difference(model_path, optimized_model_path)
            fused_operator = optimizer.get_fused_operators(model_path)
            sorted_operators_difference = optimizer.get_operators_difference(model_path, optimized_model_path)
            self.assertEqual(difference_nodes_number, 0)
            self.assertEqual(len(fused_operator), 0)
            self.assertEqual(len(sorted_operators_difference), 0)
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


if __name__ == "__main__":
    unittest.main()
