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

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx import validate_model_outputs

from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTConfig, ORTOptimizer, ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoQuantizationConfig,
    OptimizationConfig,
    QuantizationConfig,
)
from transformers.onnx.features import FeaturesManager
from parameterized import parameterized


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
    LOAD_CONFIGURATION = {
        "transformers_model": {
            "model_name_or_path": "distilbert-base-uncased-finetuned-sst-2-english",
            "from_transformers": True,
            "task": "text-classification",
        },
        "optimum_model": {
            "model_name_or_path": "optimum/distilbert-base-uncased-finetuned-sst-2-english",
            "task": "text-classification",
        },
        "local_asset": {
            "model_name_or_path": "tests/assets/onnx",
            "task": "text-classification",
        },
        "ort_model_class": {
            "model_name_or_path": ORTModelForSequenceClassification.from_pretrained(
                "optimum/distilbert-base-uncased-finetuned-sst-2-english"
            )
        },
    }

    @parameterized.expand(LOAD_CONFIGURATION.items())
    def test_from_pretrained_method(self, *args):
        _, args = args
        quantizer = ORTQuantizer.from_pretrained(**args)
        self.assertIsInstance(quantizer, ORTQuantizer)

    def test_fail_from_pretrained_method(self):
        with self.assertRaises(Exception) as context:
            ORTQuantizer.from_pretrained("bert-base-cased", from_transformers=True)

        self.assertTrue("When using from_transformers, you need to provide a feature/task.", context.exception)
        with self.assertRaises(Exception) as context:
            ORTQuantizer.from_pretrained("optimum/distilbert-base-uncased-finetuned-sst-2-english")

        self.assertTrue("Unable to load model", context.exception)

    def test_dynamic_quantization(self):
        model_names = {"bert-base-cased", "distilbert-base-uncased"}  # , "facebook/bart-base", "gpt2", "roberta-base"}

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
                    model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
                    model.save_pretrained(tmp_dir)
                    q8_model_path = output_dir.joinpath("model-quantized.onnx")

                    quantizer = ORTQuantizer.from_pretrained(model)
                    quantizer.export(
                        onnx_quantized_model_output_path=q8_model_path,
                        quantization_config=qconfig,
                    )

                    _, onnx_config_factory = FeaturesManager.check_supported_model_or_raise(
                        model, feature=model.pipeline_task
                    )
                    _onnx_config = onnx_config_factory(model.config)
                    validate_model_outputs(
                        _onnx_config,
                        AutoTokenizer.from_pretrained(model_name),
                        AutoModelForSequenceClassification.from_pretrained(model_name),
                        q8_model_path,
                        list(_onnx_config.outputs.keys()),
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
                    model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
                    model.save_pretrained(tmp_dir)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    q8_model_path = output_dir.joinpath("model-quantized.onnx")

                    quantizer = ORTQuantizer.from_pretrained(model)

                    calibration_dataset = quantizer.get_calibration_dataset(
                        "glue",
                        dataset_config_name="sst2",
                        preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                        num_samples=40,
                        dataset_split="train",
                    )
                    calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
                    ranges = quantizer.fit(
                        dataset=calibration_dataset,
                        calibration_config=calibration_config,
                    )
                    quantizer.export(
                        onnx_quantized_model_output_path=q8_model_path,
                        calibration_tensors_range=ranges,
                        quantization_config=qconfig,
                    )

                    _, onnx_config_factory = FeaturesManager.check_supported_model_or_raise(
                        model, feature=model.pipeline_task
                    )
                    _onnx_config = onnx_config_factory(model.config)
                    validate_model_outputs(
                        _onnx_config,
                        tokenizer,
                        AutoModelForSequenceClassification.from_pretrained(model_name),
                        q8_model_path,
                        list(_onnx_config.outputs.keys()),
                        atol=5e-1,
                    )
                    gc.collect()


if __name__ == "__main__":
    unittest.main()
