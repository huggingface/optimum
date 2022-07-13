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
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx.utils import get_preprocessor

import onnx
from onnx import load as onnx_load
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTConfig, ORTModelForSequenceClassification, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoQuantizationConfig,
    OptimizationConfig,
    QuantizationConfig,
)
from parameterized import parameterized


class ORTConfigTest(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            optimization_config = OptimizationConfig(optimization_level=2)
            ort_config = ORTConfig(opset=11, quantization=quantization_config, optimization=optimization_config)
            ort_config.save_pretrained(tmp_dir)
            loaded_ort_config = ORTConfig.from_pretrained(tmp_dir)
            self.assertEqual(ort_config.to_dict(), loaded_ort_config.to_dict())


class ORTOptimizerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "bert": "bert-base-cased",
        "distilbert": "distilbert-base-uncased",
        "bart": "facebook/bart-base",
        "gpt2": "gpt2",
        "roberta": "roberta-base",
        "electra": "google/electra-small-discriminator",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_optimize(self, *args, **kwargs):
        model_type, model_name = args
        optimization_config = OptimizationConfig(optimization_level=2, optimize_with_onnxruntime_only=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(tmp_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.fit(optimization_config=optimization_config, save_dir=tmp_dir)
            optimized_model = ORTModelForSequenceClassification.from_pretrained(
                tmp_dir, file_name="model_optimized.onnx", from_transformers=False
            )
            tokenizer = get_preprocessor(model_name)
            tokens = tokenizer("This is a sample output", return_tensors="pt")
            original_model_outputs = model(**tokens)
            optimized_model_outputs = optimized_model(**tokens)
            self.assertTrue(torch.allclose(original_model_outputs.logits, optimized_model_outputs.logits, atol=1e-4))
            gc.collect()

    def test_optimization_details(self):
        model_name = "bert-base-cased"
        optimization_config = OptimizationConfig(optimization_level=0, optimize_with_onnxruntime_only=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            optimized_model_path = output_dir.joinpath("model_optimized.onnx")
            model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(output_dir)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.fit(optimization_config=optimization_config, save_dir=output_dir)
            difference_nodes_number = optimizer.get_nodes_number_difference(model_path, optimized_model_path)
            fused_operator = optimizer.get_fused_operators(model_path)
            sorted_operators_difference = optimizer.get_operators_difference(model_path, optimized_model_path)
            self.assertEqual(difference_nodes_number, 0)
            self.assertEqual(len(fused_operator), 0)
            self.assertEqual(len(sorted_operators_difference), 0)
            gc.collect()

    def test_optimization_fp16(self):
        model_name = "hf-internal-testing/tiny-random-distilbert"
        optimization_config = OptimizationConfig(optimization_level=0, fp16=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            optimized_model_path = output_dir.joinpath("model_optimized.onnx")
            onnx_model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            onnx_model.save_pretrained(output_dir.as_posix())
            optimizer = ORTOptimizer.from_pretrained(onnx_model)
            optimizer.fit(optimization_config=optimization_config, save_dir=output_dir)
            model = onnx.load(optimized_model_path.as_posix())
            for w in model.graph.initializer:
                self.assertNotEqual(w.data_type, onnx.onnx_pb.TensorProto.FLOAT)

            onnx_model = ORTModelForSequenceClassification.from_pretrained(
                output_dir.as_posix(), file_name="model_optimized.onnx"
            )
            transformers_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer("This is a sample output", return_tensors="pt")
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            onnx_outputs = onnx_model(**tokens)

            # compare tensor outputs
            self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))


class ORTQuantizerTest(unittest.TestCase):
    LOAD_CONFIGURATION = {
        "transformers_model": {
            "model_name_or_path": "distilbert-base-uncased-finetuned-sst-2-english",
            "from_transformers": True,
            "feature": "sequence-classification",
        },
        "optimum_model": {
            "model_name_or_path": "optimum/distilbert-base-uncased-finetuned-sst-2-english",
            "feature": "sequence-classification",
        },
        "local_asset": {
            "model_name_or_path": "tests/assets/onnx",
            "feature": "sequence-classification",
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
        self.assertIn("When using from_transformers, you need to provide a feature.", str(context.exception))

        with self.assertRaises(Exception) as context:
            ORTQuantizer.from_pretrained("optimum/distilbert-base-uncased-finetuned-sst-2-english")
        self.assertIn("Unable to load model", str(context.exception))

        with self.assertRaises(Exception) as context:
            ORTQuantizer.from_pretrained(
                "optimum/distilbert-base-uncased-finetuned-sst-2-english", feature="object-detection"
            )
        self.assertIn("Feature object-detection is not supported.", str(context.exception))


class ORTDynamicQuantizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMUL = {
        "bert-base-cased": 72,
        "roberta-base": 72,
        "distilbert-base-uncased": 36,
        "facebook/bart-base": 96,
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMUL.items())
    def test_dynamic_quantization(self, *args, **kwargs):
        model_name, expected_quantized_matmul = args
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            model.save_pretrained(tmp_dir)
            q8_model_path = output_dir.joinpath("model-quantized.onnx")

            quantizer = ORTQuantizer.from_pretrained(model)
            quantizer.export(
                output_path=q8_model_path,
                quantization_config=qconfig,
            )
            quantized_model = onnx_load(q8_model_path)
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "MatMul" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmul, num_quantized_matmul)
            gc.collect()


class ORTStaticQuantizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMUL = {
        "bert-base-cased": 72,
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMUL.items())
    def test_static_quantization(self, *args, **kwargs):
        model_name, expected_quantized_matmul = args

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
                output_path=q8_model_path,
                calibration_tensors_range=ranges,
                quantization_config=qconfig,
            )

            quantized_model = onnx_load(q8_model_path)
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "MatMul" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmul, num_quantized_matmul)
            gc.collect()


if __name__ == "__main__":
    unittest.main()
