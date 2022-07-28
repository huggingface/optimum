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

from transformers import AutoTokenizer

from onnx import load as onnx_load
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, QuantizationConfig
from optimum.onnxruntime.modeling_ort import ORTModelForSequenceClassification
from optimum.onnxruntime.modeling_seq2seq import ORTModelForSeq2SeqLM
from parameterized import parameterized


class ORTQuantizerTest(unittest.TestCase):
    LOAD_CONFIGURATION = {
        "local_asset": {
            "model_or_path": "assets/onnx",
        },
        "local_asset_different_name": {
            "model_or_path": "assets/onnx",
            "file_name": "different_name.onnx",
        },
        "ort_model_class": {
            "model_or_path": ORTModelForSequenceClassification.from_pretrained(
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
            ORTQuantizer.from_pretrained("bert-base-cased")
        self.assertIn("Unable to load model from bert-base-cased", str(context.exception))

        with self.assertRaises(Exception) as context:
            model = ORTModelForSeq2SeqLM.from_pretrained("optimum/t5-small")
            ORTQuantizer.from_pretrained(model)
        self.assertIn("ORTQuantizer does not support multi-file quantization.", str(context.exception))


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

            quantizer = ORTQuantizer.from_pretrained(model)
            quantizer.quantize(
                output_path=output_dir,
                quantization_config=qconfig,
            )
            quantized_model = onnx_load(output_dir.joinpath("model_quantized.onnx"))
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
            quantizer.quantize(
                output_path=output_dir,
                calibration_tensors_range=ranges,
                quantization_config=qconfig,
            )

            quantized_model = onnx_load(output_dir.joinpath("model_quantized.onnx"))
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "MatMul" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmul, num_quantized_matmul)
            gc.collect()


if __name__ == "__main__":
    unittest.main()
