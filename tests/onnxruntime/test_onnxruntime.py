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
from pathlib import Path

from transformers import AutoTokenizer
from transformers.onnx import validate_model_outputs

from optimum.onnxruntime import ORTConfig, ORTOptimizer, ORTQuantizer


class TestORTOptimizer(unittest.TestCase):
    def test_optimize(self):
        model_names = {"bert-base-cased", "distilbert-base-uncased", "facebook/bart-base", "gpt2", "roberta-base"}
        ort_config = ORTConfig(opt_level=1, only_onnxruntime=False)
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    optim_model_path = output_dir.joinpath("model-optimized.onnx")
                    optimizer = ORTOptimizer(ort_config)
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
        ort_config = ORTConfig(
            quantization_approach="dynamic",
            per_channel=False,
            reduce_range=False,
            weight_type="uint8",
            activation_type="uint8",
        )
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    q8_model_path = output_dir.joinpath("model-quantized.onnx")
                    quantizer = ORTQuantizer(ort_config)
                    quantizer.fit(model_name, output_dir, feature="sequence-classification")
                    validate_model_outputs(
                        quantizer.onnx_config,
                        quantizer.tokenizer,
                        quantizer.model,
                        q8_model_path,
                        list(quantizer.onnx_config.outputs.keys()),
                        atol=8e-1,
                    )
                    gc.collect()

    def test_static_quantization(self):
        model_names = {"distilbert-base-uncased"}
        ort_config = ORTConfig(
            quantization_approach="static",
            per_channel=False,
            reduce_range=False,
            weight_type="uint8",
            activation_type="uint8",
            quant_format="operator",
            calibration_method="minmax",
            optimize_model=True,
            split="train",
            max_samples=80,
            calib_batch_size=8,
        )

        def preprocess_function(examples):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    q8_model_path = output_dir.joinpath("model-quantized.onnx")
                    quantizer = ORTQuantizer(
                        ort_config,
                        dataset_name="glue",
                        dataset_config_name="sst2",
                        preprocess_function=preprocess_function,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    quantizer.fit(model_name, output_dir, feature="sequence-classification")
                    validate_model_outputs(
                        quantizer.onnx_config,
                        tokenizer,
                        quantizer.model,
                        q8_model_path,
                        list(quantizer.onnx_config.outputs.keys()),
                        atol=5e-1,
                    )
                    gc.collect()


if __name__ == "__main__":
    unittest.main()
