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

import os
import tempfile
import unittest

from transformers.onnx import validate_model_outputs

from optimum.onnxruntime import ORTConfig, ORTQuantizer


class TestORTQuantizer(unittest.TestCase):
    def test_dynamic_quantization(self):
        model_names = {"bert-base-cased", "distilbert-base-uncased", "facebook/bart-base", "gpt2", "roberta-base"}
        ort_config_dir = os.path.dirname(os.path.abspath(__file__))
        ort_config = ORTConfig.from_pretrained(ort_config_dir)
        ort_config.quantization_approach = "dynamic"

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    quantizer = ORTQuantizer(model_name, tmp_dir, ort_config)
                    quantizer.fit()
                    validate_model_outputs(
                        quantizer.onnx_config,
                        quantizer.tokenizer,
                        quantizer.model,
                        quantizer.quant_model_path,
                        list(quantizer.onnx_config.outputs.keys()),
                        atol=10,
                    )

    def test_static_quantization(self):
        model_names = {"bert-base-cased", "distilbert-base-uncased"}
        ort_config_dir = os.path.dirname(os.path.abspath(__file__))
        ort_config = ORTConfig.from_pretrained(ort_config_dir)
        ort_config.quantization_approach = "static"

        def preprocess_function(examples):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    quantizer = ORTQuantizer(
                        model_name,
                        tmp_dir,
                        ort_config,
                        dataset_name="glue",
                        dataset_config_name="sst2",
                        preprocess_function=preprocess_function,
                    )
                    tokenizer = quantizer.tokenizer
                    quantizer.fit()
                    validate_model_outputs(
                        quantizer.onnx_config,
                        tokenizer,
                        quantizer.model,
                        quantizer.quant_model_path,
                        list(quantizer.onnx_config.outputs.keys()),
                        atol=12,
                    )


if __name__ == "__main__":
    unittest.main()
