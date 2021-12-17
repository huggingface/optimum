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

import tempfile
import unittest
from enum import Enum
from pathlib import Path

from transformers.onnx import validate_model_outputs

from optimum.onnxruntime import convert_to_onnx
from optimum.onnxruntime.quantization import ORTQuantizer

from onnxruntime.transformers.optimizer import FusionOptions


class TestORTQuantizer(unittest.TestCase):
    def test_static_quantization(self):

        model_names = {"bert-base-cased", "distilbert-base-uncased"}

        def preprocess_function(examples):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:

                    class FusionConfig(Enum):
                        model_type = "bert"
                        disable_gelu = False
                        disable_layer_norm = False
                        disable_attention = False
                        disable_skip_layer_norm = False
                        disable_bias_skip_layer_norm = False
                        disable_bias_gelu = False
                        enable_gelu_approximation = False
                        use_mask_index = False
                        no_attention_mask = False
                        disable_embed_layer_norm = True
    

                    model_type = "bert"
                    optimization_options = FusionOptions.parse(FusionConfig)

                    quantizer = ORTQuantizer(
                        model_name,
                        tmp_dir,
                        quantization_approach="static",
                        dataset_name="glue",
                        dataset_config_name="sst2",
                        split="validation",
                        preprocess_function=preprocess_function,
                        optimization_options=optimization_options,
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
