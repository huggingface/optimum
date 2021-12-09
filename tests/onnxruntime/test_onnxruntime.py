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
from pathlib import Path

from transformers.onnx import validate_model_outputs

from optimum.onnxruntime import convert_to_onnx, optimize, quantize


class TestOptimize(unittest.TestCase):
    def test_optimize(self):

        model_names = {"bert-base-cased", "distilbert-base-uncased", "roberta-base", "gpt2", "facebook/bart-base"}

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    onnx_model = Path(tmp_dir).joinpath(model_name.split("/")[-1] + ".onnx")
                    tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(model_name, onnx_model)
                    validate_model_outputs(onnx_config, tokenizer, model, onnx_model, onnx_outputs, atol=1e-4)

                    model_type = getattr(model.config, "model_type")
                    model_type = "bert" if "bert" in model_type else model_type
                    num_heads = getattr(model.config, "num_attention_heads", 0)
                    hidden_size = getattr(model.config, "hidden_size", 0)
                    optimized_model = optimize(
                        onnx_model,
                        model_type,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        opt_level=1,
                        only_onnxruntime=True,
                    )
                    validate_model_outputs(onnx_config, tokenizer, model, optimized_model, onnx_outputs, atol=1e-4)

                    quantized_model = quantize(optimized_model)
                    q_atol = 5 if model_type == "bert" else 12
                    validate_model_outputs(onnx_config, tokenizer, model, quantized_model, onnx_outputs, atol=q_atol)


if __name__ == "__main__":
    unittest.main()
