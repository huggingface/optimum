#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.albert import AlbertOnnxConfig
from transformers.onnx import export

from onnx import load as onnx_load
from onnxruntime import InferenceSession
from optimum.onnx.graph_transformations import remove_duplicate_weights


class WeightSharingTestCase(TestCase):
    def test_weight_sharing_output_match(self):
        with torch.no_grad():
            for model in {"albert-base-v1", "albert-base-v2"}:
                tokenizer = AutoTokenizer.from_pretrained(model)
                model = AutoModel.from_pretrained(model)
                onnx_config = AlbertOnnxConfig.from_model_config(model.config)

                with NamedTemporaryFile("w+b") as original_onnx_f:
                    export(tokenizer, model, onnx_config, opset=12, output=Path(original_onnx_f.name))

                    original_albert_ir = onnx_load(original_onnx_f)
                    compressed_albert_ir = remove_duplicate_weights(original_albert_ir, inplace=False)
                    compressed_albert_session = InferenceSession(
                        compressed_albert_ir.SerializeToString(), providers=["CPUExecutionProvider"]
                    )

                original_outputs = model(**tokenizer("Hello from Hugging Face", return_tensors="pt"))
                compressed_outputs = compressed_albert_session.run(
                    None, dict(tokenizer("Hello from Hugging Face", return_tensors="np"))
                )

            self.assertTrue(
                np.allclose(original_outputs.last_hidden_state.cpu().numpy(), compressed_outputs[0], atol=1e-4)
            )
            self.assertTrue(
                np.allclose(original_outputs.pooler_output.cpu().numpy(), compressed_outputs[1], atol=1e-4)
            )


if __name__ == "__main__":
    unittest.main()
