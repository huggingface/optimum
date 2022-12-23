# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import onnx
from onnx import load as onnx_load
from onnxruntime import InferenceSession
from optimum.onnx.graph_transformations import merge_decoders, remove_duplicate_weights
from parameterized import parameterized


class WeightSharingTestCase(TestCase):
    def test_weight_sharing_output_match(self):
        with torch.no_grad():

            for model_id in {"albert-base-v1", "albert-base-v2"}:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id)

                task = "default"
                with TemporaryDirectory() as tmpdir:
                    subprocess.run(
                        f"python3 -m optimum.exporters.onnx --model {model_id} --for-ort --task {task} {tmpdir}",
                        shell=True,
                        check=True,
                    )

                    original_albert_ir = onnx_load(os.path.join(tmpdir, "model.onnx"))
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


class OnnxMergingTestCase(TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "hf-internal-testing/tiny-random-GPT2Model": "causal-lm-with-past",
        "hf-internal-testing/tiny-random-T5Model": "seq2seq-lm-with-past",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_merge_decoders(self, *args):
        model_id, task = args

        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"python3 -m optimum.exporters.onnx --model {model_id} --for-ort --task {task} {tmpdir}",
                shell=True,
                check=True,
            )

            decoder = onnx.load(os.path.join(tmpdir, "decoder_model.onnx"))
            decoder_with_past = onnx.load(os.path.join(tmpdir, "decoder_with_past_model.onnx"))

            merge_decoders(decoder, decoder_with_past)


if __name__ == "__main__":
    unittest.main()
