# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import unittest
from unittest.mock import patch

from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer
from optimum.pipelines import pipeline


class BetterTransformerIntegrationTests(unittest.TestCase):
    def test_raise_error_on_double_transform_call(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")

        with self.assertRaises(Exception) as cm:
            bt_model = BetterTransformer.transform(model)
            bt_model = BetterTransformer.transform(bt_model)
        self.assertTrue("was called on a model already using Better Transformer" in str(cm.exception))

    @patch("optimum.utils.import_utils.is_onnxruntime_available")
    def test_direct_pipleine_initialization_without_onnx_installed(self, mock_onnxruntime_availability):
        mock_onnxruntime_availability.return_value = False
        pipe = pipeline("question-answering", "hf-internal-testing/tiny-random-BertModel",
                        accelerator="bettertransformer", )
        pipe(question=["Is huggingface getting better?", "Will it ever stop getting better?"],
             context=["Huggingface will never stop getting better."] * 2)
