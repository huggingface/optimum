# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from unittest import TestCase, skipUnless

import numpy as np

from optimum.utils import DTYPE_MAPPER, DummyVisionEmbeddingsGenerator, NormalizedConfig, is_torch_available


class DummyVisionEmbeddingsGeneratorTest(TestCase):
    def setUp(self):
        self.generator = DummyVisionEmbeddingsGenerator(
            task="mask-generation",
            normalized_config=NormalizedConfig({}),
            batch_size=1,
            image_embedding_size=2,
            output_channels=3,
        )

    def test_numpy_float_dtype(self):
        for input_name in self.generator.SUPPORTED_INPUT_NAMES:
            for dtype in ("fp32", "fp16"):
                with self.subTest(input_name=input_name, dtype=dtype):
                    tensor = self.generator.generate(input_name, framework="np", float_dtype=dtype)
                    self.assertEqual(tensor.shape, (1, 3, 2, 2))
                    self.assertEqual(tensor.dtype, DTYPE_MAPPER.np(dtype))

    def test_numpy_default_dtype(self):
        for input_name in self.generator.SUPPORTED_INPUT_NAMES:
            with self.subTest(input_name=input_name):
                tensor = self.generator.generate(input_name, framework="np")
                self.assertEqual(tensor.shape, (1, 3, 2, 2))
                self.assertEqual(tensor.dtype, np.float32)

    @skipUnless(is_torch_available(), "PyTorch is not installed")
    def test_pytorch_float_dtype(self):
        for input_name in self.generator.SUPPORTED_INPUT_NAMES:
            for dtype in ("fp32", "fp16", "bf16"):
                with self.subTest(input_name=input_name, dtype=dtype):
                    tensor = self.generator.generate(input_name, framework="pt", float_dtype=dtype)
                    self.assertEqual(tuple(tensor.shape), (1, 3, 2, 2))
                    self.assertEqual(tensor.dtype, DTYPE_MAPPER.pt(dtype))
