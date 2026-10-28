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

from unittest import TestCase

import torch
from parameterized import parameterized
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from optimum.utils.input_generators import DTYPE_MAPPER, DummyDecisionTransformerInputGenerator
from optimum.utils.normalized_config import NormalizedTextConfig


class DecisionTransformerInputsTest(TestCase):
    def make_generator(self, sequence_length):
        config = DecisionTransformerConfig(
            state_dim=4,
            act_dim=2,
            hidden_size=16,
            n_head=2,
            n_layer=1,
            n_positions=64,
            max_ep_len=32,
            bos_token_id=0,
            eos_token_id=0,
        )
        return config, DummyDecisionTransformerInputGenerator(
            "reinforcement-learning", NormalizedTextConfig(config), batch_size=2, sequence_length=sequence_length
        )

    @parameterized.expand(
        [
            (framework, dtype, length)
            for framework in ("pt", "np")
            for dtype in ("int64", "int32", "int8")
            for length in (1, 8)
        ]
    )
    def test_attention_mask_is_binary_integer(self, framework, dtype, sequence_length):
        _, generator = self.make_generator(sequence_length)
        mask = generator.generate("attention_mask", framework=framework, int_dtype=dtype, float_dtype="fp16")

        self.assertEqual(mask.shape, (2, sequence_length))
        self.assertEqual(mask.dtype, getattr(DTYPE_MAPPER, framework)(dtype))
        self.assertTrue(((mask == 0) | (mask == 1)).all())

    def test_generated_inputs_run_with_binary_mask_semantics(self):
        config, generator = self.make_generator(8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = DecisionTransformerModel(config).eval()
            inputs = {
                name: generator.generate(name)
                for name in ("states", "actions", "returns_to_go", "timesteps", "attention_mask")
            }
        mask = inputs["attention_mask"]
        self.assertEqual(mask.dtype, torch.int64)
        self.assertTrue(((mask == 0) | (mask == 1)).all())
        with torch.no_grad():
            actual = model(**inputs).action_preds
            reference = model(**{**inputs, "attention_mask": mask.bool()}).action_preds

        self.assertEqual(actual.shape, (2, 8, config.act_dim))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, reference)

    @parameterized.expand(
        [
            (framework, dtype, name, width)
            for framework in ("pt", "np")
            for dtype in ("fp16", "fp32")
            for name, width in (("states", 4), ("actions", 2), ("returns_to_go", 1))
        ]
    )
    def test_float_inputs_preserve_float_dtype(self, framework, dtype, name, width):
        _, generator = self.make_generator(8)
        value = generator.generate(name, framework=framework, int_dtype="int8", float_dtype=dtype)

        self.assertEqual(value.shape, (2, 8, width))
        self.assertEqual(value.dtype, getattr(DTYPE_MAPPER, framework)(dtype))

    @parameterized.expand([(framework, dtype) for framework in ("pt", "np") for dtype in ("int64", "int32", "int8")])
    def test_timesteps_preserve_integer_dtype(self, framework, dtype):
        config, generator = self.make_generator(8)
        timesteps = generator.generate("timesteps", framework=framework, int_dtype=dtype, float_dtype="fp16")

        self.assertEqual(timesteps.shape, (2, 8))
        self.assertEqual(timesteps.dtype, getattr(DTYPE_MAPPER, framework)(dtype))
        self.assertTrue(((timesteps >= 0) & (timesteps < config.max_ep_len)).all())
