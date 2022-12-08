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
import tempfile
import unittest
from typing import List, Optional

import torch
from transformers import AutoModel

from optimum.bettertransformer import BetterTransformer
from optimum.utils.testing_utils import flatten_dict


class BetterTransformersTestMixin:
    r"""
    `BetterTransformersTestMixin` to wrap necessary functions for testing `BetterTransformer`
    integration. This includes the following tests:
        - `test_logits`: This tests if the converted model produces the same logits
        than the original model.
        - `test_raise_on_save`: Test if the converion properly raises an error if someone tries to save the model using `save_pretrained`.
        - `test_raise_autocast`: A tests that checks if the conversion raises an error if the model is run under
        `torch.cuda.amp.autocast`.
        - `test_raise_train`: A tests that checks if the conversion raises an error if the model is run in training mode.
    """
    all_models_to_test = []

    def prepare_inputs_for_class(self, models_to_test=None):
        raise NotImplementedError

    def test_logits(self, models_to_test: Optional[List] = None, **preprocessor_kwargs):
        r"""
        This tests if the converted model produces the same logits
        than the original model.
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283

        if models_to_test is None:
            models_to_test = self.all_models_to_test

        for model_id in models_to_test:
            inputs = self.prepare_inputs_for_class(model_id=model_id, **preprocessor_kwargs)

            torch.manual_seed(0)
            hf_random_model = AutoModel.from_pretrained(model_id).eval()
            random_config = hf_random_model.config

            torch.manual_seed(0)
            converted_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)

            self.assertFalse(
                hasattr(hf_random_model, "use_bettertransformer"),
                f"The model {hf_random_model.__class__.__name__} has been converted to a `fast` model by mistake.",
            )

            with torch.no_grad():
                r"""
                Make sure the models are in eval mode! Make also sure that the original model
                has not been converted to a fast model. The check is done above.
                """
                torch.manual_seed(0)
                hf_hidden_states = hf_random_model(**inputs)[0]

                torch.manual_seed(0)
                bt_hidden_states = converted_model(**inputs)[0]

                if "quick_gelu" in flatten_dict(random_config.to_dict()).values():
                    # Since `quick_gelu` is a rather slightly modified version of `GeLU` we expect a discrepency.
                    tol = 3e-1
                elif "gelu_new" in flatten_dict(random_config.to_dict()).values():
                    # Since `gelu_new` is a slightly modified version of `GeLU` we expect a small
                    # discrepency.
                    tol = 4e-2
                else:
                    tol = 2e-3

                if hasattr(self, "compare_outputs"):
                    self.compare_outputs(
                        hf_hidden_states, bt_hidden_states, atol=tol, model_name=hf_random_model.__class__.__name__
                    )
                elif "attention_mask" in inputs:
                    for i, attention_mask in enumerate(inputs["attention_mask"]):
                        length = torch.argwhere(attention_mask != 0).max().item()
                        self.assert_equal(
                            tensor1=hf_hidden_states[i, : length + 1, :],
                            tensor2=bt_hidden_states[i, : length + 1, :],
                            atol=tol,
                            model_name=hf_random_model.__class__.__name__,
                        )
                else:
                    self.assert_equal(
                        tensor1=hf_hidden_states[:, :3, :],
                        tensor2=bt_hidden_states[:, :3, :],
                        atol=tol,
                        model_name=hf_random_model.__class__.__name__,
                    )

    def assert_equal(self, tensor1, tensor2, atol: float, model_name: str):
        self.assertTrue(
            torch.allclose(tensor1, tensor2, atol=atol),
            f"The BetterTransformer converted model does not produce the same logits as the original model. Failed for the model {model_name}."
            f" Maxdiff: {torch.abs(tensor1 - tensor2).max()}",
        )

    def test_raise_on_save(self):
        r"""
        Test if the converion properly raises an error if someone tries to save the model using `save_pretrained`.
        """
        for model_id in self.all_models_to_test:
            with self.assertWarns(UserWarning), tempfile.TemporaryDirectory() as tmpdirname:
                hf_model = AutoModel.from_pretrained(model_id).eval()
                bt_model = BetterTransformer.transform(hf_model, keep_original_model=False)
                bt_model.save_pretrained(tmpdirname)

    def test_raise_autocast(self, models_to_test=None, **preprocessor_kwargs):
        r"""
        A tests that checks if the conversion raises an error if the model is run under
        `torch.cuda.amp.autocast`.
        """
        if models_to_test is None:
            models_to_test = self.all_models_to_test

        for model_id in models_to_test:
            inputs = self.prepare_inputs_for_class(model_id=model_id, **preprocessor_kwargs)
            hf_random_model = AutoModel.from_pretrained(model_id).eval()

            # Check for the autocast on CPU
            with self.assertRaises(ValueError), torch.amp.autocast("cpu"):
                bt_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)
                _ = bt_model(**inputs)

    def test_raise_train(self, models_to_test=None, **preprocessor_kwargs):
        r"""
        A tests that checks if the conversion raises an error if the model is run under
        `model.train()`.
        """
        if models_to_test is None:
            models_to_test = self.all_models_to_test

        for model_id in models_to_test:
            inputs = self.prepare_inputs_for_class(model_id=model_id, **preprocessor_kwargs)

            hf_random_model = AutoModel.from_pretrained(model_id).eval()
            # Check for training mode
            with self.assertRaises(ValueError):
                bt_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)
                bt_model.train()
                _ = bt_model(**inputs)

    def test_conversion(self):
        r"""
        This tests if the conversion of a slow model to its BetterTransformer version using fastpath
        has been successfull.
        """

        for model_id in self.all_models_to_test:
            hf_random_model = AutoModel.from_pretrained(model_id)
            converted_model = BetterTransformer.transform(hf_random_model)

            self.assertTrue(
                hasattr(converted_model, "use_bettertransformer"),
                f"The model {converted_model.__class__.__name__} is not a fast model.",
            )

            self.assertTrue(isinstance(converted_model, hf_random_model.__class__))
            self.assertTrue(hasattr(converted_model, "generate"))


def get_batch(batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size, pad_idx=0):
    r"""
    Utility function to generate a batch of random sequences, together with their
    attention mask and lengths.
    Copied from: https://github.com/HamidShojanazeri/transformers/blob/ddf0299a13e7c4f54459a0731abd80204a1078f5/examples/pytorch/benchmarking/benchmark_bettertransformer.py#L149
    """
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)
    lengths = torch.clamp(lengths, min=0, max=max_sequence_length)
    tokens = torch.full(
        (batch_size, max_sequence_length),
        pad_idx,
    )
    for i in range(batch_size):
        tokens[i, : lengths[i]] = torch.randint(
            pad_idx + 1,
            vocab_size - 1,
            size=(lengths[i],),
        )
    mask = torch.full(
        (batch_size, max_sequence_length),
        0,
    )
    for i in range(batch_size):
        mask[i, : lengths[i]] = 1
    return tokens, lengths, mask
