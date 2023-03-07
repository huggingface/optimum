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
import gc
import timeit
import unittest

import pytest
import torch
import transformers
from packaging.version import parse
from parameterized import parameterized
from testing_utils import MODELS_DICT, BetterTransformersTestMixin
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from optimum.bettertransformer import BetterTransformer, BetterTransformerManager
from optimum.utils.testing_utils import grid_parameters, require_accelerate, require_torch_gpu


class BetterTransformersEncoderTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Full testing suite of the `BetterTransformers` integration into Hugging Face
    `transformers` ecosystem. Check the docstring of each test to understand the
    purpose of each test. Basically we test:
    - if the conversion dictionnary is consistent, ie if the converted model exists
    in HuggingFace `transformers` library.
    - if the converted model produces the same logits as the original model.
    - if the converted model is faster than the original model.
    """
    SUPPORTED_ARCH = [
        "albert",
        "bert",
        "bert-generation",
        "camembert",
        "data2vec-text",
        "distilbert",
        "electra",
        "ernie",
        "layoutlm",
        "markuplm",
        "rembert",
        "roberta",
        "rocbert",
        "roformer",
        "splinter",
        "tapas",
        "xlm_roberta",
    ]

    def tearDown(self):
        gc.collect()

    def prepare_inputs_for_class(self, model_id=None):
        input_dict = {
            "input_ids": torch.LongTensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
            "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]),
        }
        return input_dict

    def test_raise_pos_emb(self):
        r"""
        Test if the converion properly raises an error if the model has an activate function that is
        not supported by `BetterTransformer`. For now, only `Bert` family model support this test.
        """
        random_config = getattr(transformers, "BertConfig")()
        random_config.position_embedding_type = "relative"

        with self.assertRaises(ValueError):
            hf_model = AutoModel.from_config(random_config).eval()
            _ = BetterTransformer.transform(hf_model, keep_original_model=False)

    @torch.no_grad()
    def test_inference_speed(self):
        r"""
        The converted models should be at least slightly faster than the native
        model. This test aims to check this.
        Let's test the inference speed on bert-base-uncased only. If it works for this
        model, it should be applicable to all other models, see the test above.
        """
        model_name = "bert-base-uncased"

        hf_model = AutoModel.from_pretrained(model_name).eval()
        bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

        BATCH_SIZE = 8
        SEQ_LEN = 16
        MAX_SEQ_LEN = 32
        STD_SEQ_LEN = 10  # let's take a large sequence length standard deviation
        VOCAB_SIZE = 50
        N_REPEAT = 10

        input_ids, _, attention_mask = get_batch(BATCH_SIZE, SEQ_LEN, MAX_SEQ_LEN, STD_SEQ_LEN, VOCAB_SIZE)
        for i in range(1, BATCH_SIZE):
            attention_mask[i, SEQ_LEN // 4 :] = 0

        mean_hf_time = 0
        mean_bt_time = 0

        # warmup hf_model
        _ = hf_model(input_ids, attention_mask=attention_mask)
        # warmup bt_model
        _ = bt_model(input_ids, attention_mask=attention_mask)

        for _ in range(N_REPEAT):
            mean_hf_time += timeit.timeit(lambda: hf_model(input_ids, attention_mask=attention_mask), number=1)
            mean_bt_time += timeit.timeit(lambda: bt_model(input_ids, attention_mask=attention_mask), number=1)

        mean_hf_time /= N_REPEAT
        mean_bt_time /= N_REPEAT

        self.assertLess(mean_bt_time, mean_hf_time, "The converted model is slower than the original model.")
        gc.collect()

    def test_pipeline_on_cpu(self):
        r"""
        This test runs pipeline together with Better Transformers converted models using optimum `pipeline`.
        """
        from optimum.pipelines import pipeline

        model_name = "distilbert-base-uncased"
        unmasker = pipeline("fill-mask", model_name, accelerator="bettertransformer")

        out = unmasker("Hello I'm a [MASK] model.")

        self.assertEqual(out[0]["token_str"], "role")
        gc.collect()

    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_pipeline_on_gpu(self):
        r"""
        This test runs pipeline together with Better Transformers converted models using optimum `pipeline`.
        """
        from optimum.pipelines import pipeline

        model_name = "distilbert-base-uncased"
        unmasker = pipeline("fill-mask", model_name, accelerator="bettertransformer", device="cuda:0")

        out = unmasker("Hello I'm a [MASK] model.")

        self.assertEqual(out[0]["token_str"], "role")
        gc.collect()

    @require_torch_gpu
    @require_accelerate
    def check_accelerate_compatibility_cpu_gpu(self, keep_original_model=True, max_memory=None):
        r"""
        This tests if a model loaded with `accelerate` will be successfully converted
        into its BetterTransformers format.
        If this works for roberta, it should work for all other models too.
        """

        hf_model = AutoModel.from_pretrained("xlm-roberta-base", device_map="auto", max_memory=max_memory).eval()
        bt_model = BetterTransformer.transform(
            hf_model, keep_original_model=keep_original_model, max_memory=max_memory
        )

        inputs_ids = torch.LongTensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        attention_mask = torch.Tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])

        # Check that the model has been dispatched on CPU and GPU
        self.assertSetEqual(set(hf_model.hf_device_map.values()), set(max_memory))
        self.assertSetEqual(set(bt_model.hf_device_map.values()), set(max_memory))

        # Check that the model has weights on GPU and CPU
        self.assertEqual(bt_model.encoder.layer[0].in_proj_weight.device, torch.device("cuda:0"))
        # Weights that are offloaded on the CPU are offloaded on the `meta` device
        if "cpu" in set(max_memory):
            self.assertEqual(bt_model.encoder.layer[-1].in_proj_weight.device, torch.device("meta"))

        # Forward pass should work
        output_bt = bt_model(inputs_ids, attention_mask)
        output_hf = hf_model(inputs_ids, attention_mask)

        # Assert that the output has been correctly set to the CPU!
        self.assertEqual(output_bt[0].device, torch.device("cpu"))

        # Final step: check the logits
        self.assertTrue(torch.allclose(output_bt[0][0, :3], output_hf[0][0, :3], atol=1e-3))

        # Check that the padding has been taken into account correctly - this checks also if the hooks
        # have been correctly set.
        self.assertTrue(torch.allclose(output_bt[0][1, 3:], torch.zeros_like(output_bt[0][1, 3:])))
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        if model_type == "rocbert":
            self.skipTest(
                "unrelated issue with torch.amp.autocast with rocbert (expected scalar type BFloat16 but found Float)"
            )

        model_id = MODELS_DICT[model_type]
        super()._test_raise_autocast(model_id)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        model_id = MODELS_DICT[model_type]
        super()._test_raise_train(model_id)

    @pytest.mark.gpu_test
    def test_accelerate_compatibility_cpu_gpu(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=True`
        """
        max_memory = {0: "1GB", "cpu": "3GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=True, max_memory=max_memory)

    @pytest.mark.gpu_test
    def test_accelerate_compatibility_cpu_gpu_without_keeping(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=False`
        """
        max_memory = {0: "1GB", "cpu": "3GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=False, max_memory=max_memory)

    @pytest.mark.gpu_test
    def test_accelerate_compatibility_single_gpu(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=False`
        & `max_memory = {0: "2GB"}`
        """
        max_memory = {0: "2GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=True, max_memory=max_memory)

    @pytest.mark.gpu_test
    def test_accelerate_compatibility_single_gpu_without_keeping(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=True`
        & `max_memory = {0: "2GB"}`
        """
        max_memory = {0: "2GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=False, max_memory=max_memory)

    # TODO: re-enable once fixed
    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)


class BetterTransformersEncoderDecoderTest(BetterTransformersTestMixin, unittest.TestCase):
    r"""
    Full testing suite of the `BetterTransformers` integration into Hugging Face
    `transformers` ecosystem. Check the docstring of each test to understand the
    purpose of each test. Basically we test:
    - if the conversion dictionnary is consistent, ie if the converted model exists
    in HuggingFace `transformers` library.
    - if the converted model produces the same logits as the original model.
    - if the converted model is faster than the original model.
    """
    SUPPORTED_ARCH = [
        "bart",
        "fsmt",
        "m2m_100",
        "marian",
        "mbart",
        "t5",
    ]

    def tearDown(self):
        gc.collect()

    def _skip_on_torch_version(self, model_type: str):
        if BetterTransformerManager.requires_torch_20(model_type) and parse(torch.__version__) < parse("2.0"):
            self.skipTest(f"The model type {model_type} require PyTorch 2.0 for BetterTransformer")

    def prepare_inputs_for_class(self, model_id, **preprocessor_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        padding = preprocessor_kwargs.pop("padding", True)
        inputs = tokenizer(["a dummy input", "and two"], return_tensors="pt", padding=padding, **preprocessor_kwargs)
        inputs["decoder_input_ids"] = inputs["input_ids"]  # just a hack for m2m100
        return inputs

    # run the test over all possible combinations of `model_id` and `padding`
    @parameterized.expand(
        grid_parameters(
            {
                "model_type": SUPPORTED_ARCH,
                "padding": ["max_length", True],
            }
        )
    )
    def test_logits(self, test_name: str, model_type: str, padding, max_length=20):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        super()._test_logits(model_id, padding=padding, max_length=max_length)

    @parameterized.expand(
        grid_parameters({"model_type": SUPPORTED_ARCH, "batch_size": [1, 3], "padding": [True, "max_length"]})
    )
    def test_generation(self, test_name: str, model_type: str, batch_size: int, padding: str):
        self._skip_on_torch_version(model_type)
        if batch_size == 1 and padding == "max_length":
            self.skipTest("batch_size=1 + padding='max_length' is unsupported")

        model_id = MODELS_DICT[model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text = ["This is me and me"]
        if batch_size > 1:
            text = text + ["Please"] * (batch_size - 1)
        inp = tokenizer(text, return_tensors="pt", padding=padding, max_length=25)

        length = 50
        result_vanilla = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        model = BetterTransformer.transform(model)

        result_bettertransformer = model.generate(**inp, num_beams=1, min_length=length, max_length=length)

        self.assertTrue(
            torch.allclose(result_vanilla, result_bettertransformer),
            f" Maxdiff: {(result_vanilla - result_bettertransformer).abs().max()}",
        )

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_autocast(self, model_type: str):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        super()._test_raise_autocast(model_id)

    @parameterized.expand(SUPPORTED_ARCH)
    def test_raise_train(self, model_type: str):
        self._skip_on_torch_version(model_type)
        model_id = MODELS_DICT[model_type]
        if model_type != "t5":
            super()._test_raise_train(model_id)
        else:
            super()._test_train_decoder(model_id)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    # @parameterized.expand(
    #     grid_parameters(
    #         {
    #             "model_id": all_models_to_test,
    #             "keep_original_model": [True, False],
    #         }
    #     )
    # )
    # def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
    #     super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)


def get_batch(batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size, pad_idx=0):
    r"""
    Utility function to generate a batch of random sequences, together with their
    attention mask and lengths.
    Copied from: https://github.com/HamidShojanazeri/transformers/blob/ddf0299a13e7c4f54459a0731abd80204a1078f5/examples/pytorch/benchmarking/benchmark_bettertransformer.py#L149
    """
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)

    # need at least a sequence length of 1 for BetterTransformer to work
    lengths = torch.clamp(lengths, min=1, max=max_sequence_length)

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
