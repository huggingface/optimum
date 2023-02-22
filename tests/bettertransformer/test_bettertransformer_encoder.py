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
import gc
import timeit
import unittest

import pytest
import torch
import transformers
from parameterized import parameterized
from testing_bettertransformer_utils import BetterTransformersTestMixin
from transformers import AutoModel, AutoTokenizer

from optimum.bettertransformer import BetterTransformer, BetterTransformerManager
from optimum.utils.testing_utils import grid_parameters, require_accelerate, require_torch_gpu


ALL_ENCODER_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-AlbertModel",
    "hf-internal-testing/tiny-random-BertModel",
    "hf-internal-testing/tiny-random-camembert",
    "hf-internal-testing/tiny-random-Data2VecTextModel",
    "hf-internal-testing/tiny-random-DistilBertModel",
    "hf-internal-testing/tiny-random-ElectraModel",
    "hf-internal-testing/tiny-random-ErnieModel",
    "hf-internal-testing/tiny-random-LayoutLMModel",
    "hf-internal-testing/tiny-random-MarkupLMModel",
    "hf-internal-testing/tiny-random-rembert",
    "hf-internal-testing/tiny-random-RobertaModel",
    "hf-internal-testing/tiny-random-RoFormerModel",
    "hf-internal-testing/tiny-random-SplinterModel",
    "hf-internal-testing/tiny-random-TapasModel",
    "hf-internal-testing/tiny-xlm-roberta",
    "ybelkada/random-tiny-BertGenerationModel",
]

ALL_ENCODER_DECODER_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-bart",
    "hf-internal-testing/tiny-random-FSMTModel",
    "hf-internal-testing/tiny-random-marian",
    "hf-internal-testing/tiny-random-mbart",
    "hf-internal-testing/tiny-random-nllb",
]


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
    all_models_to_test = ALL_ENCODER_MODELS_TO_TEST

    def tearDown(self):
        gc.collect()

    def prepare_inputs_for_class(self, model_id=None):
        input_dict = {
            "input_ids": torch.LongTensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
            "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]),
        }
        return input_dict

    def test_dict_class_consistency(self):
        """
        A test to check BetterTransformerManager.MODEL_MAPPING has good names.
        """
        for model_type, item in BetterTransformerManager.MODEL_MAPPING.items():
            if isinstance(item[0], str):
                self.assertTrue(("Layer" in item[0]) or ("Block" in item[0]))
            else:
                self.assertTrue(
                    all("Layer" in sub_item for sub_item in item[0])
                    or all("Block" in sub_item for sub_item in item[0])
                )

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

    @parameterized.expand(BetterTransformerManager.MODEL_MAPPING.keys())
    def test_raise_activation_fun(self, model_type: str):
        r"""
        A tests that checks if the conversion raises an error if the model contains an activation function
        that is not supported by `BetterTransformer`. Here we need to loop over the config files
        """
        layer_class = BetterTransformerManager.MODEL_MAPPING[model_type][0]
        if isinstance(layer_class, list):
            layer_class = layer_class[0]

        if layer_class == "EncoderLayer":
            # Hardcode it for FSMT - see https://github.com/huggingface/optimum/pull/494
            class_name = "FSMT"
        elif layer_class == "TransformerBlock":
            # Hardcode it for distilbert - see https://github.com/huggingface/transformers/pull/19966
            class_name = "DistilBert"
        elif "EncoderLayer" in layer_class:
            class_name = layer_class[:-12]
        else:
            class_name = layer_class[:-5]

        hf_random_config = getattr(transformers, class_name + "Config")()  # random config class for the model to test
        hf_random_config.hidden_act = "silu"

        hf_random_model = AutoModel.from_config(hf_random_config).eval()
        with self.assertRaises(ValueError):
            _ = BetterTransformer.transform(hf_random_model, keep_original_model=True)

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

    def test_pipeline(self):
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

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
        super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
        super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
        super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_raise_save_pretrained_error(self, test_name: str, model_id, keep_original_model=False):
        super().test_raise_save_pretrained_error(model_id=model_id, keep_original_model=keep_original_model)


class BetterTransformersRoCBertTest(BetterTransformersEncoderTest):
    all_models_to_test = ["hf-internal-testing/tiny-random-RoCBertModel"]

    # unrelated issue with torch.amp.autocast with rocbert (expected scalar type BFloat16 but found Float)
    def test_raise_autocast(self):
        pass


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
    all_models_to_test = ALL_ENCODER_DECODER_MODELS_TO_TEST

    def tearDown(self):
        gc.collect()

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
                "model_id": ALL_ENCODER_DECODER_MODELS_TO_TEST,
                "padding": ["max_length", True],
            }
        )
    )
    def test_logits(self, test_name: str, model_id, padding, max_length=20):
        super().test_logits([model_id], padding=padding, max_length=max_length)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_invert_modules(self, test_name: str, model_id, keep_original_model=False):
        super().test_invert_modules(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_save_load_invertible(self, test_name: str, model_id, keep_original_model=False):
        super().test_save_load_invertible(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_invert_model_logits(self, test_name: str, model_id, keep_original_model=False):
        super().test_invert_model_logits(model_id=model_id, keep_original_model=keep_original_model)

    @parameterized.expand(
        grid_parameters(
            {
                "model_id": all_models_to_test,
                "keep_original_model": [True, False],
            }
        )
    )
    def test_raise_save_pretrained_error(self, test_name: str, model_id, keep_original_model=False):
        super().test_raise_save_pretrained_error(model_id=model_id, keep_original_model=keep_original_model)


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
