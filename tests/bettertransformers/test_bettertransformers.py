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
import timeit
import unittest

import torch
import transformers
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, pipeline

from optimum.bettertransformer import BETTER_TRANFORMER_LAYERS_MAPPING_DICT, BetterTransformer
from optimum.utils import is_accelerate_available
from optimum.utils.testing_utils import (
    convert_to_hf_classes,
    is_torch_greater_than_113,
    require_accelerate,
    require_torch_gpu,
)


if is_accelerate_available():
    from accelerate import init_empty_weights


class BetterTransformersTest(unittest.TestCase):
    r"""
    Full testing suite of the `BetterTransformers` integration into Hugging Face
    `transformers` ecosystem. Check the docstring of each test to understand the
    purpose of each test. Basically we test:

    - if the conversion dictionnary is consistent, ie if the converted model exists
    in HuggingFace `transformers` library.
    - if the converted model produces the same logits as the original model.
    - if the converted model is faster than the original model.
    """

    def test_dict_consistency(self):
        r"""
        A test to check if the modified dictionnary is consistent (same number of keys + successfully import
        the correct `PreTrainedModel` module).
        """
        for keys in BETTER_TRANFORMER_LAYERS_MAPPING_DICT.keys():
            self.assertTrue(("Layer" in keys) or ("Block" in keys))

        ALL_SUPPORTED_HF_CLASSES = convert_to_hf_classes(BETTER_TRANFORMER_LAYERS_MAPPING_DICT)
        self.assertEqual(len(ALL_SUPPORTED_HF_CLASSES.keys()), len(BETTER_TRANFORMER_LAYERS_MAPPING_DICT.keys()))

    @unittest.skipIf(not is_accelerate_available(), "Skipping the test since `accelerate` is not available...")
    @init_empty_weights()
    def test_conversion(self):
        r"""
        This tests if the conversion of a slow model to its `Fast` version
        has been successfull.
        """
        # Step 0: for each model_class that support the `Fast` version,
        # Step 1: convert the model, ie if it contains the attribute `is_fast`
        # Step 2: check also that some class attributes still remains in the model
        # (for eg, `generate`)

        ALL_SUPPORTED_HF_CLASSES = convert_to_hf_classes(BETTER_TRANFORMER_LAYERS_MAPPING_DICT)

        for layer_class in BETTER_TRANFORMER_LAYERS_MAPPING_DICT.keys():
            if layer_class == "TransformerBlock":
                # Hardcode it for distilbert - see https://github.com/huggingface/transformers/pull/19966
                class_name = "DistilBert"
            elif "EncoderLayer" in layer_class:
                class_name = layer_class[:-12]
            else:
                class_name = layer_class[:-5]
            random_config = getattr(transformers, class_name + "Config")

            hf_random_model = AutoModel.from_config(random_config())
            converted_model = BetterTransformer.transform(hf_random_model)

            self.assertTrue(
                hasattr(converted_model, "is_fast"),
                f"The model {converted_model.__class__.__name__} is not a fast model.",
            )

            self.assertTrue(isinstance(converted_model, ALL_SUPPORTED_HF_CLASSES[layer_class]))
            self.assertTrue(hasattr(converted_model, "generate"))

    def test_logits(self):
        r"""
        This tests if the converted model produces the same logits
        than the original model.
        """
        # The first row of the attention mask needs to be all ones -> check: https://github.com/pytorch/pytorch/blob/19171a21ee8a9cc1a811ac46d3abd975f0b6fc3b/test/test_nn.py#L5283
        inputs_ids = torch.LongTensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        attention_mask = torch.Tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])

        for layer_class in BETTER_TRANFORMER_LAYERS_MAPPING_DICT.keys():
            if layer_class == "TransformerBlock":
                # Hardcode it for distilbert - see https://github.com/huggingface/transformers/pull/19966
                class_name = "DistilBert"
            elif "EncoderLayer" in layer_class:
                class_name = layer_class[:-12]
            else:
                class_name = layer_class[:-5]
            random_config = getattr(transformers, class_name + "Config")

            hf_random_model = AutoModel.from_config(random_config()).eval()
            if hasattr(hf_random_model, "text_model"):
                hf_random_model = hf_random_model.text_model

            converted_model = BetterTransformer.transform(hf_random_model, keep_original_model=True)

            self.assertFalse(
                hasattr(hf_random_model, "is_fast"),
                f"The model {hf_random_model.__class__.__name__} has been converted to a `fast` model by mistake.",
            )

            with torch.no_grad():
                r"""
                Make sure the models are in eval mode! Make also sure that the original model
                has not been converted to a fast model. The check is done above.
                """
                hf_hidden_states = hf_random_model(inputs_ids, attention_mask=attention_mask)[0]
                bt_hidden_states = converted_model(inputs_ids, attention_mask=attention_mask)[0]

                self.assertTrue(
                    torch.allclose(hf_hidden_states[:, :3, :], bt_hidden_states[:, :3, :], atol=1e-3),
                    "The BetterTransformers Converted model does not produce the same logits as the original model. Failed for the model {}".format(
                        hf_random_model.__class__.__name__
                    ),
                )

    def test_raise_pos_emb(self):
        r"""
        Test if the converion properly raises an error if the model has an activate function that is
        not supported by `BtterTransformer`.
        """
        random_config = getattr(transformers, "BertConfig")()
        random_config.hidden_act = "silu"

        with self.assertRaises(ValueError):
            hf_model = AutoModel.from_config(random_config).eval()
            _ = BetterTransformer.transform(hf_model, keep_original_model=False)

    def test_raise_activation_fun(self):
        r"""
        Test if the converion properly raises an error if the model has an activate function that is
        not supported by `BtterTransformer`.
        """
        random_config = getattr(transformers, "BertConfig")()
        random_config.position_embedding_type = "relative"

        with self.assertRaises(ValueError):
            hf_model = AutoModel.from_config(random_config).eval()
            _ = BetterTransformer.transform(hf_model, keep_original_model=False)

    def test_raise_on_save(self):
        r"""
        Test if the converion properly raises an error if someone tries to save the model using `save_pretrained`.
        """
        random_config = getattr(transformers, "BertConfig")()
        with self.assertWarns(UserWarning), tempfile.TemporaryDirectory() as tmpdirname:
            hf_model = AutoModel.from_config(random_config).eval()
            bt_model = BetterTransformer.transform(hf_model, keep_original_model=False)
            bt_model.save_pretrained(tmpdirname)

    @unittest.skipIf(not is_torch_greater_than_113(), "the test needs Pytorch >= 1.13.0")
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

        BATCH_SIZE = 1
        SEQ_LEN = 16
        MAX_SEQ_LEN = 32
        STD_SEQ_LEN = 10  # let's take a large sequence length standard deviation
        VOCAB_SIZE = 50
        N_REPEAT = 10

        input_ids, _, attention_mask = get_batch(BATCH_SIZE, SEQ_LEN, MAX_SEQ_LEN, STD_SEQ_LEN, VOCAB_SIZE)

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

    def test_pipeline(self):
        r"""
        This test runs pipeline together with Better Transformers converted models using optimum `pipeline`.
        """
        from optimum.pipelines import pipeline

        model_name = "distilbert-base-uncased"
        unmasker = pipeline("fill-mask", model_name, accelerator="bettertransformer")

        out = unmasker("Hello I'm a [MASK] model.")

        self.assertEqual(out[0]["token_str"], "role")

    @unittest.skipIf(not is_torch_greater_than_113(), "The test needs accelerate and torch>=1.13 installed")
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
        self.assertSetEqual(set(list(hf_model.hf_device_map.values())), set(max_memory))
        self.assertSetEqual(set(list(bt_model.hf_device_map.values())), set(max_memory))

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

    def test_accelerate_compatibility_cpu_gpu(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=True`
        """
        max_memory = {0: "1GB", "cpu": "3GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=True, max_memory=max_memory)

    def test_accelerate_compatibility_cpu_gpu_without_keeping(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=False`
        """
        max_memory = {0: "1GB", "cpu": "3GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=False, max_memory=max_memory)

    def test_accelerate_compatibility_single_gpu(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=False`
        & `max_memory = {0: "2GB"}`
        """
        max_memory = {0: "2GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=True, max_memory=max_memory)

    def test_accelerate_compatibility_single_gpu_without_keeping(self):
        r"""
        Wrapper around the `check_accelerate_compatibility_cpu_gpu` test with `keep_original_model=True`
        & `max_memory = {0: "2GB"}`
        """
        max_memory = {0: "2GB"}
        self.check_accelerate_compatibility_cpu_gpu(keep_original_model=False, max_memory=max_memory)


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
