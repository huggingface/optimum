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
import timeit
import unittest

import torch
import transformers
from transformers import AutoModel

from optimum.bettertransformer import FAST_LAYERS_MAPPING_DICT, BetterTransformer, convert_to_hf_classes
from optimum.utils.testing_utils import is_accelerate_available


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
        for keys in FAST_LAYERS_MAPPING_DICT.keys():
            self.assertTrue("Layer" in keys)

        ALL_SUPPORTED_HF_CLASSES = convert_to_hf_classes(FAST_LAYERS_MAPPING_DICT)
        self.assertEqual(len(ALL_SUPPORTED_HF_CLASSES.keys()), len(FAST_LAYERS_MAPPING_DICT.keys()))

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

        ALL_SUPPORTED_HF_CLASSES = convert_to_hf_classes(FAST_LAYERS_MAPPING_DICT)

        for layer_class in FAST_LAYERS_MAPPING_DICT.keys():
            random_config = getattr(transformers, layer_class[:-5] + "Config")

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

        for layer_class in FAST_LAYERS_MAPPING_DICT.keys():
            random_config = getattr(transformers, layer_class[:-5] + "Config")

            hf_random_model = AutoModel.from_config(random_config()).eval()
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

        BATCH_SIZE = 32
        SEQ_LEN = 128
        MAX_SEQ_LEN = 256
        STD_SEQ_LEN = 60  # let's take a large sequence length
        VOCAB_SIZE = 50
        N_REPEAT = 10

        input_ids, _, attention_mask = get_batch(BATCH_SIZE, SEQ_LEN, MAX_SEQ_LEN, STD_SEQ_LEN, VOCAB_SIZE)

        mean_hf_time = 0
        mean_bt_time = 0

        for _ in range(N_REPEAT):
            mean_hf_time += timeit.timeit(lambda: hf_model(input_ids, attention_mask=attention_mask), number=1)
            mean_bt_time += timeit.timeit(lambda: bt_model(input_ids, attention_mask=attention_mask), number=1)

        mean_hf_time /= N_REPEAT
        mean_bt_time /= N_REPEAT

        self.assertLess(
            mean_bt_time, mean_hf_time, "The converted model is slower than the original model. Failed for the model"
        )

    def test_class_functions(self):
        r"""
        This test runs class functions such as `generate` and checks if the
        function works as expected.
        """
        pass

    def test_accelerate_compatibility(self):
        r"""
        This tests if a model loaded with `accelerate` will be successfully converted
        into its BetterTransformers format.
        """
        pass


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
    lengths[0] = max_sequence_length
    for i in range(batch_size):
        mask[i, : lengths[i]] = 1
    return tokens, lengths, mask
