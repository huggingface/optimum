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

import tempfile
import unittest

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.gptq import GPTQQuantizer, load_quantized_model
from optimum.gptq.data import get_dataset
from optimum.utils.testing_utils import require_accelerate, require_autogptq, require_torch_gpu


@require_autogptq
@require_torch_gpu
class GTPQTest(unittest.TestCase):
    model_name = "bigscience/bloom-560m"

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a professional photographer. I")
    EXPECTED_OUTPUTS.add("Hello my name is John.\nI am a friend of your father.\n")

    # this seems a little small considering that we are doing 4bit quant but we have a small model and ww don't quantize the embeddings
    EXPECTED_RELATIVE_DIFFERENCE = 1.664253062

    bits = 4
    group_size = 128
    desc_act = False

    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map={"": 0}
        )
        cls.mem_fp16 = cls.model_fp16.get_memory_footprint()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)
        cls.quantizer = GPTQQuantizer(bits=cls.bits, group_size=cls.group_size, desc_act=cls.desc_act)

        cls.quantized_model = cls.quantizer.quantize_model(cls.model_fp16, cls.tokenizer, cls.dataset)

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        mem_quantized = self.quantized_model.get_memory_footprint()

        self.assertAlmostEqual(self.mem_fp16 / mem_quantized, self.EXPECTED_RELATIVE_DIFFERENCE)

    def test_quantized_layers_class(self):
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False, desc_act=self.desc_act, group_size=self.group_size
        )
        self.assertTrue(self.quantized_model.transformer.h[0].mlp.dense_4h_to_h.__class__ == QuantLinear)

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_parallel = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Get the generation
        self.assertIn(self.tokenizer.decode(output_parallel[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality(self):
        self.check_inference_correctness(self.quantized_model)

    @require_accelerate
    def test_serialization(self):
        from accelerate import init_empty_weights

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantizer.save(self.quantized_model, tmpdirname)
            self.quantized_model.config.save_pretrained(tmpdirname)

            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
            empty_model.tie_weights()
            quantized_model_from_saved = load_quantized_model(empty_model, save_folder=tmpdirname, device_map={"": 0})
            self.check_inference_correctness(quantized_model_from_saved)


class GTPQUtilsTest(unittest.TestCase):
    model_name = "facebook/opt-125m"
    expected_seqlen = 2048
    expected_block_name = "model.decoder.layers"
    expected_block_name_class = "OPTDecoderLayer"
    expected_preceding_modules = [
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model.decoder.final_layer_norm",
    ]

    def test_get_seqlen(self):
        from optimum.gptq.utils import get_seqlen

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        seqlen = get_seqlen(model)
        self.assertEqual(seqlen, self.expected_seqlen)

    def test_get_block_name(self):
        from optimum.gptq.utils import get_block_name_with_pattern, get_module_by_name_prefix

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        block_name = get_block_name_with_pattern(model)
        self.assertEqual(block_name, self.expected_block_name)
        block_name_class = get_module_by_name_prefix(model, block_name)[0].__class__.__name__
        self.assertEqual(block_name_class, self.expected_block_name_class)

    def test_get_preceding_modules(self):
        from optimum.gptq.utils import get_preceding_modules

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        modules_names = get_preceding_modules(model, self.expected_block_name)
        self.assertCountEqual(modules_names, self.expected_preceding_modules)


class BloomGTPQUtilsTest(GTPQUtilsTest):
    model_name = "bigscience/bloom-560m"
    expected_seqlen = 2048
    expected_block_name = "transformer.h"
    expected_block_name_class = "BloomBlock"
    expected_preceding_modules = ["transformer.word_embeddings", "transformer.word_embeddings_layernorm"]


class GTPQDataTest(unittest.TestCase):
    model_name = "facebook/opt-125m"
    NBSAMPLES = 128
    SEQLEN = 2048

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    @parameterized.expand(["wikitext2", "c4", "ptb", "c4-new", "ptb-new"])
    def test_dataset(self, dataset):
        train_dataset = get_dataset(
            dataset, self.tokenizer, nsamples=self.NBSAMPLES, seqlen=self.SEQLEN, split="train"
        )
        self.assertEqual(len(train_dataset), self.NBSAMPLES)
        self.assertCountEqual(list(train_dataset[0].keys()), ["input_ids", "attention_mask"])
        self.assertEqual(list(train_dataset[0]["input_ids"].size()), [1, self.SEQLEN])
