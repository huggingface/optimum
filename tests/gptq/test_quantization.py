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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, enable_full_determinism
from transformers.testing_utils import slow

from optimum.gptq import GPTQQuantizer, load_quantized_model
from optimum.gptq.data import get_dataset
from optimum.gptq.utils import get_block_name_with_pattern, get_preceding_modules, get_seqlen
from optimum.utils import recurse_getattr
from optimum.utils.import_utils import is_accelerate_available, is_auto_gptq_available
from optimum.utils.testing_utils import require_accelerate, require_auto_gptq, require_torch_gpu


if is_auto_gptq_available():
    from auto_gptq import AutoGPTQForCausalLM
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

if is_accelerate_available():
    from accelerate import init_empty_weights


@slow
@require_auto_gptq
@require_torch_gpu
class GPTQTest(unittest.TestCase):
    model_name = "bigscience/bloom-560m"

    input_text = "Hello my name is"

    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is Alyson and I am a very sweet,")
    EXPECTED_OUTPUTS.add("Hello my name is Aiden and I am a very sweet, car")
    EXPECTED_OUTPUTS.add("Hello my name is Aiden and I am a very good looking,")
    EXPECTED_OUTPUTS.add("Hello my name is Aiden, I am a student in the University")
    EXPECTED_OUTPUTS.add("Hello my name is Aiden, I am a student at the University")
    EXPECTED_OUTPUTS.add("Hello my name is nancy and i am a very sweet and very")
    EXPECTED_OUTPUTS.add("Hello my name is jessie and i am a new member of")
    EXPECTED_OUTPUTS.add("Hello my name is A.I. and I am a student of")
    EXPECTED_OUTPUTS.add("Hello my name is A.I.I.I.I.")

    compression_ratio = 1.66

    bits = 4
    group_size = 128
    desc_act = False
    disable_exllama = True
    exllama_config = None
    cache_block_outputs = True
    modules_in_block_to_quantize = None
    device_map_for_quantization = "cuda"
    device_for_inference = 0
    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

    # called only once for all tests in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """

        enable_full_determinism(42)

        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map=cls.device_map_for_quantization
        )
        cls.fp16_mem = cls.model_fp16.get_memory_footprint()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)
        cls.quantizer = GPTQQuantizer(
            bits=cls.bits,
            dataset=cls.dataset,
            group_size=cls.group_size,
            desc_act=cls.desc_act,
            disable_exllama=cls.disable_exllama,
            exllama_config=cls.exllama_config,
            cache_block_outputs=cls.cache_block_outputs,
            modules_in_block_to_quantize=cls.modules_in_block_to_quantize,
        )
        cls.quantized_model = cls.quantizer.quantize_model(cls.model_fp16, cls.tokenizer)
        cls.quantized_model = cls.quantized_model.to(cls.device_for_inference)
        cls.quantized_mem = cls.quantized_model.get_memory_footprint()

    def test_memory_footprint(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.assertAlmostEqual(self.fp16_mem / self.quantized_mem, self.compression_ratio, places=2)

    def test_quantized_layers_class(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        the class type of the linear layers of the converted models
        """

        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            use_qigen=False,
            desc_act=self.desc_act,
            group_size=self.group_size,
            bits=self.bits,
            disable_exllama=self.disable_exllama or self.exllama_config["version"] != 1,
            disable_exllamav2=self.disable_exllama or self.exllama_config["version"] != 2,
        )
        self.assertTrue(self.quantized_model.transformer.h[0].mlp.dense_4h_to_h.__class__ == QuantLinear)

    def check_quantized_layers_type(self, model, value):
        self.assertTrue(model.transformer.h[0].mlp.dense_4h_to_h.QUANT_TYPE == value)

    def check_inference_correctness(self, model):
        """
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        enable_full_determinism(42)

        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to(self.device_for_inference)
        output_ids = model.generate(input_ids, do_sample=False, min_new_tokens=10, max_new_tokens=10)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        self.assertIn(output_text, self.EXPECTED_OUTPUTS)

    def test_generate_quality(self):
        self.check_inference_correctness(self.quantized_model)

    # TODO: decouple the serialization test into more atomic tests
    @slow
    @require_torch_gpu
    @require_accelerate
    def test_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantizer.save(self.quantized_model, tmpdirname)
            self.quantized_model.config.save_pretrained(tmpdirname)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(self.model_name), torch_dtype=torch.float16
                )
            empty_model.tie_weights()
            quantized_model_from_saved = load_quantized_model(
                empty_model,
                save_folder=tmpdirname,
                device_map={"": self.device_for_inference},
                disable_exllama=self.disable_exllama,
                exllama_config=self.exllama_config,
            )
            if self.disable_exllama:
                self.check_quantized_layers_type(quantized_model_from_saved, "cuda-old")
            else:
                self.check_quantized_layers_type(quantized_model_from_saved, "exllama")

            self.check_inference_correctness(quantized_model_from_saved)

            # transformers and auto-gptq compatibility
            # quantized models are more compatible with device map than
            # device context managers (they're never used in transformers testing suite)
            _ = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": self.device_for_inference})
            _ = AutoGPTQForCausalLM.from_quantized(tmpdirname, device_map={"": self.device_for_inference})


class GPTQTestCPUInit(GPTQTest):
    device_map_for_quantization = "cpu"


class GPTQTestExllama(GPTQTest):
    disable_exllama = False
    exllama_config = {"version": 1}


class GPTQTestActOrder(GPTQTest):
    disable_exllama = True
    desc_act = True

    def test_generate_quality(self):
        # act_order don't work with qlinear_cuda kernel
        pass

    def test_serialization(self):
        # act_order don't work with qlinear_cuda kernel
        pass

    @require_torch_gpu
    def test_exllama_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights with exllama kernel
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantizer.save(self.quantized_model, tmpdirname)
            self.quantized_model.config.save_pretrained(tmpdirname)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(self.model_name), torch_dtype=torch.float16
                )
            empty_model.tie_weights()
            quantized_model_from_saved = load_quantized_model(
                empty_model,
                save_folder=tmpdirname,
                device_map={"": self.device_for_inference},
                exllama_config={"version": 1},
            )
            self.check_quantized_layers_type(quantized_model_from_saved, "exllama")
            self.check_inference_correctness(quantized_model_from_saved)

            # transformers and auto-gptq compatibility
            # quantized models are more compatible with device map than
            # device context managers (they're never used in transformers testing suite)
            _ = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": self.device_for_inference})
            _ = AutoGPTQForCausalLM.from_quantized(tmpdirname, device_map={"": self.device_for_inference})

    def test_exllama_max_input_length(self):
        """
        Test if the max_input_length works with exllama + act_order
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantizer.save(self.quantized_model, tmpdirname)
            self.quantized_model.config.save_pretrained(tmpdirname)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(self.model_name), torch_dtype=torch.float16
                )
            empty_model.tie_weights()
            quantized_model_from_saved = load_quantized_model(
                empty_model,
                save_folder=tmpdirname,
                device_map={"": self.device_for_inference},
                exllama_config={"version": 1},
                max_input_length=4028,
            )
            self.check_quantized_layers_type(quantized_model_from_saved, "exllama")

            prompt = "I am in Paris and" * 1000
            inp = self.tokenizer(prompt, return_tensors="pt").to(0)
            self.assertTrue(inp["input_ids"].shape[1] > 4028)
            with self.assertRaises(RuntimeError) as cm:
                quantized_model_from_saved.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
                self.assertTrue("temp_state buffer is too small" in str(cm.exception))

            prompt = "I am in Paris and" * 500
            inp = self.tokenizer(prompt, return_tensors="pt").to(0)
            self.assertTrue(inp["input_ids"].shape[1] < 4028)
            quantized_model_from_saved.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)


class GPTQTestExllamav2(GPTQTest):
    desc_act = False
    disable_exllama = True
    exllama_config = {"version": 2}

    def test_generate_quality(self):
        # don't need to test
        pass

    def test_serialization(self):
        # don't need to test
        pass

    @require_torch_gpu
    def test_exllama_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights with exllamav2 kernel
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantizer.save(self.quantized_model, tmpdirname)
            self.quantized_model.config.save_pretrained(tmpdirname)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(self.model_name), torch_dtype=torch.float16
                )
            empty_model.tie_weights()
            quantized_model_from_saved = load_quantized_model(
                empty_model,
                save_folder=tmpdirname,
                device_map={"": self.device_for_inference},
            )
            self.check_quantized_layers_type(quantized_model_from_saved, "exllamav2")
            self.check_inference_correctness(quantized_model_from_saved)

            # transformers and auto-gptq compatibility
            # quantized models are more compatible with device map than
            # device context managers (they're never used in transformers testing suite)
            _ = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": self.device_for_inference})
            _ = AutoGPTQForCausalLM.from_quantized(tmpdirname, device_map={"": self.device_for_inference})


class GPTQTestNoBlockCaching(GPTQTest):
    cache_block_outputs = False


class GPTQTestModuleQuant(GPTQTest):
    # all layers are quantized apart from self_attention.dense
    modules_in_block_to_quantize = [
        ["self_attention.query_key_value"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
    compression_ratio = 1.577

    def test_not_converted_layers(self):
        # self_attention.dense should not be converted
        self.assertTrue(self.quantized_model.transformer.h[0].self_attention.dense.__class__.__name__ == "Linear")


class GPTQUtilsTest(unittest.TestCase):
    """
    Test utilities
    """

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
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        seqlen = get_seqlen(model)
        self.assertEqual(seqlen, self.expected_seqlen)

    def test_get_block_name(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        block_name = get_block_name_with_pattern(model)
        self.assertEqual(block_name, self.expected_block_name)
        block_class_name = recurse_getattr(model, block_name)[0].__class__.__name__
        self.assertEqual(block_class_name, self.expected_block_name_class)

    def test_get_preceding_modules(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        modules_names = get_preceding_modules(model, self.expected_block_name)
        self.assertCountEqual(modules_names, self.expected_preceding_modules)


class BloomGPTQUtilsTest(GPTQUtilsTest):
    model_name = "bigscience/bloom-560m"
    expected_seqlen = 2048
    expected_block_name = "transformer.h"
    expected_block_name_class = "BloomBlock"
    expected_preceding_modules = ["transformer.word_embeddings", "transformer.word_embeddings_layernorm"]


class GPTQDataTest(unittest.TestCase):
    """
    Test data
    """

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
