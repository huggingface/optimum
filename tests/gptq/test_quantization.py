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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import slow

from optimum.gptq import GPTQQuantizer, load_quantized_model
from optimum.gptq.data import get_dataset
from optimum.utils.import_utils import is_auto_gptq_available
from optimum.utils.testing_utils import require_accelerate, require_auto_gptq, require_torch_gpu


if is_auto_gptq_available():
    from auto_gptq import AutoGPTQForCausalLM


@slow
@require_auto_gptq
@require_torch_gpu
class GPTQTest(unittest.TestCase):
    model_name = "bigscience/bloom-560m"

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a professional photographer. I")
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a professional photographer and I")
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a very good looking man.")
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a student in the University of")

    # this seems a little small considering that we are doing 4bit quant but we have a small model and ww don't quantize the embeddings
    EXPECTED_RELATIVE_DIFFERENCE = 1.664253062

    bits = 4
    group_size = 128
    desc_act = False
    disable_exllama = True
    exllama_config = None
    cache_block_outputs = True
    modules_to_quantize_inside_block = None

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
        cls.quantizer = GPTQQuantizer(
            bits=cls.bits,
            dataset=cls.dataset,
            group_size=cls.group_size,
            desc_act=cls.desc_act,
            disable_exllama=cls.disable_exllama,
            exllama_config=cls.exllama_config,
            cache_block_outputs=cls.cache_block_outputs,
            modules_to_quantize_inside_block=cls.modules_to_quantize_inside_block,
        )

        cls.quantized_model = cls.quantizer.quantize_model(cls.model_fp16, cls.tokenizer)

    def test_memory_footprint(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        mem_quantized = self.quantized_model.get_memory_footprint()

        self.assertAlmostEqual(self.mem_fp16 / mem_quantized, self.EXPECTED_RELATIVE_DIFFERENCE)

    def test_quantized_layers_class(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        the class type of the linear layers of the converted models
        """
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
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
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Get the generation
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Check the exactness of the result
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality(self):
        self.check_inference_correctness(self.quantized_model)

    @require_torch_gpu
    @require_accelerate
    @slow
    def test_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights
        """
        from accelerate import init_empty_weights

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
                device_map={"": 0},
                disable_exllama=self.disable_exllama,
                exllama_config=self.exllama_config,
            )
            if self.disable_exllama:
                self.check_quantized_layers_type(quantized_model_from_saved, "cuda-old")
            else:
                self.check_quantized_layers_type(quantized_model_from_saved, "exllama")

            with torch.device("cuda"):
                _ = AutoModelForCausalLM.from_pretrained(tmpdirname)
            _ = AutoGPTQForCausalLM.from_quantized(tmpdirname)

            self.check_inference_correctness(quantized_model_from_saved)


class GPTQTestExllama(GPTQTest):
    disable_exllama = False
    exllama_config = {"version": 1}
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a professional photographer and I")
    EXPECTED_OUTPUTS.add("Hello my name is jay and i am a student at university.")
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a student in the University of")
    EXPECTED_OUTPUTS.add("Hello my name is Nate and I am a new member of the")


class GPTQTestActOrder(GPTQTest):
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is jay and i am a student at university.")
    EXPECTED_OUTPUTS.add("Hello my name is jessie and i am a very sweet and")
    EXPECTED_OUTPUTS.add("Hello my name is nathalie, I am a young girl from")
    EXPECTED_OUTPUTS.add("Hello my name is\nI am a student of the University of the'")

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
        from accelerate import init_empty_weights

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantizer.save(self.quantized_model, tmpdirname)
            self.quantized_model.config.save_pretrained(tmpdirname)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(self.model_name), torch_dtype=torch.float16
                )
            empty_model.tie_weights()
            quantized_model_from_saved = load_quantized_model(
                empty_model, save_folder=tmpdirname, device_map={"": 0}, exllama_config={"version": 1}
            )
            self.check_quantized_layers_type(quantized_model_from_saved, "exllama")

            with torch.device("cuda"):
                _ = AutoModelForCausalLM.from_pretrained(tmpdirname)
            _ = AutoGPTQForCausalLM.from_quantized(tmpdirname)

            self.check_inference_correctness(quantized_model_from_saved)

    def test_exllama_max_input_length(self):
        """
        Test if the max_input_length works with exllama + act_order
        """
        from accelerate import init_empty_weights

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
                device_map={"": 0},
                max_input_length=4028,
                exllama_config={"version": 1},
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
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a professional photographer and I")
    EXPECTED_OUTPUTS.add("Hello my name is jay and i am a student at university.")
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a student in the University of")
    EXPECTED_OUTPUTS.add("Hello my name is Nate and I am a new member of the")

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
        from accelerate import init_empty_weights

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
                device_map={"": 0},
            )
            self.check_quantized_layers_type(quantized_model_from_saved, "exllamav2")

            with torch.device("cuda"):
                _ = AutoModelForCausalLM.from_pretrained(tmpdirname)
            _ = AutoGPTQForCausalLM.from_quantized(tmpdirname)

            self.check_inference_correctness(quantized_model_from_saved)


class GPTQTestNoBlockCaching(GPTQTest):
    cache_block_outputs = False
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a professional photographer and I")
    EXPECTED_OUTPUTS.add("Hello my name is jay and i am a student at university.")
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a student in the University of")
    EXPECTED_OUTPUTS.add("Hello my name is Aiden and I am a very good looking")


class GPTQTestModuleQuant(GPTQTest):
    # all layers are quantized apart from self_attention.dense
    modules_in_block_to_quantize = [
        ["self_attention.query_key_value"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
    EXPECTED_RELATIVE_DIFFERENCE = 1.57705236164535

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
        from optimum.gptq.utils import get_seqlen

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        seqlen = get_seqlen(model)
        self.assertEqual(seqlen, self.expected_seqlen)

    def test_get_block_name(self):
        from optimum.gptq.utils import get_block_name_with_pattern
        from optimum.utils import recurse_getattr

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        block_name = get_block_name_with_pattern(model)
        self.assertEqual(block_name, self.expected_block_name)
        block_class_name = recurse_getattr(model, block_name)[0].__class__.__name__
        self.assertEqual(block_class_name, self.expected_block_name_class)

    def test_get_preceding_modules(self):
        from optimum.gptq.utils import get_preceding_modules

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
