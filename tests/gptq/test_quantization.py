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
from optimum.gptq.eval import evaluate_perplexity
from optimum.gptq.utils import get_block_name_with_pattern, get_preceding_modules, get_seqlen
from optimum.utils import recurse_getattr
from optimum.utils.import_utils import is_accelerate_available, is_auto_gptq_available, is_gptqmodel_available
from optimum.utils.testing_utils import require_gptq, require_torch_gpu


if is_auto_gptq_available():
    from auto_gptq import AutoGPTQForCausalLM
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear as hf_select_quant_linear

if is_gptqmodel_available():
    from gptqmodel import GPTQModel
    from gptqmodel.utils.importer import hf_select_quant_linear

if is_accelerate_available():
    from accelerate import init_empty_weights


@slow
@require_gptq
class GPTQTest(unittest.TestCase):
    model_name = "Felladrin/Llama-68M-Chat-v1"

    expected_fp16_perplexity = 30
    expected_quantized_perplexity = 34

    expected_compression_ratio = 1.2577

    bits = 4
    group_size = 128
    desc_act = False
    sym = True
    disable_exllama = True
    exllama_config = None
    cache_block_outputs = True
    modules_in_block_to_quantize = None
    device_map_for_quantization = "cpu"
    device_for_inference = "cpu"
    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

    # called only once for all tests in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.config = AutoConfig.from_pretrained(cls.model_name)

        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map=cls.device_map_for_quantization
        )
        cls.fp16_mem = cls.model_fp16.get_memory_footprint()

        if cls.device_map_for_quantization != "cpu":
            cls.fp16_ppl = evaluate_perplexity(cls.model_fp16, cls.tokenizer)

        cls.quantizer = GPTQQuantizer(
            bits=cls.bits,
            dataset=cls.dataset,
            group_size=cls.group_size,
            desc_act=cls.desc_act,
            sym=cls.sym,
            disable_exllama=cls.disable_exllama,
            exllama_config=cls.exllama_config,
            cache_block_outputs=cls.cache_block_outputs,
            modules_in_block_to_quantize=cls.modules_in_block_to_quantize,
        )
        cls.quantized_model = cls.quantizer.quantize_model(cls.model_fp16, cls.tokenizer).to(cls.device_for_inference)
        cls.quantized_mem = cls.quantized_model.get_memory_footprint()

        if cls.device_map_for_quantization != "cpu":
            cls.quantized_ppl = evaluate_perplexity(cls.quantized_model, cls.tokenizer)

    def test_memory_footprint(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.assertAlmostEqual(self.fp16_mem / self.quantized_mem, self.expected_compression_ratio, places=2)

    def test_perplexity(self):
        pass

    def test_quantized_layers_class(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        the class type of the linear layers of the converted models
        """

        if is_gptqmodel_available():
            if hasattr(self.config, "quantization_config"):
                checkpoint_format = self.config.quantization_config.get("checkpoint_format")
                meta = self.config.quantization_config.get("meta")
            else:
                checkpoint_format = "gptq"
                meta = None
            QuantLinear = hf_select_quant_linear(
                bits=self.bits,
                group_size=self.group_size,
                desc_act=self.desc_act,
                sym=self.sym,
                device_map=self.device_map_for_quantization,
                checkpoint_format=checkpoint_format,
                meta=meta,
            )
        else:
            QuantLinear = hf_select_quant_linear(
                use_triton=False,
                desc_act=self.desc_act,
                group_size=self.group_size,
                bits=self.bits,
                disable_exllama=self.disable_exllama or self.exllama_config["version"] != 1,
                disable_exllamav2=self.disable_exllama or self.exllama_config["version"] != 2,
            )
        self.assertEqual(self.quantized_model.model.layers[0].mlp.gate_proj.__class__, QuantLinear)

    def check_quantized_layers_type(self, model, value):
        self.assertEqual(model.model.layers[0].mlp.gate_proj.QUANT_TYPE, value)

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
            if is_auto_gptq_available() and not is_gptqmodel_available():
                quant_type = "cuda-old" if self.disable_exllama else "exllama"
            else:
                quant_type = "ipex" if self.device_map_for_quantization == "cpu" else "exllama"

            self.check_quantized_layers_type(quantized_model_from_saved, quant_type)

            # transformers and auto-gptq compatibility
            # quantized models are more compatible with device map than
            # device context managers (they're never used in transformers testing suite)
            _ = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": self.device_for_inference})
            if is_gptqmodel_available():
                _ = GPTQModel.load(tmpdirname, device_map={"": self.device_for_inference})
            else:
                _ = AutoGPTQForCausalLM.from_quantized(tmpdirname, device_map={"": self.device_for_inference})


@require_torch_gpu
class GPTQTestCUDA(GPTQTest):
    device_map_for_quantization = "cuda"
    device_for_inference = 0
    expected_compression_ratio = 1.2577
    expected_fp16_perplexity = 38
    expected_quantized_perplexity = 45

    def test_perplexity(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        the perplexity of the converted models
        """

        self.assertLessEqual(int(self.fp16_ppl), self.expected_fp16_perplexity)
        self.assertLessEqual(int(self.quantized_ppl), self.expected_quantized_perplexity)


class GPTQTestExllama(GPTQTestCUDA):
    disable_exllama = False
    exllama_config = {"version": 1}


class GPTQTestActOrder(GPTQTestCUDA):
    disable_exllama = True
    desc_act = True
    expected_quantized_perplexity = 46

    def test_serialization(self):
        # act_order don't work with qlinear_cuda kernel
        pass

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

            # transformers and auto-gptq compatibility
            # quantized models are more compatible with device map than
            # device context managers (they're never used in transformers testing suite)
            _ = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": self.device_for_inference})
            if is_gptqmodel_available():
                _ = GPTQModel.load(tmpdirname, device_map={"": self.device_for_inference})
            else:
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


class GPTQTestExllamav2(GPTQTestCUDA):
    desc_act = False
    disable_exllama = True
    exllama_config = {"version": 2}

    def test_serialization(self):
        # don't need to test
        pass

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
            self.check_quantized_layers_type(
                quantized_model_from_saved, "exllama" if is_gptqmodel_available else "exllamav2"
            )

            # transformers and auto-gptq compatibility
            # quantized models are more compatible with device map than
            # device context managers (they're never used in transformers testing suite)
            _ = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": self.device_for_inference})
            if is_gptqmodel_available():
                _ = GPTQModel.load(tmpdirname, device_map={"": self.device_for_inference})
            else:
                _ = AutoGPTQForCausalLM.from_quantized(tmpdirname, device_map={"": self.device_for_inference})


class GPTQTestNoBlockCaching(GPTQTestCUDA):
    cache_block_outputs = False


class GPTQTestModuleQuant(GPTQTestCUDA):
    # all layers are quantized apart from self_attention.dense
    modules_in_block_to_quantize = [
        ["self_attn.q_proj"],
        ["mlp.gate_proj"],
    ]
    expected_compression_ratio = 1.068
    expected_quantized_perplexity = 39

    def test_not_converted_layers(self):
        # self_attention.dense should not be converted
        self.assertEqual(self.quantized_model.model.layers[0].self_attn.k_proj.__class__.__name__, "Linear")


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

    @parameterized.expand(["wikitext2", "c4", "c4-new"])
    def test_dataset(self, dataset):
        train_dataset = get_dataset(
            dataset, self.tokenizer, nsamples=self.NBSAMPLES, seqlen=self.SEQLEN, split="train"
        )
        self.assertEqual(len(train_dataset), self.NBSAMPLES)
        self.assertCountEqual(list(train_dataset[0].keys()), ["input_ids", "attention_mask"])
        self.assertEqual(list(train_dataset[0]["input_ids"].size()), [1, self.SEQLEN])
