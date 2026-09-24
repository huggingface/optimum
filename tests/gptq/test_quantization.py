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
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import torch
from parameterized import parameterized
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers.testing_utils import slow

from optimum.gptq import GPTQQuantizer, load_quantized_model
from optimum.gptq.data import get_dataset
from optimum.gptq.eval import evaluate_perplexity
from optimum.gptq.utils import get_block_name_with_pattern, get_preceding_modules, get_seqlen
from optimum.utils import recurse_getattr
from optimum.utils.import_utils import is_accelerate_available, is_gptqmodel_available
from optimum.utils.testing_utils import require_gptqmodel, require_torch_gpu


if is_gptqmodel_available():
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.importer import hf_select_quant_linear_v2

if is_accelerate_available():
    from accelerate import init_empty_weights


@slow
@require_gptqmodel
@require_torch_gpu
class GPTQTest(unittest.TestCase):
    model_name = "bigscience/bloom-560m"

    expected_fp16_perplexity = 30
    expected_quantized_perplexity = 34

    expected_compression_ratio = 1.66

    bits = 4
    group_size = 128
    sym = True
    desc_act = False
    act_group_aware = True
    cache_block_outputs = True
    modules_in_block_to_quantize = None
    device_map_for_quantization = "cuda"
    device_for_inference = 0
    dataset = [
        "GPT-QModel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

    # called only once for all tests in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

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
            sym=cls.sym,
            desc_act=cls.desc_act,
            act_group_aware=cls.act_group_aware,
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
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        the perplexity of the converted models
        """

        self.assertEqual(int(self.fp16_ppl), self.expected_fp16_perplexity)
        self.assertEqual(int(self.quantized_ppl), self.expected_quantized_perplexity)

    def test_quantized_layers_class(self):
        """
        A simple test to check if the model conversion has been done correctly by checking on the
        the class type of the linear layers of the converted models
        """
        QuantLinear = hf_select_quant_linear_v2(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            sym=self.sym,
            format=FORMAT.GPTQ,
            quant_method=METHOD.GPTQ,
            device_map=self.device_map_for_quantization,
            pack=True,
        )
        self.assertEqual(self.quantized_model.transformer.h[0].mlp.dense_4h_to_h.__class__, QuantLinear)

    def check_quantized_layers_class(self, model):
        QuantLinear = hf_select_quant_linear_v2(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            sym=self.sym,
            format=FORMAT.GPTQ,
            quant_method=METHOD.GPTQ,
            device_map={"": self.device_for_inference},
            pack=False,
        )
        self.assertEqual(model.transformer.h[0].mlp.dense_4h_to_h.__class__, QuantLinear)

    def run_serialization_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.tokenizer.save_pretrained(tmpdirname)
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

            self.check_quantized_layers_class(quantized_model_from_saved)

            # Transformers and GPT-QModel compatibility.
            # quantized models are more compatible with device map than
            # device context managers (they're never used in transformers testing suite)
            _ = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": self.device_for_inference})
            _ = GPTQModel.load(tmpdirname, device_map={"": self.device_for_inference})

    def test_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights
        """

        self.run_serialization_round_trip()


class GPTQTestCPUInit(GPTQTest):
    device_map_for_quantization = "cpu"

    def test_perplexity(self):
        pass


class GPTQTestActOrder(GPTQTest):
    # `act_group_aware` == `True` requires `desc_act` == `False` when both are explicitly set
    desc_act = True
    act_group_aware = False
    expected_quantized_perplexity = 33

    def test_serialization(self):
        """
        Test the serialization and post-quant load flow for act-order models.
        """

        self.run_serialization_round_trip()


class GPTQTestNoBlockCaching(GPTQTest):
    cache_block_outputs = False


class GPTQTestModuleQuant(GPTQTest):
    # all layers are quantized apart from self_attention.dense
    modules_in_block_to_quantize = [
        ["self_attention.query_key_value"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
    expected_compression_ratio = 1.577

    def test_not_converted_layers(self):
        # self_attention.dense should not be converted
        self.assertEqual(self.quantized_model.transformer.h[0].self_attention.dense.__class__.__name__, "Linear")


@require_gptqmodel
class GPTQPostInitTest(unittest.TestCase):
    def test_post_init_model_with_real_quant_linear(self):
        quantizer = GPTQQuantizer(
            bits=4,
            dataset=["gptq"],
            desc_act=True,
            act_group_aware=False,
        )
        quantizer.quant_linear = hf_select_quant_linear_v2(
            bits=quantizer.bits,
            group_size=quantizer.group_size,
            desc_act=quantizer.desc_act,
            sym=quantizer.sym,
            format=FORMAT.GPTQ,
            quant_method=METHOD.GPTQ,
            device_map={"": "cpu"},
            pack=True,
        )

        class Wrapper(torch.nn.Module):
            # Minimal module tree that exercises the real GPT-QModel conversion and post-init path.
            def __init__(self):
                super().__init__()
                self.layer = quantizer.quant_linear(
                    bits=quantizer.bits,
                    group_size=quantizer.group_size,
                    sym=quantizer.sym,
                    desc_act=quantizer.desc_act,
                    in_features=32,
                    out_features=32,
                    bias=False,
                )

        model = Wrapper()
        self.assertEqual(model.layer.qzero_format(), 1)

        result = quantizer.post_init_model(model)

        self.assertIs(result, model)
        self.assertTrue(model.quantize_config.desc_act)
        self.assertEqual(model.layer.qzero_format(), 2)


@require_gptqmodel
class GPTQNativeLoadBridgeTest(unittest.TestCase):
    @patch("optimum.gptq.quantizer._gptqmodel_load_prepare_model")
    def test_load_context_is_scoped_per_model(self, prepare_model):
        quantizer = GPTQQuantizer(bits=4)
        original_quantize_config = quantizer.quantizeConfig
        first_model = torch.nn.Module()
        second_model = torch.nn.Module()
        first_context = SimpleNamespace(name="first")
        second_context = SimpleNamespace(name="second")
        prepare_model.side_effect = [first_context, second_context]

        first_result = quantizer.convert_model(
            first_model,
            checkpoint_files=["model.safetensors"],
            device_map={"": "cpu"},
            dtype=torch.float16,
        )
        second_result = quantizer.convert_model(
            second_model,
            checkpoint_files=["model.safetensors"],
            device_map={"": "cpu"},
            dtype=torch.float16,
        )

        self.assertIs(first_result, first_model)
        self.assertIs(second_result, second_model)
        self.assertIs(first_model._gptqmodel_load_context, first_context)
        self.assertIs(second_model._gptqmodel_load_context, second_context)
        self.assertFalse(hasattr(quantizer, "_gptqmodel_load_context"))
        self.assertIs(quantizer.quantizeConfig, original_quantize_config)
        self.assertFalse(hasattr(quantizer, "quant_linear"))
        prepare_model.assert_has_calls([
            call(
                first_model,
                checkpoint_files=["model.safetensors"],
                device_map={"": "cpu"},
                backend=quantizer.backend,
                dtype=torch.float16,
            ),
            call(
                second_model,
                checkpoint_files=["model.safetensors"],
                device_map={"": "cpu"},
                backend=quantizer.backend,
                dtype=torch.float16,
            ),
        ])

        with patch(
            "optimum.gptq.quantizer._gptqmodel_load_post_init",
            side_effect=lambda model, context: model,
        ) as post_init:
            self.assertIs(quantizer.post_init_model(first_model), first_model)
            self.assertIs(quantizer.post_init_model(second_model), second_model)

        self.assertEqual(
            post_init.call_args_list,
            [
                call(first_model, context=first_context),
                call(second_model, context=second_context),
            ],
        )
        self.assertFalse(hasattr(first_model, "_gptqmodel_load_context"))
        self.assertFalse(hasattr(second_model, "_gptqmodel_load_context"))


@slow
@require_gptqmodel
class GPTQNativeLoadBridgeIntegrationTest(unittest.TestCase):
    model_id = "ModelCloud/Phi-tiny-MoE-instruct-GPTQ-W4-MixedGroup-G32-G128"
    num_hidden_layers = 32
    num_local_experts = 16
    global_group_size = 128

    @classmethod
    def expected_quantized_modules(cls):
        # Q/K/V and expert gate/up use G32; output and expert down projections keep the global G128.
        expected = {}
        for layer_index in range(cls.num_hidden_layers):
            layer = f"model.layers.{layer_index}"
            for projection in ("q_proj", "k_proj", "v_proj"):
                expected[f"{layer}.self_attn.{projection}"] = (4, 32)
            expected[f"{layer}.self_attn.o_proj"] = (4, cls.global_group_size)
            for expert_index in range(cls.num_local_experts):
                expert = f"{layer}.mlp.experts.{expert_index}"
                expected[f"{expert}.gate_proj"] = (4, 32)
                expected[f"{expert}.up_proj"] = (4, 32)
                expected[f"{expert}.down_proj"] = (4, cls.global_group_size)
        return expected

    @classmethod
    def load_model(cls, model_id):
        # Use the generic CPU kernel so this test isolates loading rather than optimized-kernel shape limits.
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": "cpu"},
            dtype=torch.float16,
            quantization_config=GPTQConfig(bits=4, backend="torch"),
        )

    @staticmethod
    def checkpoint_tensor_names(model_id):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        # Resolve the public Hub fixture through the standard cache; no machine-local fixture path is required.
        checkpoint_file = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        with safe_open(checkpoint_file, framework="pt", device="cpu") as checkpoint:
            return set(checkpoint.keys())

    def test_all_dynamic_moe_projections_load_correctly_only_with_delegate(self):
        from optimum.gptq import quantizer as optimum_quantizer

        if optimum_quantizer._gptqmodel_load_prepare_model is None:
            self.skipTest("requires the GPTQModel native load delegate")

        expected_quantized_modules = self.expected_quantized_modules()
        expected_dynamic = {
            f"+:^{re.escape(name)}$": {"bits": bits, "group_size": group_size}
            for name, (bits, group_size) in expected_quantized_modules.items()
            if group_size != self.global_group_size
        }
        self.assertEqual(len(expected_quantized_modules), 1664)
        self.assertEqual(len(expected_dynamic), 1120)
        expected_dense_linear_modules = {
            *(f"model.layers.{layer_index}.mlp.router" for layer_index in range(self.num_hidden_layers)),
            "lm_head",
        }

        # Check tensors on disk so incomplete weights cannot pass by exposing only correct config metadata.
        checkpoint_tensor_names = self.checkpoint_tensor_names(self.model_id)
        for component in ("qweight", "qzeros", "scales", "g_idx"):
            suffix = f".{component}"
            actual_names = {
                name.removesuffix(suffix) for name in checkpoint_tensor_names if name.endswith(suffix)
            }
            self.assertEqual(actual_names, set(expected_quantized_modules))

        # The delegate must rebuild every per-expert module and preserve its mixed group size.
        model = self.load_model(self.model_id)
        modules = dict(model.named_modules())
        actual_quantized_modules = {
            name: (module.bits, module.group_size)
            for name, module in modules.items()
            if isinstance(module, BaseQuantLinear)
        }
        self.assertEqual(actual_quantized_modules, expected_quantized_modules)

        actual_dense_linear_modules = {
            name for name, module in modules.items() if isinstance(module, torch.nn.Linear)
        }
        self.assertEqual(actual_dense_linear_modules, expected_dense_linear_modules)
        self.assertEqual(model.config.quantization_config.group_size, self.global_group_size)
        self.assertEqual(model.config.quantization_config.dynamic, expected_dynamic)
        self.assertEqual(
            [
                name
                for name, tensor in (*model.named_parameters(), *model.named_buffers())
                if tensor.is_meta
            ],
            [],
        )

        with torch.inference_mode():
            output = model(input_ids=torch.tensor([[1, 42, 314, 2718, 7, 11]], dtype=torch.long))
        self.assertEqual(output.logits.shape, (1, 6, 32064))
        self.assertTrue(torch.isfinite(output.logits).all())

        del model
        gc.collect()
        # The legacy path returns a model but silently loses G32 overrides and per-expert modules.
        with patch("optimum.gptq.quantizer._gptqmodel_load_prepare_model", None):
            legacy_model = self.load_model(self.model_id)

        legacy_modules = dict(legacy_model.named_modules())
        legacy_q_proj = legacy_modules["model.layers.0.self_attn.q_proj"]
        self.assertEqual((legacy_q_proj.bits, legacy_q_proj.group_size), (4, self.global_group_size))
        self.assertNotIn("model.layers.0.mlp.experts.0.down_proj", legacy_modules)


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
