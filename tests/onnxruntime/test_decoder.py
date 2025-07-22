# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import os
import tempfile
import unittest

import torch
from onnxruntime import InferenceSession
from parameterized import parameterized
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.generation import GenerationConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.onnx.utils import get_preprocessor

from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import (
    BloomOnnxConfig,
    GemmaOnnxConfig,
    GraniteOnnxConfig,
    InternLM2OnnxConfig,
    MPTOnnxConfig,
    Olmo2OnnxConfig,
    OlmoOnnxConfig,
    OPTOnnxConfig,
    Phi3OnnxConfig,
    PhiOnnxConfig,
    Qwen2OnnxConfig,
    Qwen3MoeOnnxConfig,
    Qwen3OnnxConfig,
    SmolLM3OnnxConfig,
)
from optimum.exporters.tasks import TasksManager
from optimum.onnx.utils import has_onnx_input
from optimum.onnxruntime import (
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_WEIGHTS_NAME,
    ORTModelForCausalLM,
)
from optimum.pipelines import pipeline
from optimum.utils.import_utils import is_transformers_version
from optimum.utils.logging import get_logger
from optimum.utils.testing_utils import grid_parameters, remove_directory, require_hf_token


logger = get_logger(__name__)


class ORTModelForCausalLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "codegen",
        "falcon",
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "gptj",
        "llama",
        "mistral",
        "bart",
        "blenderbot-small",
        "bigbird_pegasus",
        "marian",
        "pegasus",
        "blenderbot",
        "mbart",
    ]

    if is_transformers_version(">=", str(OPTOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("opt")
    if is_transformers_version(">=", str(PhiOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("phi")
    if is_transformers_version(">=", str(BloomOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("bloom")
    if is_transformers_version(">=", str(OlmoOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("olmo")
    if is_transformers_version(">=", str(Olmo2OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("olmo2")
    if is_transformers_version(">=", str(Qwen2OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("qwen2")
    if is_transformers_version(">=", str(GemmaOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("gemma")
    if is_transformers_version(">=", str(MPTOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("mpt")
    if is_transformers_version(">=", str(GraniteOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("granite")
    if is_transformers_version(">=", str(Phi3OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("phi3")
    if is_transformers_version(">=", str(Qwen3OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("qwen3")
    if is_transformers_version(">=", str(Qwen3MoeOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("qwen3_moe")
    if is_transformers_version(">=", str(InternLM2OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("internlm2")
    if is_transformers_version(">=", str(SmolLM3OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("smollm3")

    GEN_KWARGS = {"max_new_tokens": 10, "min_new_tokens": 10, "do_sample": False, "num_beams": 1}
    BEAM_KWARGS = {"max_new_tokens": 3, "min_new_tokens": 3, "num_beams": 4}

    MODEL_TRUST_REMOTE_CODE = {"internlm2"}
    TASK = "text-generation"
    ORTMODEL_CLASS = ORTModelForCausalLM
    AUTOMODEL_CLASS = AutoModelForCausalLM

    def get_inputs(self, batch_size=1):
        return ["This is a sample input"] + ["This is another sample input"] * (batch_size - 1)

    # INTEGRATION TESTS
    def test_find_untested_architectures(self):
        if len(self.SUPPORTED_ARCHITECTURES) != len(set(self.SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"For the task `{self.TASK}`, some architectures are duplicated in the list of tested architectures: "
                f"{self.SUPPORTED_ARCHITECTURES}.\n"
            )

        tested_architectures = set(self.SUPPORTED_ARCHITECTURES)
        transformers_architectures = set(CONFIG_MAPPING_NAMES.keys())
        onnx_architectures = set(TasksManager.get_supported_model_type_for_task(task=self.TASK, exporter="onnx"))
        supported_architectures = onnx_architectures & transformers_architectures
        untested_architectures = supported_architectures - tested_architectures

        if len(untested_architectures) > 0:
            raise ValueError(
                f"For the task `{self.TASK}`, the ONNX exporter supports {supported_architectures} but some of them are not "
                f"tested: {untested_architectures}.\n"
            )

    def test_load_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["vit"], export=True)
        self.assertIn("only supports the tasks", str(context.exception))

    def test_load_model_from_hub(self):
        model = self.ORTMODEL_CLASS.from_pretrained("fxmarty/onnx-tiny-random-gpt2-without-merge")
        self.assertEqual(model.path.name, ONNX_DECODER_WITH_PAST_NAME)
        self.assertIsInstance(model.session, InferenceSession)
        self.assertFalse(model.use_merged)
        self.assertTrue(model.use_cache)

        model = self.ORTMODEL_CLASS.from_pretrained("fxmarty/onnx-tiny-random-gpt2-with-merge")
        self.assertEqual(model.path.name, ONNX_DECODER_MERGED_NAME)
        self.assertIsInstance(model.session, InferenceSession)
        self.assertTrue(model.use_merged)
        self.assertTrue(model.use_cache)

        model = self.ORTMODEL_CLASS.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        self.assertEqual(model.path.name, ONNX_WEIGHTS_NAME)
        self.assertIsInstance(model.session, InferenceSession)
        self.assertFalse(model.use_merged)
        self.assertTrue(model.use_cache)

    @parameterized.expand(
        grid_parameters({"use_cache": [False, True], "use_merged": [False, True]}, add_test_name=False)
    )
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_load_model_with_external_data(self, use_cache: bool, use_merged: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_id = MODEL_NAMES["gpt2"]
            # export=True because there's a folder with onnx model in hf-internal-testing/tiny-random-GPT2LMHeadModel
            model = self.ORTMODEL_CLASS.from_pretrained(
                model_id, use_cache=use_cache, use_merged=use_merged, export=True
            )
            model.save_pretrained(tmpdirname)
            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(ONNX_WEIGHTS_NAME + "_data" in folder_contents)
            # verify loading from local folder works
            model = self.ORTMODEL_CLASS.from_pretrained(tmpdirname, use_cache=use_cache, use_merged=use_merged)
            model.generate(**self.GEN_KWARGS)
            remove_directory(tmpdirname)

    @require_hf_token
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_push_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_id = MODEL_NAMES["gpt2"]
            repo_dir = model_id.split("/")[-1] + "-onnx"
            token = os.environ.get("HF_AUTH_TOKEN", None)
            model = ORTModelForCausalLM.from_pretrained(model_id, export=True)
            # verify the model can be pushed to the hub
            model.save_pretrained(tmpdirname, token=token, repository_id=repo_dir, push_to_hub=True)
            # verify pulling from hub works
            model = ORTModelForCausalLM.from_pretrained(repo_dir, token=token, export=False)
            model.generate(**self.GEN_KWARGS)
            remove_directory(tmpdirname)

    def test_trust_remote_code(self):
        model_id = "optimum-internal-testing/tiny-testing-gpt2-remote-code"

        inputs = self.get_inputs()
        tokenizer = get_preprocessor(model_id)
        inputs = tokenizer(inputs, return_tensors="pt")

        model = self.AUTOMODEL_CLASS.from_pretrained(model_id, trust_remote_code=True).eval()
        ort_model = self.ORTMODEL_CLASS.from_pretrained(model_id, export=True, trust_remote_code=True)

        pt_logits = model(**inputs).logits
        ort_logits = ort_model(**inputs).logits
        torch.testing.assert_close(pt_logits, ort_logits, atol=self.ATOL, rtol=self.RTOL)

    def test_load_model_from_hub_infer_onnx_model(self):
        model_id = "optimum-internal-testing/tiny-random-llama"
        # export from hub
        model = self.ORTMODEL_CLASS.from_pretrained(model_id)
        self.assertEqual(model.path.name, "model.onnx")
        # load from hub
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx")
        self.assertEqual(model.path.name, "model.onnx")
        # load from hub with revision
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx")
        self.assertEqual(model.path.name, "decoder_model_merged.onnx")
        # load from hub with revision and subfolder
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx", subfolder="subfolder")
        self.assertEqual(model.path.name, "model.onnx")
        # load from hub with revision and file_name
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx", file_name="model_optimized.onnx")
        self.assertEqual(model.path.name, "model_optimized.onnx")
        # revision + file_name (decoder with past)
        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", file_name="decoder_with_past_model.onnx"
        )
        self.assertEqual(model.path.name, "decoder_with_past_model.onnx")

        # TODO: something went wrong here
        # revision + subfolder + file_name (target file exists but it loaded a different one)
        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", subfolder="subfolder", file_name="optimized_model.onnx"
        )
        self.assertEqual(model.path.name, "model.onnx")

        with self.assertRaises(FileNotFoundError):
            self.ORTMODEL_CLASS.from_pretrained(
                "hf-internal-testing/tiny-random-LlamaForCausalLM", file_name="doesnt_exist.onnx"
            )

    # SANITY TESTS
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.MODEL_TRUST_REMOTE_CODE
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        set_seed(SEED)
        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")
        model = self.AUTOMODEL_CLASS.from_pretrained(
            model_id, use_cache=use_cache, trust_remote_code=trust_remote_code
        ).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, trust_remote_code=trust_remote_code
        )

        self.assertEqual(onnx_model.use_cache, use_cache)
        self.assertEqual(model.config.use_cache, use_cache)

        outputs = model(**tokens)
        onnx_outputs = onnx_model(**tokens)
        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)
        torch.testing.assert_close(onnx_outputs.logits, outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        if use_cache:
            self.assertTrue("past_key_values" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.past_key_values, tuple)
            for i in range(len(onnx_outputs.past_key_values)):
                if model_arch == "gpt_bigcode":
                    self.assertIsInstance(onnx_outputs.past_key_values[i], torch.Tensor)
                    torch.testing.assert_close(
                        onnx_outputs.past_key_values[i],
                        outputs.past_key_values[i],
                        atol=self.ATOL,
                        rtol=self.RTOL,
                    )
                else:
                    for j in range(len(onnx_outputs.past_key_values[i])):
                        self.assertIsInstance(onnx_outputs.past_key_values[i][j], torch.Tensor)
                        torch.testing.assert_close(
                            onnx_outputs.past_key_values[i][j],
                            outputs.past_key_values[i][j],
                            atol=self.ATOL,
                            rtol=self.RTOL,
                        )

    # generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_generation_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.MODEL_TRUST_REMOTE_CODE
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        set_seed(SEED)
        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")

        model = self.AUTOMODEL_CLASS.from_pretrained(
            model_id, use_cache=use_cache, trust_remote_code=trust_remote_code
        ).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, trust_remote_code=trust_remote_code
        )
        self.assertEqual(model.config.use_cache, use_cache)
        self.assertEqual(onnx_model.use_cache, use_cache)

        outputs = model.generate(**tokens, **self.GEN_KWARGS)
        onnx_outputs = onnx_model.generate(**tokens, **self.GEN_KWARGS)
        torch.testing.assert_close(outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    # beam search is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_beam_search_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.MODEL_TRUST_REMOTE_CODE
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        set_seed(SEED)
        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")
        model = self.AUTOMODEL_CLASS.from_pretrained(
            model_id, use_cache=use_cache, trust_remote_code=trust_remote_code
        ).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, trust_remote_code=trust_remote_code
        )
        self.assertEqual(model.config.use_cache, use_cache)
        self.assertEqual(onnx_model.use_cache, use_cache)

        # beam search with random sampling
        gen_config = GenerationConfig(**self.BEAM_KWARGS, do_sample=True)
        set_seed(SEED)
        outputs = model.generate(**tokens, generation_config=gen_config)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**tokens, generation_config=gen_config)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

        # group beam search with diversity penalty
        model.generation_config.do_sample = False  # some models have hardcoded generation configs
        onnx_model.generation_config.do_sample = False  # some models have hardcoded generation configs
        gen_config = GenerationConfig(
            **self.BEAM_KWARGS,
            diversity_penalty=0.0001,
            num_beam_groups=2,
            do_sample=False,
        )
        outputs = model.generate(**tokens, generation_config=gen_config)
        onnx_outputs = onnx_model.generate(**tokens, generation_config=gen_config)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch):
        trust_remote_code = model_arch in self.MODEL_TRUST_REMOTE_CODE
        model_args = {
            "test_name": model_arch + "_False",
            "model_arch": model_arch,
            "use_cache": False,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)
        model_args = {
            "test_name": model_arch + "_True",
            "model_arch": model_arch,
            "use_cache": True,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokens = tokenizer(inputs, return_tensors="pt")
        with_pkv_dir = self.onnx_model_dirs[model_arch + "_True"]
        without_pkv_dir = self.onnx_model_dirs[model_arch + "_False"]
        model_with_pkv = self.ORTMODEL_CLASS.from_pretrained(
            with_pkv_dir, use_cache=True, trust_remote_code=trust_remote_code
        )
        model_without_pkv = self.ORTMODEL_CLASS.from_pretrained(
            without_pkv_dir, use_cache=False, trust_remote_code=trust_remote_code
        )
        self.assertFalse(model_without_pkv.use_cache)
        self.assertTrue(model_with_pkv.use_cache)

        new_tokens = self.GEN_KWARGS["max_new_tokens"]
        outputs_model_with_pkv = model_with_pkv.generate(**tokens, **self.GEN_KWARGS)
        outputs_model_without_pkv = model_without_pkv.generate(**tokens, **self.GEN_KWARGS)
        self.assertEqual(outputs_model_with_pkv.shape[1], tokens["input_ids"].shape[1] + new_tokens)
        self.assertEqual(outputs_model_without_pkv.shape[1], tokens["input_ids"].shape[1] + new_tokens)
        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, atol=self.ATOL, rtol=self.RTOL)

    # TODO: remove when io binding is the default
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.MODEL_TRUST_REMOTE_CODE
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokens = tokenizer(inputs, return_tensors="pt")
        model_dir = self.onnx_model_dirs[test_name]
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            model_dir,
            use_cache=use_cache,
            use_io_binding=False,
            provider="CPUExecutionProvider",
            trust_remote_code=trust_remote_code,
        )
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            model_dir,
            use_cache=use_cache,
            use_io_binding=True,
            provider="CPUExecutionProvider",
            trust_remote_code=trust_remote_code,
        )

        self.assertTrue(io_model.use_io_binding)
        self.assertFalse(onnx_model.use_io_binding)
        self.assertEqual(io_model.use_cache, use_cache)
        self.assertEqual(onnx_model.use_cache, use_cache)

        io_outputs = io_model(**tokens, **self.GEN_KWARGS)
        onnx_outputs = onnx_model(**tokens, **self.GEN_KWARGS)
        torch.testing.assert_close(io_outputs.logits, onnx_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        if use_cache:
            self.assertTrue("past_key_values" in io_outputs)
            self.assertIsInstance(io_outputs.past_key_values, tuple)
            for i in range(len(io_outputs.past_key_values)):
                if model_arch == "gpt_bigcode":
                    self.assertIsInstance(io_outputs.past_key_values[i], torch.Tensor)
                    torch.testing.assert_close(
                        io_outputs.past_key_values[i],
                        onnx_outputs.past_key_values[i],
                        atol=self.ATOL,
                        rtol=self.RTOL,
                    )
                else:
                    for j in range(len(io_outputs.past_key_values[i])):
                        self.assertIsInstance(io_outputs.past_key_values[i][j], torch.Tensor)
                        torch.testing.assert_close(
                            io_outputs.past_key_values[i][j],
                            onnx_outputs.past_key_values[i][j],
                            atol=self.ATOL,
                            rtol=self.RTOL,
                        )

    # generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_generation_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.MODEL_TRUST_REMOTE_CODE
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")

        model_dir = self.onnx_model_dirs[test_name]
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            model_dir,
            provider="CPUExecutionProvider",
            use_io_binding=False,
            use_cache=use_cache,
            trust_remote_code=trust_remote_code,
        )
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            model_dir,
            provider="CPUExecutionProvider",
            use_io_binding=True,
            use_cache=use_cache,
            trust_remote_code=trust_remote_code,
        )

        self.assertTrue(io_model.use_io_binding)
        self.assertFalse(onnx_model.use_io_binding)
        self.assertEqual(io_model.use_cache, use_cache)
        self.assertEqual(onnx_model.use_cache, use_cache)

        io_outputs = io_model.generate(**tokens, **self.GEN_KWARGS)
        onnx_outputs = onnx_model.generate(**tokens, **self.GEN_KWARGS)
        torch.testing.assert_close(io_outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    # PIPELINE TESTS
    def test_pipeline_with_none(self):
        pipe = pipeline("text-generation")
        text = "The capital of France is"
        set_seed(SEED)
        outputs = pipe(text)

        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline("text-generation", model=tmpdir)
            set_seed(SEED)
            outputs_local_model = pipe(text)
            self.assertEqual(outputs[0]["generated_text"], outputs_local_model[0]["generated_text"])

    @parameterized.expand(grid_parameters({"model_arch": ["llama"], "use_cache": [True, False]}))
    def test_pipeline_with_ort_model(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        text = "The capital of France is"
        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)

        set_seed(SEED)
        outputs = pipe(text)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline("text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache})
            set_seed(SEED)
            local_pipe_outputs = pipe(text)
            self.assertEqual(outputs[0]["generated_text"], local_pipe_outputs[0]["generated_text"])

    @parameterized.expand(grid_parameters({"model_arch": ["llama"], "use_cache": [True, False]}))
    def test_pipeline_with_hub_model_id(self, test_name: str, model_arch: str, use_cache: bool):
        text = "The capital of France is"
        model_id = MODEL_NAMES[model_arch]
        pipe = pipeline("text-generation", model=model_id, accelerator="ort", model_kwargs={"use_cache": use_cache})

        set_seed(SEED)
        outputs = pipe(text)

        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline("text-generation", model=tmpdir, accelerator="ort", model_kwargs={"use_cache": use_cache})
            set_seed(SEED)
            outputs_local_model = pipe(text)
            self.assertEqual(outputs[0]["generated_text"], outputs_local_model[0]["generated_text"])

    @parameterized.expand([(False,), (True,)])
    def test_inference_with_old_onnx_model(self, use_cache):
        tokenizer = get_preprocessor("gpt2")
        inputs = self.get_inputs(batch_size=2)
        tokens = tokenizer(inputs, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        onnx_model = self.ORTMODEL_CLASS.from_pretrained("optimum/gpt2", use_cache=use_cache)

        self.assertEqual(onnx_model.use_cache, use_cache)
        self.assertEqual(onnx_model.path.name, ONNX_DECODER_WITH_PAST_NAME if use_cache else ONNX_DECODER_NAME)

        outputs = model.generate(**tokens, **self.GEN_KWARGS)
        onnx_outputs = onnx_model.generate(**tokens, **self.GEN_KWARGS)
        torch.testing.assert_close(outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    # TODO: remove once legacy export is removed
    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_merged_and_not_merged_models_outputs(self, model_arch: str):
        trust_remote_code = model_arch in self.MODEL_TRUST_REMOTE_CODE

        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(SEED)
            inputs = self.get_inputs()
            task = "text-generation-with-past"
            model_id = MODEL_NAMES[model_arch]
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            tokens = tokenizer(inputs, return_tensors="pt")
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

            set_seed(SEED)
            main_export(
                model_id,
                output=tmpdir,
                task=task,
                legacy=True,
                no_post_processing=False,
                do_validation=False,
                trust_remote_code=trust_remote_code,
            )

            # not merged model, without cache
            self.assertIn(ONNX_DECODER_NAME, os.listdir(tmpdir))
            not_merged_without_cache_file = os.path.join(tmpdir, ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(not_merged_without_cache_file, "use_cache_branch"))
            not_merged_without_cache_model = self.ORTMODEL_CLASS.from_pretrained(
                tmpdir,
                use_cache=False,
                use_merged=False,
                trust_remote_code=trust_remote_code,
            )
            self.assertFalse(not_merged_without_cache_model.generation_config.use_cache)
            self.assertFalse(not_merged_without_cache_model.config.use_cache)
            self.assertFalse(not_merged_without_cache_model.use_merged)
            self.assertFalse(not_merged_without_cache_model.use_cache)

            # merged model
            self.assertIn(ONNX_DECODER_MERGED_NAME, os.listdir(tmpdir))
            merged_file = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(merged_file, "use_cache_branch"))
            merged_model = self.ORTMODEL_CLASS.from_pretrained(
                tmpdir, use_merged=True, trust_remote_code=trust_remote_code
            )
            self.assertTrue(merged_model.generation_config.use_cache)
            self.assertTrue(merged_model.config.use_cache)
            self.assertTrue(merged_model.use_merged)
            self.assertTrue(merged_model.use_cache)

            # forward
            logits = model(**tokens).logits
            merged_logits = merged_model(**tokens).logits
            not_merged_without_cache_logits = not_merged_without_cache_model(**tokens).logits

            # compare merged to transformers
            torch.testing.assert_close(logits, merged_logits, atol=self.ATOL, rtol=self.RTOL)
            # compare not merged without cache to transformers
            torch.testing.assert_close(logits, not_merged_without_cache_logits, atol=self.ATOL, rtol=self.RTOL)

            # generate
            outputs = model.generate(**tokens, **self.GEN_KWARGS)
            merged_outputs = merged_model.generate(**tokens, **self.GEN_KWARGS)
            not_merged_without_cache_outputs = not_merged_without_cache_model.generate(**tokens, **self.GEN_KWARGS)

            # compare merged to transformers
            torch.testing.assert_close(outputs, merged_outputs, atol=self.ATOL, rtol=self.RTOL)
            # compare not merged without cache to transformers
            torch.testing.assert_close(outputs, not_merged_without_cache_outputs, atol=self.ATOL, rtol=self.RTOL)

            # load and save (only merged is loaded and saved)
            loaded_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir, trust_remote_code=trust_remote_code)
            self.assertEqual(loaded_model.path.name, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(loaded_model.use_merged)
            self.assertTrue(loaded_model.use_cache)
            save_dir = os.path.join(tmpdir, "save")
            loaded_model.save_pretrained(save_dir)
            self.assertNotIn(ONNX_DECODER_NAME, os.listdir(save_dir))
            self.assertNotIn(ONNX_DECODER_WITH_PAST_NAME, os.listdir(save_dir))
            self.assertIn(ONNX_DECODER_MERGED_NAME, os.listdir(save_dir))
            reloaded_model = self.ORTMODEL_CLASS.from_pretrained(save_dir, trust_remote_code=trust_remote_code)
            self.assertEqual(reloaded_model.path.name, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(reloaded_model.use_merged)
            self.assertTrue(reloaded_model.use_cache)
