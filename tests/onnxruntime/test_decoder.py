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
import gc
import os
import tempfile
from pathlib import Path

import pytest
import torch
from onnxruntime import InferenceSession
from parameterized import parameterized
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import AutoModelForCausalLM, set_seed
from transformers.generation import GenerationConfig
from transformers.onnx.utils import get_preprocessor

from optimum.exporters import TasksManager
from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from optimum.onnx.utils import has_onnx_input
from optimum.onnxruntime import (
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_WEIGHTS_NAME,
    ORTModelForCausalLM,
)
from optimum.pipelines import pipeline
from optimum.utils import logging
from optimum.utils.import_utils import is_transformers_version
from optimum.utils.testing_utils import grid_parameters


logger = logging.get_logger()


class ORTModelForCausalLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "bloom",
        "codegen",
        "falcon",
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "gptj",
        "llama",
        "mistral",
        "opt",
    ]

    if is_transformers_version(">=", "4.37"):
        SUPPORTED_ARCHITECTURES.append("qwen2")

    if is_transformers_version(">=", "4.38"):
        SUPPORTED_ARCHITECTURES.append("gemma")

    if is_transformers_version(">=", "4.41"):
        # TODO: fix "mpt" for which inference fails for transformers < v4.41
        SUPPORTED_ARCHITECTURES.append("mpt")

    if is_transformers_version(">=", "4.45"):
        SUPPORTED_ARCHITECTURES.append("granite")

    if is_transformers_version(">=", "4.50"):
        SUPPORTED_ARCHITECTURES.append("phi3")

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [False, True]}

    AUTOMODEL_CLASS = AutoModelForCausalLM
    ORTMODEL_CLASS = ORTModelForCausalLM
    TASK = "text-generation"
    GENERATION_LENGTH = 100

    def get_inputs(self, batch_size=1):
        return ["This is a sample input"] + ["This is another sample input"] * (batch_size - 1)

    @parameterized.expand([(False,), (True,)])
    @pytest.mark.run_in_series
    def test_inference_with_old_onnx_model(self, use_cache):
        tokenizer = get_preprocessor("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        ort_model = self.ORTMODEL_CLASS.from_pretrained("optimum/gpt2", use_cache=use_cache, use_io_binding=use_cache)

        self.assertEqual(ort_model.use_cache, use_cache)
        self.assertEqual(ort_model.model_path.name, ONNX_DECODER_WITH_PAST_NAME if use_cache else ONNX_DECODER_NAME)

        inputs = self.get_inputs(batch_size=2)
        tokens = tokenizer(inputs, return_tensors="pt")

        onnx_outputs = ort_model.generate(**tokens, min_new_tokens=30, max_new_tokens=30, do_sample=False)
        outputs = model.generate(**tokens, min_new_tokens=30, max_new_tokens=30, do_sample=False)
        onnx_text_outputs = tokenizer.decode(onnx_outputs[0], skip_special_tokens=True)
        text_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertEqual(onnx_text_outputs, text_outputs)

    def test_load_model_from_hub_onnx(self):
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

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["vit"], export=True)
        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        ort_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        tokens = tokenizer("This is a sample input", return_tensors="pt")

        position_ids = None
        if model_arch.replace("_", "-") in MODEL_TYPES_REQUIRING_POSITION_IDS:
            position_ids = torch.arange(0, tokens["input_ids"].shape[1], dtype=torch.long).view(1, -1)

        outputs = model(**tokens, position_ids=position_ids)
        onnx_outputs = ort_model(**tokens, position_ids=position_ids)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)
        torch.testing.assert_close(onnx_outputs.logits, outputs.logits, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_generation_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        ort_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        inputs = self.get_inputs()
        tokens = tokenizer(inputs, return_tensors="pt")
        gen_kwargs = {"max_new_tokens": 10, "min_new_tokens": 10, "do_sample": False}

        outputs = model.generate(**tokens, **gen_kwargs)
        onnx_outputs = ort_model.generate(**tokens, **gen_kwargs)
        torch.testing.assert_close(outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(grid_parameters({**FULL_GRID, "num_beams": [1, 4]}))
    def test_compare_beam_search_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, num_beams: int
    ):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        ort_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        inputs = self.get_inputs()
        tokens = tokenizer(inputs, return_tensors="pt")
        gen_kwargs = {"max_new_tokens": 10, "min_new_tokens": 10, "num_beams": num_beams}
        gen_configs = [GenerationConfig(do_sample=False, **gen_kwargs), GenerationConfig(do_sample=True, **gen_kwargs)]
        if num_beams == 4:
            gen_configs.append(
                GenerationConfig(do_sample=False, num_beam_groups=2, diversity_penalty=0.0000001, **gen_kwargs)
            )

        for gen_config in gen_configs:
            with torch.no_grad():
                set_seed(SEED)
                outputs = model.generate(**tokens, generation_config=gen_config)
            set_seed(SEED)
            onnx_outputs = ort_model.generate(**tokens, generation_config=gen_config)
            torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

    def test_pipeline_with_none(self):
        pipe = pipeline("text-generation")
        text = "The capital of France is"
        outputs = pipe(text)

        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    @parameterized.expand(grid_parameters({"model_arch": ["llama"], "use_cache": [True, False]}))
    def test_pipeline_with_ort_model(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        ort_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        pipe = pipeline("text-generation", model=ort_model, tokenizer=tokenizer)
        text = "The capital of France is"
        outputs = pipe(text)

        self.assertEqual(pipe.device, ort_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": ["llama"], "use_cache": [True, False]}))
    def test_pipeline_with_hub_model(self, test_name: str, model_arch: str, use_cache: bool):
        hub_model = "optimum-internal-testing/tiny-random-llama"

        pipe = pipeline("text-generation", model=hub_model, revision="onnx", accelerator="ort")
        text = "The capital of France is"
        outputs = pipe(text)

        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            model_kwargs = {"use_cache": use_cache}
            pipe = pipeline("text-generation", model=tmpdir, model_kwargs=model_kwargs, accelerator="ort")
            outputs_local_model = pipe(text)
            self.assertEqual(outputs[0]["generated_text"], outputs_local_model[0]["generated_text"])

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        text = "My Name is Philipp and i live"
        tokens = tokenizer(text, return_tensors="pt", return_token_type_ids=False if model_arch == "llama" else None)

        new_tokens = 10  # some models have a short max length

        model_with_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        outputs_model_with_pkv = model_with_pkv.generate(
            **tokens, min_new_tokens=new_tokens, max_new_tokens=new_tokens
        )

        model_without_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        outputs_model_without_pkv = model_without_pkv.generate(
            **tokens, min_new_tokens=new_tokens, max_new_tokens=new_tokens
        )

        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, atol=self.ATOL, rtol=self.RTOL)
        self.assertEqual(outputs_model_without_pkv.shape[1], tokens["input_ids"].shape[1] + new_tokens)
        self.assertEqual(outputs_model_with_pkv.shape[1], tokens["input_ids"].shape[1] + new_tokens)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        ort_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=False, provider="CPUExecutionProvider"
        )
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=True, provider="CPUExecutionProvider"
        )

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")

        position_ids = None
        if model_arch.replace("_", "-") in MODEL_TYPES_REQUIRING_POSITION_IDS:
            input_shape = tokens["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).expand(2, input_shape[-1])

        onnx_outputs = ort_model(**tokens, position_ids=position_ids)
        io_outputs = io_model(**tokens, position_ids=position_ids)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        torch.testing.assert_close(io_outputs.logits, onnx_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_generation_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        ort_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], provider="CPUExecutionProvider", use_io_binding=False, use_cache=use_cache
        )
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name],
            provider="CPUExecutionProvider",
            use_io_binding=True,
            use_cache=use_cache,
        )

        self.assertEqual(ort_model.use_cache, use_cache)
        self.assertEqual(io_model.use_cache, use_cache)
        self.assertFalse(ort_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokens = tokenizer("This is a sample output", return_tensors="pt")

        io_outputs = io_model.generate(**tokens)
        ort_outputs = ort_model.generate(**tokens)
        torch.testing.assert_close(io_outputs, ort_outputs, atol=self.ATOL, rtol=self.RTOL)

    # TODO: remove once legacy export is removed
    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = "text-generation-with-past"
            model_id = MODEL_NAMES[model_arch]
            main_export(model_id, tmpdir, task=task, legacy=True)

            # all models should be there except the non-legacy one
            folder_contents = os.listdir(tmpdir)
            self.assertIn(ONNX_DECODER_NAME, folder_contents)
            self.assertIn(ONNX_DECODER_WITH_PAST_NAME, folder_contents)
            self.assertIn(ONNX_DECODER_MERGED_NAME, folder_contents)
            self.assertNotIn(ONNX_WEIGHTS_NAME, folder_contents)
            decoder_merged_path = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(decoder_merged_path, "use_cache_branch"))
            model = self.ORTMODEL_CLASS.from_pretrained(tmpdir)
            self.assertTrue(model.use_merged)

            # only saves the merged model
            model.save_pretrained(tmpdir + "_save")
            folder_contents = os.listdir(tmpdir + "_save")
            self.assertNotIn(ONNX_DECODER_NAME, folder_contents)
            self.assertNotIn(ONNX_DECODER_WITH_PAST_NAME, folder_contents)
            self.assertNotIn(ONNX_WEIGHTS_NAME, folder_contents)
            self.assertIn(ONNX_DECODER_MERGED_NAME, folder_contents)

    # TODO: remove once legacy export is removed
    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_merged_and_not_merged_models_outputs(self, model_arch: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = "text-generation-with-past"
            model_id = MODEL_NAMES[model_arch]
            tokenizer = get_preprocessor(model_id)
            text = "My Name is Philipp and i live"
            tokens = tokenizer(text, return_tensors="pt")
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
            main_export(model_id, output=tmpdir, task=task, legacy=True)

            # not merged model
            self.assertIn(ONNX_DECODER_WITH_PAST_NAME, os.listdir(tmpdir))
            not_merged_file = os.path.join(tmpdir, ONNX_DECODER_WITH_PAST_NAME)
            self.assertFalse(has_onnx_input(not_merged_file, "use_cache_branch"))
            not_merged_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir, use_cache=False, use_merged=False)
            self.assertFalse(not_merged_model.use_merged)
            self.assertFalse(not_merged_model.use_cache)

            # merged model
            self.assertIn(ONNX_DECODER_MERGED_NAME, os.listdir(tmpdir))
            merged_file = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(merged_file, "use_cache_branch"))
            merged_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir, use_cache=True, use_merged=True)
            self.assertTrue(merged_model.use_merged)
            self.assertTrue(merged_model.use_cache)

            gen_kwargs = {"max_new_tokens": 10, "min_new_tokens": 10, "do_sample": False}
            not_merged_outputs = not_merged_model.generate(**tokens, **gen_kwargs)
            merged_outputs = merged_model.generate(**tokens, **gen_kwargs)
            outputs = model.generate(**tokens, **gen_kwargs)

            # compare merged to transformers
            torch.testing.assert_close(outputs, merged_outputs, atol=self.ATOL, rtol=self.RTOL)
            # compare not merged to transformers
            torch.testing.assert_close(outputs, not_merged_outputs, atol=self.ATOL, rtol=self.RTOL)
            # compare merged to not merged
            torch.testing.assert_close(merged_outputs, not_merged_outputs, atol=self.ATOL, rtol=self.RTOL)
