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

import torch
from onnxruntime import InferenceSession
from parameterized import parameterized
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import AutoModelForCausalLM, set_seed
from transformers.generation import GenerationConfig
from transformers.onnx.utils import get_preprocessor

from optimum.exporters.onnx import main_export
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
from optimum.utils.testing_utils import grid_parameters


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

    GEN_KWARGS = {"max_new_tokens": 10, "min_new_tokens": 10, "do_sample": False, "num_beams": 1}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [False, True]}
    AUTOMODEL_CLASS = AutoModelForCausalLM
    ORTMODEL_CLASS = ORTModelForCausalLM
    TASK = "text-generation"

    def get_inputs(self, batch_size=1):
        return ["This is a sample input"] + ["This is another sample input"] * (batch_size - 1)

    def test_load_vanilla_transformers_which_is_not_supported(self):
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

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")

        model = self.AUTOMODEL_CLASS.from_pretrained(model_id, use_cache=use_cache).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

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

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_generation_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")

        model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        outputs = model.generate(**tokens, **self.GEN_KWARGS)
        onnx_outputs = onnx_model.generate(**tokens, **self.GEN_KWARGS)
        torch.testing.assert_close(outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(grid_parameters({**FULL_GRID, "num_beams": [1, 4]}))
    def test_compare_beam_search_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, num_beams: int
    ):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")

        model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        gen_kwargs = self.GEN_KWARGS.copy()
        gen_kwargs["num_beams"] = num_beams
        gen_kwargs.pop("do_sample")

        gen_configs = [GenerationConfig(**gen_kwargs, do_sample=False), GenerationConfig(**gen_kwargs, do_sample=True)]
        if num_beams == 4:
            gen_configs.append(
                GenerationConfig(**gen_kwargs, do_sample=False, num_beam_groups=2, diversity_penalty=0.0000001)
            )

        for gen_config in gen_configs:
            with torch.no_grad():
                set_seed(SEED)
                outputs = model.generate(**tokens, generation_config=gen_config)
            set_seed(SEED)
            onnx_outputs = onnx_model.generate(**tokens, generation_config=gen_config)
            torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(inputs, return_tensors="pt")

        model_with_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        model_without_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )

        num_new_tokens = self.GEN_KWARGS["max_new_tokens"]
        outputs_model_with_pkv = model_with_pkv.generate(**tokens, **self.GEN_KWARGS)
        outputs_model_without_pkv = model_without_pkv.generate(**tokens, **self.GEN_KWARGS)
        self.assertEqual(outputs_model_with_pkv.shape[1], tokens["input_ids"].shape[1] + num_new_tokens)
        self.assertEqual(outputs_model_without_pkv.shape[1], tokens["input_ids"].shape[1] + num_new_tokens)
        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(inputs, return_tensors="pt")

        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=False, provider="CPUExecutionProvider"
        )
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=True, provider="CPUExecutionProvider"
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

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_generation_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        inputs = self.get_inputs()
        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(inputs, return_tensors="pt")

        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], provider="CPUExecutionProvider", use_io_binding=False, use_cache=use_cache
        )
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], provider="CPUExecutionProvider", use_io_binding=True, use_cache=use_cache
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
        outputs = pipe(text)

        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline("text-generation", model=tmpdir)
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

        outputs = pipe(text)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline("text-generation", model=tmpdir, tokenizer=tokenizer)
            outputs_local_model = pipe(text)
            self.assertEqual(outputs[0]["generated_text"], outputs_local_model[0]["generated_text"])

    @parameterized.expand(grid_parameters({"model_arch": ["llama"], "use_cache": [True, False]}, add_test_name=False))
    def test_pipeline_with_hub_model_id(self, model_arch: str, use_cache: bool):
        model_id = MODEL_NAMES[model_arch]

        pipe = pipeline("text-generation", model=model_id, accelerator="ort", model_kwargs={"use_cache": use_cache})
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
        with tempfile.TemporaryDirectory() as tmpdir:
            task = "text-generation-with-past"
            model_id = MODEL_NAMES[model_arch]
            tokenizer = get_preprocessor(model_id)
            text = "My Name is Philipp and i live"
            tokens = tokenizer(text, return_tensors="pt")
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
            main_export(model_id, output=tmpdir, task=task, legacy=True)

            # not merged model, no cache
            self.assertIn(ONNX_DECODER_NAME, os.listdir(tmpdir))
            not_merged_no_cache_file = os.path.join(tmpdir, ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(not_merged_no_cache_file, "use_cache_branch"))
            not_merged_no_cache_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir, use_cache=False, use_merged=False)
            self.assertFalse(not_merged_no_cache_model.use_merged)
            self.assertFalse(not_merged_no_cache_model.use_cache)

            # not merged model, with cache
            self.assertIn(ONNX_DECODER_WITH_PAST_NAME, os.listdir(tmpdir))
            not_merged_with_cache_file = os.path.join(tmpdir, ONNX_DECODER_WITH_PAST_NAME)
            self.assertFalse(has_onnx_input(not_merged_with_cache_file, "use_cache_branch"))
            not_merged_with_cache_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir, use_cache=True, use_merged=False)
            self.assertFalse(not_merged_with_cache_model.use_merged)
            self.assertTrue(not_merged_with_cache_model.use_cache)

            # merged model
            self.assertIn(ONNX_DECODER_MERGED_NAME, os.listdir(tmpdir))
            merged_file = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(merged_file, "use_cache_branch"))
            merged_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir, use_merged=True)
            self.assertTrue(merged_model.use_merged)
            self.assertTrue(merged_model.use_cache)

            # inference
            outputs = model.generate(**tokens, **self.GEN_KWARGS)
            merged_outputs = merged_model.generate(**tokens, **self.GEN_KWARGS)
            not_merged_no_cache_outputs = not_merged_no_cache_model.generate(**tokens, **self.GEN_KWARGS)
            not_merged_with_cache_outputs = not_merged_with_cache_model.generate(**tokens, **self.GEN_KWARGS)

            # compare merged to transformers
            torch.testing.assert_close(outputs, merged_outputs, atol=self.ATOL, rtol=self.RTOL)
            # compare not merged no cache to transformers
            torch.testing.assert_close(outputs, not_merged_no_cache_outputs, atol=self.ATOL, rtol=self.RTOL)
            # compare not merged with cache to transformers
            torch.testing.assert_close(outputs, not_merged_with_cache_outputs, atol=self.ATOL, rtol=self.RTOL)

            # load and save (only merged is loaded and saved)
            loaded_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir)
            self.assertEqual(loaded_model.path.name, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(loaded_model.use_merged)
            self.assertTrue(loaded_model.use_cache)
            loaded_model.save_pretrained(tmpdir)
            self.assertIn(ONNX_DECODER_MERGED_NAME, os.listdir(tmpdir))
            self.assertNotIn(ONNX_DECODER_NAME, os.listdir(tmpdir))
            self.assertNotIn(ONNX_DECODER_WITH_PAST_NAME, os.listdir(tmpdir))
