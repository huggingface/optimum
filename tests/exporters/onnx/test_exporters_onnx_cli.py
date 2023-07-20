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
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import pytest
from parameterized import parameterized
from transformers import AutoModelForSequenceClassification, AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, require_torch_gpu, require_vision, slow

from optimum.exporters.error_utils import MinimumVersionError
from optimum.exporters.onnx.__main__ import main_export
from optimum.onnxruntime import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME
from optimum.utils.testing_utils import require_diffusers


if is_torch_available():
    from optimum.exporters.tasks import TasksManager

from ..exporters_utils import PYTORCH_EXPORT_MODELS_TINY, PYTORCH_STABLE_DIFFUSION_MODEL


def _get_models_to_test(export_models_dict: Dict):
    models_to_test = []
    if is_torch_available():
        for model_type, model_names_tasks in export_models_dict.items():
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")

            if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
                tasks = list(task_config_mapping.keys())
                model_tasks = {model_names_tasks: tasks}
            else:
                unique_tasks = set()
                for tasks in model_names_tasks.values():
                    for task in tasks:
                        unique_tasks.add(task)
                n_tested_tasks = len(unique_tasks)
                if n_tested_tasks != len(task_config_mapping):
                    raise ValueError(f"Not all tasks are tested for {model_type}.")
                model_tasks = model_names_tasks  # possibly, test different tasks on different models

            for model_name, tasks in model_tasks.items():
                for task in tasks:
                    models_to_test.append((f"{model_type}_{task}", model_type, model_name, task, False, False))

                    # -with-past and monolith cases are absurd, so we don't test them as not supported
                    if any(
                        task == ort_special_task
                        for ort_special_task in [
                            "text-generation",
                            "text2text-generation",
                            "automatic-speech-recognition",
                            "image-to-text",
                        ]
                    ):
                        models_to_test.append(
                            (f"{model_type}_{task}_monolith", model_type, model_name, task, True, False)
                        )

                    # For other tasks, we don't test --no-post-process as there is none anyway
                    if task in [
                        "feature-extraction-with-past",
                        "text-generation-with-past",
                        "automatic-speech-recognition-with-past",
                        "image-to-text-with-past",
                        "text2text-generation-with-past",
                    ]:
                        models_to_test.append(
                            (f"{model_type}_{task}_no_postprocess", model_type, model_name, task, False, True)
                        )

            # TODO: segformer task can not be automatically inferred
            # TODO: xlm-roberta model auto-infers text-generation, but we don't support it
            # TODO: perceiver auto-infers default, but we don't support it (why?)
            if model_type not in ["segformer", "xlm-roberta", "perceiver", "vision-encoder-decoder"]:
                models_to_test.append((f"{model_type}_no_task", model_type, model_name, "auto", False, False))

        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy")]


class OnnxCLIExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _onnx_export(
        self,
        model_name: str,
        task: str,
        monolith: bool = False,
        no_post_process: bool = False,
        optimization_level: Optional[str] = None,
        device: str = "cpu",
        fp16: bool = False,
    ):
        with TemporaryDirectory() as tmpdir:
            try:
                main_export(
                    model_name_or_path=model_name,
                    output=tmpdir,
                    task=task,
                    device=device,
                    fp16=fp16,
                    optimize=optimization_level,
                    monolith=monolith,
                    no_post_process=no_post_process,
                )
            except MinimumVersionError as e:
                pytest.skip(f"Skipping due to minimum version requirements not met. Full error: {e}")

    def test_all_models_tested(self):
        # make sure we test all models
        missing_models_set = TasksManager._SUPPORTED_CLI_MODEL_TYPE - set(PYTORCH_EXPORT_MODELS_TINY.keys())
        if len(missing_models_set) > 0:
            self.fail(f"Not testing all models. Missing models: {missing_models_set}")

    @parameterized.expand(PYTORCH_STABLE_DIFFUSION_MODEL.items())
    @require_torch
    @require_vision
    @require_diffusers
    def test_exporters_cli_pytorch_cpu_stable_diffusion(self, model_type: str, model_name: str):
        self._onnx_export(model_name, model_type)

    @parameterized.expand(PYTORCH_STABLE_DIFFUSION_MODEL.items())
    @require_torch_gpu
    @require_vision
    @require_diffusers
    @slow
    @pytest.mark.run_slow
    def test_exporters_cli_pytorch_gpu_stable_diffusion(self, model_type: str, model_name: str):
        self._onnx_export(model_name, model_type, device="cuda")

    @parameterized.expand(PYTORCH_STABLE_DIFFUSION_MODEL.items())
    @require_torch_gpu
    @require_vision
    @require_diffusers
    @slow
    @pytest.mark.run_slow
    def test_exporters_cli_fp16_stable_diffusion(self, model_type: str, model_name: str):
        self._onnx_export(model_name, model_type, device="cuda", fp16=True)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    def test_exporters_cli_pytorch_cpu(
        self, test_name: str, model_type: str, model_name: str, task: str, monolith: bool, no_post_process: bool
    ):
        # TODO: re-enable those tests
        # Failing due to https://github.com/huggingface/transformers/pull/22212
        # It is not as simple as changing "logits" by "reconstruction" as not all
        # masked-im models use MaskedImageModelingOutput
        if model_type in ["vit", "deit"] and task == "masked-im":
            self.skipTest("Temporarily disabled upon transformers 4.28 release")
        self._onnx_export(model_name, task, monolith, no_post_process)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_vision
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_exporters_cli_pytorch_gpu(
        self, test_name: str, model_type: str, model_name: str, task: str, monolith: bool, no_post_process: bool
    ):
        # TODO: disable due to a bug in PyTorch: https://github.com/pytorch/pytorch/issues/95377
        if model_type == "yolos":
            self.skipTest("Export on cuda device fails for yolos due to a bug in PyTorch")

        self._onnx_export(model_name, task, monolith, no_post_process, device="cuda")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    @slow
    @pytest.mark.run_slow
    def test_exporters_cli_pytorch_with_optimization(
        self, test_name: str, model_type: str, model_name: str, task: str, monolith: bool, no_post_process: bool
    ):
        for optimization_level in ["O1", "O2", "O3"]:
            try:
                self._onnx_export(model_name, task, monolith, no_post_process, optimization_level=optimization_level)
            except NotImplementedError as e:
                if "Tried to use ORTOptimizer for the model type" in str(
                    e
                ) or "doesn't support the graph optimization" in str(e):
                    self.skipTest(f"unsupported model type in ORTOptimizer: {model_type}")
                else:
                    raise e

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch_gpu
    @require_vision
    @slow
    @pytest.mark.gpu_test
    @pytest.mark.run_slow
    def test_exporters_cli_pytorch_with_O4_optimization(
        self, test_name: str, model_type: str, model_name: str, task: str, monolith: bool, no_post_process: bool
    ):
        # TODO: disable due to a bug in PyTorch: https://github.com/pytorch/pytorch/issues/95377
        if model_type == "yolos":
            self.skipTest("Export on cuda device fails for yolos due to a bug in PyTorch")

        try:
            self._onnx_export(model_name, task, monolith, no_post_process, optimization_level="O4", device="cuda")
        except NotImplementedError as e:
            if "Tried to use ORTOptimizer for the model type" in str(
                e
            ) or "doesn't support the graph optimization" in str(e):
                self.skipTest(f"unsupported model type in ORTOptimizer: {model_type}")
            else:
                raise e

    @parameterized.expand([(False,), (True,)])
    def test_external_data(self, use_cache: bool):
        os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data

        with TemporaryDirectory() as tmpdirname:
            task = "text2text-generation"
            if use_cache:
                task += "-with-past"

            subprocess.run(
                f"python3 -m optimum.exporters.onnx --model hf-internal-testing/tiny-random-t5 --task {task} {tmpdirname}",
                shell=True,
                check=True,
            )

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_ENCODER_NAME + "_data" in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME + "_data" in folder_contents)

            if use_cache:
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME in folder_contents)
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME + "_data" in folder_contents)

        os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    def test_trust_remote_code(self):
        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.onnx --model fxmarty/tiny-testing-gpt2-remote-code --task text-generation {tmpdirname}",
                shell=True,
                capture_output=True,
            )
            self.assertFalse(out.returncode)
            # self.assertTrue("requires you to execute the modeling file in that repo" in out.stderr.decode("utf-8"))

        with TemporaryDirectory() as tmpdirname:
            out = subprocess.run(
                f"python3 -m optimum.exporters.onnx --trust-remote-code --model fxmarty/tiny-testing-gpt2-remote-code --task text-generation {tmpdirname}",
                shell=True,
                check=True,
            )

    def test_stable_diffusion(self):
        with TemporaryDirectory() as tmpdirname:
            subprocess.run(
                f"python3 -m optimum.exporters.onnx --model hf-internal-testing/tiny-stable-diffusion-torch --task stable-diffusion {tmpdirname}",
                shell=True,
                check=True,
            )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_vision
    @require_torch_gpu
    @slow
    @pytest.mark.run_slow
    def test_export_on_fp16(
        self, test_name: str, model_type: str, model_name: str, task: str, monolith: bool, no_post_process: bool
    ):
        # TODO: refer to https://github.com/pytorch/pytorch/issues/95377
        if model_type == "yolos":
            self.skipTest("yolos export on fp16 not supported due to a pytorch bug")

        # TODO: refer to https://huggingface.slack.com/archives/C014N4749J9/p1677245766278129
        if model_type == "deberta":
            self.skipTest("deberta export on fp16 not supported due to a transformers bug")

        # TODO: test once https://github.com/huggingface/transformers/pull/21789 is released
        if (model_type == "vit" and task == "masked-im") or model_type == "vision-encoder-decoder":
            self.skipTest(
                "vit + masked-im, and vision-encoder-decoder export on fp16 not supported due to a transformers bug"
            )

        # TODO: test once https://github.com/huggingface/transformers/pull/21787 is released
        if model_type == "perceiver" and task == "image-classification":
            self.skipTest("perceiver + image-classification export on fp16 not supported due to a transformers bug")

        if model_type == "ibert":
            self.skipTest("ibert can not be supported in fp16")

        self._onnx_export(model_name, task, monolith, no_post_process, fp16=True)

    @parameterized.expand(
        [
            ["causal-lm", "gpt2"],
            ["causal-lm-with-past", "gpt2"],
            ["seq2seq-lm", "t5"],
            ["seq2seq-lm-with-past", "t5"],
            ["speech2seq-lm", "whisper"],
            ["speech2seq-lm-with-past", "whisper"],
            ["vision2seq-lm", "vision-encoder-decoder"],
            ["sequence-classification", "bert"],
            ["masked-lm", "bert"],
            ["default", "blenderbot"],
            ["default-with-past", "blenderbot"],
            ["audio-ctc", "wav2vec2-conformer"],
        ]
    )
    @slow
    @pytest.mark.run_slow
    def test_synonym_tasks_backward_compatibility(self, task: str, model_type: str):
        model_name = PYTORCH_EXPORT_MODELS_TINY[model_type]

        if isinstance(model_name, dict):
            for _model_name in model_name.keys():
                with TemporaryDirectory() as tmpdir:
                    main_export(model_name_or_path=_model_name, output=tmpdir, task=task)
        else:
            with TemporaryDirectory() as tmpdir:
                main_export(model_name_or_path=model_name, output=tmpdir, task=task)

    @slow
    def test_complex_synonyms(self):
        # conversational (text2text-generation)
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path="facebook/blenderbot-400M-distill", output=tmpdir)
            self.assertTrue(Path(tmpdir, "decoder_with_past_model.onnx").is_file())

        # conversational (text-generation)
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path="microsoft/DialoGPT-small", output=tmpdir)
            self.assertTrue(Path(tmpdir, "decoder_with_past_model.onnx").is_file())

        # summarization
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path="facebook/bart-large-cnn", output=tmpdir)

        # zero-shot-classification
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path="facebook/bart-large-mnli", output=tmpdir)

        # translation
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path="t5-small", output=tmpdir)
            self.assertTrue(Path(tmpdir, "decoder_with_past_model.onnx").is_file())

        # sentence-similarity
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path="sentence-transformers/paraphrase-TinyBERT-L6-v2", output=tmpdir)

        # from local
        with TemporaryDirectory() as tmpdir_in, TemporaryDirectory() as tmpdir_out:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

            tokenizer.save_pretrained(tmpdir_in)
            model.save_pretrained(tmpdir_in)

            main_export(model_name_or_path=tmpdir_in, output=tmpdir_out, task="text-classification")
