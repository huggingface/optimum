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
import gc
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
import requests
import torch
from huggingface_hub import HfApi
from huggingface_hub.constants import default_cache_path
from parameterized import parameterized
from PIL import Image
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForImageToImage,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoTokenizer,
    MBartForConditionalGeneration,
    Pix2StructForConditionalGeneration,  # Pix2Struct does not work with AutoModel
    PretrainedConfig,
    set_seed,
)
from transformers.modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from transformers.modeling_utils import no_init_weights
from transformers.models.swin2sr.configuration_swin2sr import Swin2SRConfig
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import get_gpu_count, require_torch_gpu
from transformers.utils import http_user_agent

from optimum.exporters import TasksManager
from optimum.exporters.onnx import main_export
from optimum.onnx.utils import has_onnx_input
from optimum.onnxruntime import (
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    ONNX_WEIGHTS_NAME,
    ORTModelForAudioClassification,
    ORTModelForAudioFrameClassification,
    ORTModelForAudioXVector,
    ORTModelForCTC,
    ORTModelForCustomTasks,
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForImageToImage,
    ORTModelForMaskedLM,
    ORTModelForMultipleChoice,
    ORTModelForPix2Struct,
    ORTModelForQuestionAnswering,
    ORTModelForSemanticSegmentation,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTModelForSpeechSeq2Seq,
    ORTModelForTokenClassification,
    ORTModelForVision2Seq,
)
from optimum.onnxruntime.modeling_ort import ORTModel
from optimum.onnxruntime.modeling_seq2seq import ORTDecoderForSeq2Seq, ORTEncoder
from optimum.pipelines import pipeline
from optimum.utils import CONFIG_NAME, logging
from optimum.utils.save_utils import maybe_load_preprocessors
from optimum.utils.testing_utils import grid_parameters, remove_directory, require_hf_token, require_ort_rocm


logger = logging.get_logger()


class ORTModelIntegrationTest(unittest.TestCase):
    ORTMODEL_CLASS = ORTModel
    AUTOMODEL_CLASS = AutoModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LOCAL_MODEL_PATH = "tests/assets/onnx"
        self.ONNX_MODEL_ID = "philschmid/distilbert-onnx"
        self.TINY_ONNX_MODEL_ID = "fxmarty/resnet-tiny-beans"
        self.FAIL_ONNX_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.ONNX_SEQ2SEQ_MODEL_ID = "optimum/t5-small"
        self.LARGE_ONNX_SEQ2SEQ_MODEL_ID = "facebook/mbart-large-en-ro"
        self.TINY_ONNX_SEQ2SEQ_MODEL_ID = "fxmarty/sshleifer-tiny-mbart-onnx"

    def test_load_model_from_hub_infer_onnx_model(self):
        model_id = "optimum-internal-testing/tiny-random-llama"
        file_name = "model_optimized.onnx"

        model = self.ORTMODEL_CLASS.from_pretrained(model_id)
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx")
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx", file_name=file_name)
        self.assertEqual(model.path.name, file_name)

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx", file_name=file_name)
        self.assertEqual(model.path.name, file_name)

        model = self.ORTMODEL_CLASS.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx", subfolder="subfolder")
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", subfolder="subfolder", file_name=file_name
        )
        self.assertEqual(model.path.name, file_name)

        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", file_name="decoder_with_past_model.onnx"
        )
        self.assertEqual(model.path.name, "decoder_with_past_model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.assertEqual(model.path.name, "model.onnx")

        with self.assertRaises(FileNotFoundError):
            self.ORTMODEL_CLASS.from_pretrained(
                "hf-internal-testing/tiny-random-LlamaForCausalLM", file_name="test.onnx"
            )

    def test_load_model_seq2seq_from_hub_infer_onnx_model(self):
        model_id = "hf-internal-testing/tiny-random-T5Model"
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id)
        model_parts = {part.model_path.name for part in model.parts}
        self.assertEqual(model_parts, {"encoder_model.onnx", "decoder_model_merged.onnx"})
        self.assertTrue(model.use_merged)

        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, use_merged=False)
        model_parts = {part.model_path.name for part in model.parts}
        expected_model_parts = {"encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"}
        self.assertTrue(model.use_cache)
        self.assertFalse(model.use_merged)
        self.assertEqual(model_parts, expected_model_parts)

        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, use_merged=False, use_cache=False)
        model_parts = {part.model_path.name for part in model.parts}
        expected_model_parts = {"encoder_model.onnx", "decoder_model.onnx"}
        self.assertFalse(model.use_cache)
        self.assertFalse(model.use_merged)
        self.assertEqual(model_parts, expected_model_parts)

        model_id = "optimum-internal-testing/tiny-random-T5Model"
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id)
        model_parts = {part.model_path.name for part in model.parts}
        self.assertEqual(model_parts, {"encoder_model.onnx", "decoder_model_merged.onnx"})
        self.assertTrue(model.use_cache)
        self.assertTrue(model.use_merged)

        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, revision="onnx-legacy")
        model_parts = {part.model_path.name for part in model.parts}
        expected_model_parts = {"encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"}
        self.assertEqual(model_parts, expected_model_parts)
        self.assertTrue(model.use_cache)
        self.assertFalse(model.use_merged)

        file_names = {
            "encoder_file_name": "encoder_model_quantized.onnx",
            "decoder_file_name": "decoder_model_quantized.onnx",
            "decoder_with_past_file_name": "decoder_with_past_model_quantized.onnx",
        }
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, revision="optimized", subfolder="onnx", **file_names)
        self.assertEqual({part.model_path.name for part in model.parts}, set(file_names.values()))
        self.assertTrue(model.use_cache)
        self.assertFalse(model.use_merged)

        model = ORTModelForSeq2SeqLM.from_pretrained(
            model_id, revision="optimized", subfolder="subfolder", **file_names
        )
        self.assertEqual({part.model_path.name for part in model.parts}, set(file_names.values()))
        self.assertTrue(model.use_cache)
        self.assertFalse(model.use_merged)
        self.assertTrue("subfolder" in str(model.model_save_dir))

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            self.assertTrue(set(file_names.values()).issubset(set(os.listdir(tmpdirname))))
            model = ORTModelForSeq2SeqLM.from_pretrained(tmpdirname)

    def test_load_model_from_local_path(self):
        model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        model = ORTModel.from_pretrained(
            "fxmarty/tiny-bert-sst2-distilled-subfolder",
            subfolder="my_subfolder",
        )
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = ORTModel.from_pretrained("fxmarty/tiny-bert-sst2-distilled-onnx-subfolder", subfolder="my_subfolder")
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_from_hub_subfolder(self):
        model = ORTModelForSeq2SeqLM.from_pretrained("fxmarty/tiny-mbart-subfolder", subfolder="my_folder")
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.decoder_with_past, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = ORTModelForSeq2SeqLM.from_pretrained("fxmarty/tiny-mbart-onnx-subfolder", subfolder="my_folder")
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.decoder_with_past, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_cache(self):
        _ = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID)  # caching

        model = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)

        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.TINY_ONNX_MODEL_ID.replace("/", "--"))
        remove_directory(dirpath)

        with self.assertRaises(Exception):
            _ = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)

    def test_load_seq2seq_model_from_cache(self):
        _ = ORTModelForSeq2SeqLM.from_pretrained(self.TINY_ONNX_SEQ2SEQ_MODEL_ID)  # caching

        model = ORTModelForSeq2SeqLM.from_pretrained(self.TINY_ONNX_SEQ2SEQ_MODEL_ID, local_files_only=True)

        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.decoder_with_past, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.TINY_ONNX_SEQ2SEQ_MODEL_ID.replace("/", "--"))
        remove_directory(dirpath)

        with self.assertRaises(Exception):
            _ = ORTModelForSeq2SeqLM.from_pretrained(self.TINY_ONNX_SEQ2SEQ_MODEL_ID, local_files_only=True)

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_load_model_cuda_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    @require_torch_gpu
    @pytest.mark.trt_ep_test
    def test_load_model_tensorrt_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="TensorrtExecutionProvider")
        self.assertListEqual(
            model.providers, ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_load_model_rocm_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="ROCMExecutionProvider")
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    def test_load_model_cpu_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cpu"))

    def test_load_model_unknown_provider(self):
        with self.assertRaises(ValueError):
            ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="FooExecutionProvider")

    def test_load_seq2seq_model_from_hub(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.decoder_with_past, ORTDecoderForSeq2Seq)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_without_past_from_hub(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=False)
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoderForSeq2Seq)
        self.assertTrue(model.decoder_with_past is None)
        self.assertIsInstance(model.config, PretrainedConfig)

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_load_seq2seq_model_cuda_provider(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.decoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_load_seq2seq_model_rocm_provider(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, provider="ROCMExecutionProvider")
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.decoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    def test_load_seq2seq_model_cpu_provider(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, provider="CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])
        self.assertListEqual(model.encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.decoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cpu"))

    def test_load_seq2seq_model_unknown_provider(self):
        with self.assertRaises(ValueError):
            ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, provider="FooExecutionProvider")

    def test_load_model_from_hub_without_onnx_model(self):
        ORTModel.from_pretrained(self.FAIL_ONNX_MODEL_ID)

    def test_model_on_cpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    # test string device input for to()
    def test_model_on_cpu_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        cpu = torch.device("cpu")
        model.to("cpu")
        self.assertEqual(model.device, cpu)
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    def test_missing_execution_provider(self):
        with self.assertRaises(ValueError) as cm:
            ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="ThisProviderDoesNotExist")

        self.assertTrue("but the available execution providers" in str(cm.exception))

        is_onnxruntime_gpu_installed = (
            subprocess.run("pip list | grep onnxruntime-gpu", shell=True, capture_output=True).stdout.decode("utf-8")
            != ""
        )
        is_onnxruntime_installed = "onnxruntime " in subprocess.run(
            "pip list | grep onnxruntime", shell=True, capture_output=True
        ).stdout.decode("utf-8")

        if not is_onnxruntime_gpu_installed:
            for provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
                with self.assertRaises(ValueError) as cm:
                    _ = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider=provider)

            self.assertTrue("but the available execution providers" in str(cm.exception))

        else:
            logger.info("Skipping CUDAExecutionProvider/TensorrtExecutionProvider without `onnxruntime-gpu` test")

        # need to install first onnxruntime-gpu, then onnxruntime for this test to pass,
        # thus overwritting onnxruntime/capi/_ld_preload.py
        if is_onnxruntime_installed and is_onnxruntime_gpu_installed:
            for provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
                with self.assertRaises(ValueError) as cm:
                    _ = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider=provider)

                self.assertTrue("but the available execution providers" in str(cm.exception))
        else:
            logger.info("Skipping double onnxruntime + onnxruntime-gpu install test")

        # despite passing CUDA_PATH='' LD_LIBRARY_PATH='', this test does not pass in nvcr.io/nvidia/tensorrt:22.08-py3
        # It does pass locally.
        """
        # LD_LIBRARY_PATH can't be set at runtime,
        # see https://stackoverflow.com/questions/856116/changing-ld-library-path-at-runtime-for-ctypes
        # testing only for TensorRT as having ORT_CUDA_UNAVAILABLE is hard
        if is_onnxruntime_gpu_installed:
            commands = [
                "from optimum.onnxruntime import ORTModel",
                "model = ORTModel.from_pretrained('philschmid/distilbert-onnx', provider='TensorrtExecutionProvider')",
            ]

            full_command = json.dumps(";".join(commands))

            out = subprocess.run(
                f"CUDA_PATH='' LD_LIBRARY_PATH='' python -c {full_command}", shell=True, capture_output=True
            )
            self.assertTrue("requirements could not be loaded" in out.stderr.decode("utf-8"))
        else:
            logger.info("Skipping broken CUDA/TensorRT install test")
        """

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_model_on_gpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_model_on_rocm_ep(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    # test string device input for to()
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_model_on_gpu_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_model_on_rocm_ep_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    def test_passing_session_options(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 3
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, session_options=options)
        self.assertEqual(model.model.get_session_options().intra_op_num_threads, 3)

    def test_passing_session_options_seq2seq(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 3
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, session_options=options)
        self.assertEqual(model.encoder.session.get_session_options().intra_op_num_threads, 3)
        self.assertEqual(model.decoder.session.get_session_options().intra_op_num_threads, 3)

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_passing_provider_options(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1")

        model = ORTModel.from_pretrained(
            self.ONNX_MODEL_ID,
            provider="CUDAExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
        )
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0")

        # two providers case
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="TensorrtExecutionProvider")
        self.assertEqual(
            model.model.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "0"
        )

        model = ORTModel.from_pretrained(
            self.ONNX_MODEL_ID,
            provider="TensorrtExecutionProvider",
            provider_options={"trt_engine_cache_enable": True},
        )
        self.assertEqual(
            model.model.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "1"
        )

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_passing_provider_options_rocm_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="ROCMExecutionProvider")
        self.assertEqual(model.model.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "1")

        model = ORTModel.from_pretrained(
            self.ONNX_MODEL_ID,
            provider="ROCMExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
        )
        self.assertEqual(model.model.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "0")

    @unittest.skipIf(get_gpu_count() <= 1, "this test requires multi-gpu")
    def test_model_on_gpu_id(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to(torch.device("cuda:1"))
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to(1)
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to("cuda:1")
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_passing_provider_options_seq2seq(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertEqual(
            model.encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        self.assertEqual(
            model.decoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["CUDAExecutionProvider"][
                "do_copy_in_default_stream"
            ],
            "1",
        )

        model = ORTModelForSeq2SeqLM.from_pretrained(
            self.ONNX_SEQ2SEQ_MODEL_ID,
            provider="CUDAExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
            use_cache=True,
        )
        self.assertEqual(
            model.encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0"
        )
        self.assertEqual(
            model.decoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0"
        )
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["CUDAExecutionProvider"][
                "do_copy_in_default_stream"
            ],
            "0",
        )

        # two providers case
        model = ORTModelForSeq2SeqLM.from_pretrained(
            self.ONNX_SEQ2SEQ_MODEL_ID,
            provider="TensorrtExecutionProvider",
            use_cache=True,
        )
        self.assertEqual(
            model.encoder.session.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "0"
        )
        self.assertEqual(
            model.decoder.session.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "0"
        )
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["TensorrtExecutionProvider"][
                "trt_engine_cache_enable"
            ],
            "0",
        )

        model = ORTModelForSeq2SeqLM.from_pretrained(
            self.ONNX_SEQ2SEQ_MODEL_ID,
            provider="TensorrtExecutionProvider",
            provider_options={"trt_engine_cache_enable": True},
            use_cache=True,
        )
        self.assertEqual(
            model.encoder.session.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "1"
        )
        self.assertEqual(
            model.decoder.session.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "1"
        )
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["TensorrtExecutionProvider"][
                "trt_engine_cache_enable"
            ],
            "1",
        )

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_passing_provider_options_seq2seq_rocm_provider(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, provider="ROCMExecutionProvider")
        self.assertEqual(
            model.encoder.session.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        self.assertEqual(
            model.decoder.session.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["ROCMExecutionProvider"][
                "do_copy_in_default_stream"
            ],
            "1",
        )

        model = ORTModelForSeq2SeqLM.from_pretrained(
            self.ONNX_SEQ2SEQ_MODEL_ID,
            provider="ROCMExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
            use_cache=True,
        )
        self.assertEqual(
            model.encoder.session.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "0"
        )
        self.assertEqual(
            model.decoder.session.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "0"
        )
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["ROCMExecutionProvider"][
                "do_copy_in_default_stream"
            ],
            "0",
        )

    def test_seq2seq_model_on_cpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.encoder.device, cpu)
        self.assertEqual(model.decoder.device, cpu)
        self.assertEqual(model.decoder_with_past.device, cpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    # test string device input for to()
    def test_seq2seq_model_on_cpu_str(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        cpu = torch.device("cpu")
        model.to("cpu")
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.encoder.device, cpu)
        self.assertEqual(model.decoder.device, cpu)
        self.assertEqual(model.decoder_with_past.device, cpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_seq2seq_model_on_gpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder_with_past.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_seq2seq_model_on_rocm_ep(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder_with_past.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    @unittest.skipIf(get_gpu_count() <= 1, "this test requires multi-gpu")
    def test_seq2seq_model_on_gpu_id(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        model.to(torch.device("cuda:1"))
        self.assertEqual(model.encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1"
        )

        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        model.to(1)
        self.assertEqual(model.encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1"
        )

        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        model.to("cuda:1")
        self.assertEqual(model.encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(
            model.decoder_with_past.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1"
        )

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_seq2seq_model_on_gpu_str(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder_with_past.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_seq2seq_model_on_rocm_ep_str(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.decoder_with_past.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    def test_load_model_from_hub_private(self):
        token = os.environ.get("HF_HUB_READ_TOKEN", None)

        if not token:
            self.skipTest(
                "Test requires a read access token for optimum-internal-testing in the environment variable `HF_HUB_READ_TOKEN`."
            )

        model = ORTModelForCustomTasks.from_pretrained(
            "optimum-internal-testing/tiny-random-phi-private", revision="onnx", token=token
        )

        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and ONNX exported model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_save_seq2seq_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            # Verify config and ONNX exported encoder, decoder and decoder with past are present in folder
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_save_seq2seq_model_without_past(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=False)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            # Verify config and ONNX exported encoder and decoder present in folder
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_load_ort_model_with_external_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], export=True)
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(ONNX_WEIGHTS_NAME, folder_contents)
            self.assertIn(ONNX_WEIGHTS_NAME + "_data", folder_contents)

            # verify loading from local folder works
            model = ORTModelForSequenceClassification.from_pretrained(tmpdirname, export=False)
            remove_directory(tmpdirname)

    @parameterized.expand([(False,), (True,)])
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_load_seq2seq_model_with_external_data(self, use_cache: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["t5"], export=True, use_cache=use_cache, use_io_binding=use_cache
            )
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_ENCODER_NAME + "_data" in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME + "_data" in folder_contents)

            if use_cache:
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME in folder_contents)
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME + "_data" in folder_contents)

            # verify loading from local folder works
            model = ORTModelForSeq2SeqLM.from_pretrained(
                tmpdirname, export=False, use_cache=use_cache, use_io_binding=use_cache
            )
            remove_directory(tmpdirname)

    @parameterized.expand([(False,), (True,)])
    @unittest.skip("Skipping as this test consumes too much memory")
    def test_save_load_large_seq2seq_model_with_external_data(self, use_cache: bool):
        # with tempfile.TemporaryDirectory() as tmpdirname:
        if True:
            tmpdirname = tempfile.mkdtemp()
            # randomly intialize large model
            config = AutoConfig.from_pretrained(self.LARGE_ONNX_SEQ2SEQ_MODEL_ID)
            with no_init_weights():
                model = MBartForConditionalGeneration(config)

            # save transformers model to be able to load it with `ORTModel...`
            model.save_pretrained(tmpdirname)

            model = ORTModelForSeq2SeqLM.from_pretrained(tmpdirname, use_cache=use_cache, export=True)
            model.save_pretrained(os.path.join(tmpdirname, "onnx"))

            # Verify config and ONNX exported encoder, decoder and decoder with past are present each in their own folder
            folder_contents = os.listdir(os.path.join(tmpdirname, "onnx"))
            self.assertTrue(CONFIG_NAME in folder_contents)

            # try loading models to check if they are valid
            try:
                onnx.load(os.path.join(tmpdirname, "onnx", ONNX_ENCODER_NAME))
                onnx.load(os.path.join(tmpdirname, "onnx", ONNX_DECODER_NAME))
                if use_cache:
                    onnx.load(os.path.join(tmpdirname, "onnx", ONNX_DECODER_WITH_PAST_NAME))
            except Exception as e:
                self.fail("Model with external data wasn't saved properly.\nCould not load model from disk: " + str(e))

            # verify loading from local folder works
            model = ORTModelForSeq2SeqLM.from_pretrained(tmpdirname, use_cache=use_cache, subfolder="onnx")

    @require_hf_token
    def test_save_model_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(
                tmpdirname,
                token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id=self.HUB_REPOSITORY,
                private=True,
            )

    @require_hf_token
    def test_push_ort_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], export=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["bert"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTModelForSequenceClassification.from_pretrained(
                MODEL_NAMES["bert"] + "-onnx",
                export=False,
                token=os.environ.get("HF_AUTH_TOKEN", None),
            )
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    @require_hf_token
    def test_push_seq2seq_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAMES["mbart"], export=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["mbart"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["mbart"] + "-onnx",
                export=False,
                token=os.environ.get("HF_AUTH_TOKEN", None),
            )
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    @parameterized.expand(("", "onnx"))
    def test_loading_with_config_not_from_subfolder(self, subfolder):
        # config.json file in the root directory and not in the subfolder
        model_id = "sentence-transformers-testing/stsb-bert-tiny-onnx"
        # hub model
        ORTModelForFeatureExtraction.from_pretrained(model_id, subfolder=subfolder, export=subfolder == "")
        with tempfile.TemporaryDirectory() as tmpdirname:
            local_dir = Path(tmpdirname) / "model"
            HfApi(user_agent=http_user_agent()).snapshot_download(repo_id=model_id, local_dir=local_dir)
            ORTModelForFeatureExtraction.from_pretrained(local_dir, subfolder=subfolder, export=subfolder == "")
            remove_directory(tmpdirname)


class ORTModelForQuestionAnsweringIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bart",
        "bert",
        "big_bird",
        "bigbird_pegasus",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        # "flaubert", # currently fails for some reason (squad multiprocessing),
        # but also couldn't find any real qa checkpoints on the hub for this model
        "gptj",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in ORTModel
        # "layoutlm"
        # "layoutlmv3",
        "mbart",
        "mobilebert",
        "nystromformer",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm-qa",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForQuestionAnswering
    TASK = "question-answering"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForQuestionAnswering.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        tokens = tokenizer("This is a sample output", return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer("This is a sample output", return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("start_logits", onnx_outputs)
            self.assertIn("end_logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.start_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(onnx_outputs.end_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.start_logits),
                transformers_outputs.start_logits,
                atol=self.ATOL,
                rtol=self.RTOL,
            )
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.end_logits), transformers_outputs.end_logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        self.assertEqual(pipe.device, pipe.model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("question-answering")
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer, device=0)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        provider = "ROCMExecutionProvider"
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer, device=0)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForQuestionAnswering.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("start_logits" in io_outputs)
        self.assertTrue("end_logits" in io_outputs)
        self.assertIsInstance(io_outputs.start_logits, torch.Tensor)
        self.assertIsInstance(io_outputs.end_logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            torch.Tensor(io_outputs.start_logits), onnx_outputs.start_logits, atol=self.ATOL, rtol=self.RTOL
        )
        torch.testing.assert_close(
            torch.Tensor(io_outputs.end_logits), onnx_outputs.end_logits, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class ORTModelForMaskedLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "big_bird",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "mobilebert",
        "mpnet",
        "perceiver_text",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForMaskedLM
    TASK = "fill-mask"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForMaskedLM.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = f"The capital of France is {tokenizer.mask_token}."
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer)
        MASK_TOKEN = tokenizer.mask_token
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("fill-mask")
        text = f"The capital of France is {pipe.tokenizer.mask_token}."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer, device=0)
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["token_str"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer, device=0)
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["token_str"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForMaskedLM.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer([f"The capital of France is {tokenizer.mask_token}."] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(io_outputs.logits, onnx_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    def test_load_sentence_transformers_model_as_fill_mask(self):
        model_id = "sparse-encoder-testing/splade-bert-tiny-nq"
        onnx_model = ORTModelForMaskedLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer)
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

        gc.collect()


class ORTModelForSequenceClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bart",
        "bert",
        "big_bird",
        "bigbird_pegasus",
        "bloom",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        # "gpt2",  # see tasks.py
        # "gpt_neo",  # see tasks.py
        # "gptj",  # see tasks.py
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in ORTModel
        # "layoutlm"
        # "layoutlmv3",
        "mbart",
        "mobilebert",
        "nystromformer",
        "perceiver_text",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForSequenceClassification
    TASK = "text-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("text-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    def test_pipeline_zero_shot_classification(self):
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            "typeform/distilbert-base-uncased-mnli", export=True
        )
        tokenizer = get_preprocessor("typeform/distilbert-base-uncased-mnli")
        pipe = pipeline("zero-shot-classification", model=onnx_model, tokenizer=tokenizer)
        sequence_to_classify = "Who are you voting for in 2020?"
        candidate_labels = ["Europe", "public health", "politics", "elections"]
        hypothesis_template = "This text is about {}."
        outputs = pipe(
            sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template
        )

        # compare model output class
        self.assertTrue(all(score > 0.0 for score in outputs["scores"]))
        self.assertTrue(all(isinstance(label, str) for label in outputs["labels"]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForTokenClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "big_bird",
        "bloom",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "gpt2",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in ORTModel
        # "layoutlm"
        # "layoutlmv3",
        "mobilebert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForTokenClassification
    TASK = "token-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForTokenClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("token-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForFeatureExtractionIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        "distilbert",
        "electra",
        "mpnet",
        "roberta",
        "xlm-roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForFeatureExtraction
    TASK = "feature-extraction"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            # Test default behavior (return_dict=True)
            onnx_outputs = onnx_model(**tokens)
            self.assertIsInstance(onnx_outputs, BaseModelOutput)
            self.assertIn("last_hidden_state", onnx_outputs)
            self.assertIsInstance(onnx_outputs.last_hidden_state, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Test return_dict=False
            onnx_outputs_dict = onnx_model(**tokens, return_dict=False)
            self.assertIsInstance(onnx_outputs_dict, tuple)
            self.assertIsInstance(onnx_outputs_dict[0], self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.last_hidden_state),
                transformers_outputs.last_hidden_state,
                atol=self.ATOL,
                rtol=self.RTOL,
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("feature-extraction")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("last_hidden_state" in io_outputs)
        self.assertIsInstance(io_outputs.last_hidden_state, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            onnx_outputs.last_hidden_state, io_outputs.last_hidden_state, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()

    def test_default_token_type_ids(self):
        model_id = MODEL_NAMES["bert"]
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("this is a simple input", return_tensors="np")
        self.assertTrue("token_type_ids" in model.input_names)
        token_type_ids = tokens.pop("token_type_ids")
        outs = model(token_type_ids=token_type_ids, **tokens)
        outs_without_token_type_ids = model(**tokens)
        torch.testing.assert_close(
            outs.last_hidden_state, outs_without_token_type_ids.last_hidden_state, atol=self.ATOL, rtol=self.RTOL
        )
        gc.collect()


class ORTModelForFeatureExtractionFromImageModelsIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["vit", "dinov2"]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForFeatureExtraction
    TASK = "feature-extraction"

    def get_raw_input(self, model_arch):
        image_url = "https://picsum.photos/id/237/200/300"
        return Image.open(requests.get(image_url, stream=True).raw)

    def get_input(self, model_arch, processor, return_tensors="pt"):
        raw_input = self.get_raw_input(model_arch)
        return processor(images=raw_input, return_tensors=return_tensors)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        processor = get_preprocessor(model_id)
        inputs = self.get_input(model_arch, processor, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = self.get_input(model_arch, processor, return_tensors=input_type)
            onnx_outputs = onnx_model(**inputs)

            self.assertIn("last_hidden_state", onnx_outputs)
            self.assertIsInstance(onnx_outputs.last_hidden_state, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.last_hidden_state),
                transformers_outputs.last_hidden_state,
                atol=self.ATOL,
                rtol=self.RTOL,
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_ort_model_inference(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])
        processor = get_preprocessor(model_id)
        raw_input = self.get_raw_input(model_arch)
        processed_inputs = processor(images=raw_input, return_tensors="pt")
        outputs = onnx_model(**processed_inputs)

        # Check device and output format
        assert onnx_model.device.type == "cpu"
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        features = outputs.last_hidden_state.detach().cpu().numpy().tolist()
        assert all(isinstance(item, float) for row in features for inner in row for item in inner)
        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_inference_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        processor = get_preprocessor(model_id)
        raw_input = self.get_raw_input(model_arch)
        processed_inputs = processor(images=raw_input, return_tensors="pt").to("cuda")
        outputs = onnx_model(**processed_inputs)

        # Check device and output format
        assert onnx_model.device.type == "cuda"
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        features = outputs.last_hidden_state.detach().cpu().numpy().tolist()
        assert all(isinstance(item, float) for row in features for inner in row for item in inner)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        processor = get_preprocessor(model_id)
        raw_input = self.get_raw_input(model_arch)
        tokens = processor(images=[raw_input, raw_input], return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("last_hidden_state" in io_outputs)
        self.assertIsInstance(io_outputs.last_hidden_state, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            onnx_outputs.last_hidden_state, io_outputs.last_hidden_state, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class ORTModelForMultipleChoiceIntegrationTest(ORTModelTestMixin):
    # Multiple Choice tests are conducted on different models due to mismatch size in model's classifier
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "big_bird",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "mobilebert",
        "nystromformer",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForMultipleChoice
    TASK = "multiple-choice"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMultipleChoice.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        num_choices = 4
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        start = "The color of the sky is"
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

        pt_inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
        with torch.no_grad():
            transformers_outputs = transformers_model(**pt_inputs)

        for input_type in ["pt", "np"]:
            inps = dict(inputs.convert_to_tensors(tensor_type=input_type))
            onnx_outputs = onnx_model(**inps)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMultipleChoice.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForMultipleChoice.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        num_choices = 4
        start = "The color of the sky is"
        tokenizer = get_preprocessor(model_id)
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)
        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
        inputs = dict(inputs.convert_to_tensors(tensor_type="pt").to("cuda"))

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(io_outputs.logits, onnx_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForImageClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "beit",
        "convnext",
        "convnextv2",
        "data2vec-vision",
        "deit",
        "dinov2",
        "efficientnet",
        "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "perceiver_vision",
        "poolformer",
        "resnet",
        "segformer",
        "swin",
        "swin-window",
        "vit",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForImageClassification
    TASK = "image-classification"

    def _get_model_ids(self, model_arch):
        model_ids = MODEL_NAMES[model_arch]
        if isinstance(model_ids, dict):
            model_ids = list(model_ids.keys())
        else:
            model_ids = [model_ids]
        return model_ids

    def _get_onnx_model_dir(self, model_id, model_arch, test_name):
        onnx_model_dir = self.onnx_model_dirs[test_name]
        if isinstance(MODEL_NAMES[model_arch], dict):
            onnx_model_dir = onnx_model_dir[model_id]

        return onnx_model_dir

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForImageClassification.from_pretrained(MODEL_NAMES["t5"])
        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = maybe_load_preprocessors(model_id)[-1]
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")

        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)

            onnx_outputs = onnx_model(**inputs)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), trtfs_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        preprocessor = maybe_load_preprocessors(model_id)[-1]
        pipe = pipeline("image-classification", model=onnx_model, image_processor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("image-classification")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )
        io_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForSemanticSegmentationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ("segformer", "dpt")

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForSemanticSegmentation
    TASK = "semantic-segmentation"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSemanticSegmentation.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)

            onnx_outputs = onnx_model(**inputs)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), trtfs_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(self.onnx_model_dirs[model_arch])
        preprocessor = maybe_load_preprocessors(model_id)[-1]
        pipe = pipeline("image-segmentation", model=onnx_model, image_processor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("image-segmentation")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # compare model output class
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    # TODO: enable TensorrtExecutionProvider test once https://github.com/huggingface/optimum/issues/798 is fixed
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider"]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-segmentation", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-segmentation", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForAudioClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "audio-spectrogram-transformer",
        "data2vec-audio",
        "hubert",
        "sew",
        "sew-d",
        "unispeech",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
        "whisper",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForAudioClassification
    TASK = "audio-classification"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForAudioClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=onnx_model, feature_extractor=processor, sampling_rate=220)
        data = self._generate_random_audio_data()
        outputs = pipe(data)

        self.assertEqual(pipe.device, onnx_model.device)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("audio-classification")
        data = self._generate_random_audio_data()
        outputs = pipe(data)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=onnx_model, feature_extractor=processor, device=0)
        data = self._generate_random_audio_data()
        outputs = pipe(data)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=onnx_model, feature_extractor=processor, device=0)
        data = self._generate_random_audio_data()
        outputs = pipe(data)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._generate_random_audio_data()
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForCTCIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "data2vec-audio",
        "hubert",
        "sew",
        "sew-d",
        "unispeech",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForCTC
    TASK = "ctc"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForCTC.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCTC.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForCTC.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCTC.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )
        io_model = ORTModelForCTC.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._generate_random_audio_data()
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            torch.Tensor(onnx_outputs.logits), io_outputs.logits, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class ORTModelForAudioXVectorIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "data2vec-audio",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForAudioXVector
    TASK = "audio-xvector"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForAudioXVector.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioXVector.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioXVector.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)
        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(onnx_outputs.embeddings, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.embeddings), transformers_outputs.embeddings, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioXVector.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForAudioXVector.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._generate_random_audio_data()
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)
        self.assertIsInstance(io_outputs.embeddings, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(onnx_outputs.embeddings, io_outputs.embeddings, atol=self.ATOL, rtol=self.RTOL)
        gc.collect()


class ORTModelForAudioFrameClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "data2vec-audio",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForAudioFrameClassification
    TASK = "audio-frame-classification"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForAudioFrameClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioFrameClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()


class ORTModelForSeq2SeqLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "bart",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        "encoder-decoder",
        "longt5",
        "m2m_100",
        "marian",
        "mbart",
        "mt5",
        "pegasus",
        "t5",
    ]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForSeq2SeqLM
    TASK = "text2text-generation"

    GENERATION_LENGTH = 100

    def _get_model_ids(self, model_arch):
        model_ids = MODEL_NAMES[model_arch]
        if isinstance(model_ids, dict):
            model_ids = list(model_ids.keys())
        else:
            model_ids = [model_ids]
        return model_ids

    def _get_onnx_model_dir(self, model_id, model_arch, test_name):
        onnx_model_dir = self.onnx_model_dirs[test_name]
        if isinstance(MODEL_NAMES[model_arch], dict):
            onnx_model_dir = onnx_model_dir[model_id]

        return onnx_model_dir

    @parameterized.expand([(True,)])  # old exported model ouputs gibberish when use_cache=False
    @pytest.mark.run_in_series
    def test_inference_old_seq2seq_onnx_model(self, use_cache):
        tokenizer = get_preprocessor("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            "optimum/t5-small", use_cache=use_cache, use_io_binding=False, use_merged=False
        )

        self.assertEqual(onnx_model.use_cache, use_cache)
        self.assertEqual(onnx_model.encoder.path.name, ONNX_ENCODER_NAME)
        self.assertEqual(onnx_model.decoder.path.name, ONNX_DECODER_NAME)
        if use_cache:
            self.assertEqual(onnx_model.decoder_with_past.path.name, ONNX_DECODER_WITH_PAST_NAME)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")

        onnx_outputs = onnx_model.generate(**tokens, min_new_tokens=30, max_new_tokens=30, do_sample=False)
        outputs = model.generate(**tokens, min_new_tokens=30, max_new_tokens=30, do_sample=False)
        onnx_text_outputs = tokenizer.decode(onnx_outputs[0], skip_special_tokens=True)
        text_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertEqual(onnx_text_outputs, text_outputs)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and use_cache is True
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)
            text = "This is a sample output"
            tokens = tokenizer(text, return_tensors="pt")

            # General case
            outputs = model.generate(**tokens)
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.assertIsInstance(res[0], str)

            # With input ids
            outputs = model.generate(input_ids=tokens["input_ids"])
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_transformers_and_save(self, model_arch):
        if "text2text-generation-with-past" not in TasksManager.get_supported_tasks_for_model_type(
            model_arch, exporter="onnx", library_name="transformers"
        ):
            self.skipTest("Unsupported -with-past export case")

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_merged=True is not supported for bert as a decoder")
                continue

            model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True, use_merged=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_pretrained(tmpdir)
                save_path = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
                self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

                folder_contents = os.listdir(tmpdir)
                self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
                self.assertTrue(ONNX_DECODER_NAME not in folder_contents)
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch):
        task = "text2text-generation-with-past"

        if task not in TasksManager.get_supported_tasks_for_model_type(model_arch, exporter="onnx"):
            self.skipTest("Unsupported export case", library_name="transformers")

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_merged=True is not supported for bert as a decoder")
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                main_export(model_id, tmpdir, task=task)

                model = ORTModelForSeq2SeqLM.from_pretrained(tmpdir)

                self.assertTrue(model.use_merged)
                self.assertTrue(model.decoder_with_past is None)

                model.save_pretrained(tmpdir + "_save")
                save_path = os.path.join(tmpdir + "_save", ONNX_DECODER_MERGED_NAME)
                self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

                folder_contents = os.listdir(tmpdir + "_save")
                self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
                self.assertFalse(ONNX_DECODER_NAME in folder_contents)
                self.assertFalse(ONNX_DECODER_WITH_PAST_NAME in folder_contents)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and use_cache is True
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            self.assertIsInstance(onnx_model.encoder, ORTEncoder)
            if use_merged is False:
                model_path = Path(onnx_model_dir, ONNX_DECODER_NAME)
                self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
                self.assertEqual(onnx_model.use_merged, False)
            else:
                model_path = Path(onnx_model_dir, ONNX_DECODER_MERGED_NAME)
                self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
                self.assertEqual(onnx_model.use_merged, True)

            self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
            if onnx_model.use_cache is True and onnx_model.use_merged is False:
                self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
            if onnx_model.use_cache is True and onnx_model.use_merged is True:
                self.assertTrue(onnx_model.decoder_with_past is None)

            self.assertIsInstance(onnx_model.config, PretrainedConfig)

            set_seed(SEED)
            transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            tokenizer = get_preprocessor(model_id)
            inputs = "This is a sample output"
            tokens = tokenizer(inputs, return_tensors="pt", padding=True)
            decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id

            decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens, **decoder_inputs)

            for input_type in ["pt", "np"]:
                tokens = tokenizer(inputs, return_tensors=input_type, padding=True)

                if input_type == "np":
                    decoder_inputs = {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id}

                onnx_outputs = onnx_model(**tokens, **decoder_inputs)

                self.assertTrue("logits" in onnx_outputs)
                self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

                # Compare tensor outputs
                torch.testing.assert_close(
                    torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
                )

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_text_generation(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and use_cache is True
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)

            decoder_start_token_id = onnx_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id

            # Text2Text generation
            pipe = pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id, min_new_tokens=10, max_new_tokens=10)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["generated_text"], str)

            # Summarization
            pipe = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id, min_new_tokens=10, max_new_tokens=10)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["summary_text"], str)

            # Translation
            pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id, min_new_tokens=10, max_new_tokens=10)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["translation_text"], str)

            if model_arch == "t5":
                with tempfile.TemporaryDirectory() as tmpdir:
                    pipe.save_pretrained(tmpdir)
                    model_kwargs = {"use_cache": use_cache}
                    pipe = pipeline(
                        "translation_en_to_de",
                        model=tmpdir,
                        model_kwargs=model_kwargs,
                        accelerator="ort",
                    )
                    outputs_local_model = pipe(text, min_new_tokens=10, max_new_tokens=10)
                    self.assertEqual(outputs[0]["translation_text"], outputs_local_model[0]["translation_text"])

        gc.collect()

    def test_load_pipeline(self):
        pipe = pipeline(
            "text2text-generation",
            model="echarlaix/t5-small-onnx",
            accelerator="ort",
        )
        outputs = pipe("this is an example input")
        self.assertIsInstance(outputs[0]["generated_text"], str)

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        # Text2text generation
        pipe = pipeline("text2text-generation")
        text = "This is a test"
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["generated_text"], str)

        # Summarization
        pipe = pipeline("summarization")
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_de")
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["translation_text"], str)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder"
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)
            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=False, device=0
            )
            text = "My Name is Philipp and i live"
            outputs = pipe(text, max_length=2 * len(text) + 1)
            # check model device
            self.assertEqual(pipe.model.device.type.lower(), "cuda")
            # compare model output class
            self.assertTrue(isinstance(outputs[0]["translation_text"], str))

            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=True, device=0
            )

            outputs = pipe(text, min_length=len(text) + 1, max_length=2 * len(text) + 1)
            self.assertTrue(isinstance(outputs[0]["translation_token_ids"], torch.Tensor))
            self.assertTrue(len(outputs[0]["translation_token_ids"]) > len(text))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder"
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)
            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=False, device=0
            )
            text = "My Name is Philipp and i live"
            outputs = pipe(text, max_length=2 * len(text) + 1)
            # check model device
            self.assertEqual(pipe.model.device.type.lower(), "cuda")
            # compare model output class
            self.assertTrue(isinstance(outputs[0]["translation_text"], str))

            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=True, device=0
            )

            outputs = pipe(text, min_length=len(text) + 1, max_length=2 * len(text) + 1)
            self.assertTrue(isinstance(outputs[0]["translation_token_ids"], torch.Tensor))
            self.assertTrue(len(outputs[0]["translation_token_ids"]) > len(text))

    # TRT EP compile time can be long, so we don't test all archs
    @parameterized.expand(grid_parameters({"model_arch": ["t5"], "use_cache": [True, False]}))
    @require_torch_gpu
    @pytest.mark.trt_ep_test
    def test_pipeline_on_trt_execution_provider(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        with tempfile.TemporaryDirectory() as engine_cache_dir:
            provider_options = {"trt_engine_cache_enable": True, "trt_engine_cache_path": engine_cache_dir}

            model_id = MODEL_NAMES[model_arch]
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self.onnx_model_dirs[test_name],
                provider="TensorrtExecutionProvider",
                provider_options=provider_options,
                use_cache=use_cache,
            )

            tokenizer = get_preprocessor(model_id)

            decoder_inputs = {
                "decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * onnx_model.config.decoder_start_token_id
            }

            # build engine for a short sequence
            text = ["short"]
            encoded_input = tokenizer(text, return_tensors="pt").to("cuda")
            _ = onnx_model(**encoded_input, **decoder_inputs)

            # build engine for a long sequence
            text = [" a very long input just for demo purpose, this is very long" * 10]
            encoded_input = tokenizer(text, return_tensors="pt").to("cuda")
            _ = onnx_model(**encoded_input, **decoder_inputs)

            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=True, device=0
            )
            text = "My Name is Philipp and i live"
            outputs = pipe(text, min_length=len(text) + 1, max_length=2 * len(text) + 1)
            self.assertTrue(isinstance(outputs[0]["translation_token_ids"], torch.Tensor))
            self.assertTrue(len(outputs[0]["translation_token_ids"]) > len(text))

            encoded_input = tokenizer("Please continue this", return_tensors="pt").to("cuda")
            _ = onnx_model.generate(**encoded_input)

            gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        if model_arch == "m2m_100":
            generation_length = 20  # model's predefined maximum length
        else:
            generation_length = self.GENERATION_LENGTH

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            tokenizer = get_preprocessor(model_id)
            text = "This is a sample output"
            tokens = tokenizer(text, return_tensors="pt")
            model_with_pkv = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, model_arch + "_True"), use_cache=True
            )

            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
            )

            model_without_pkv = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, model_arch + "_False"), use_cache=False
            )

            outputs_model_without_pkv = model_without_pkv.generate(
                **tokens, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
            )

            torch.testing.assert_close(
                outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL
            )
            self.assertEqual(outputs_model_with_pkv.shape[1], generation_length + 1)
            self.assertEqual(outputs_model_without_pkv.shape[1], generation_length + 1)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_merged_and_not_merged_models_outputs(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name + "_True",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": True,
        }
        self._setup(model_args)
        model_args = {
            "test_name": test_name + "_False",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": False,
        }
        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            tokenizer = get_preprocessor(model_id)
            text = "My Name is Philipp and i live"
            tokens = tokenizer(text, return_tensors="pt")

            model_not_merged_dir = self._get_onnx_model_dir(model_id, model_arch, test_name + "_False")
            model_merged_dir = self._get_onnx_model_dir(model_id, model_arch, test_name + "_True")

            model_not_merged = ORTModelForSeq2SeqLM.from_pretrained(model_not_merged_dir)
            not_merged_onnx_path = Path(model_not_merged_dir, ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
            self.assertEqual(model_not_merged.use_merged, False)

            model_merged = ORTModelForSeq2SeqLM.from_pretrained(model_merged_dir)
            merged_onnx_path = Path(model_merged_dir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
            self.assertEqual(model_merged.decoder_with_past, None)
            self.assertEqual(model_merged.use_merged, True)

            outputs_model_not_merged = model_not_merged.generate(**tokens)
            outputs_model_merged = model_merged.generate(**tokens)

            torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=False,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )
            io_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=True,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )

            self.assertFalse(onnx_model.use_io_binding)
            self.assertTrue(io_model.use_io_binding)

            tokenizer = get_preprocessor(model_id)
            tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")
            decoder_start_token_id = onnx_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id
            decoder_inputs = {
                "decoder_input_ids": torch.ones((2, 1), dtype=torch.long).to("cuda") * decoder_start_token_id
            }

            onnx_outputs = onnx_model(**tokens, **decoder_inputs)
            io_outputs = io_model(**tokens, **decoder_inputs)

            self.assertTrue("logits" in io_outputs)
            self.assertIsInstance(io_outputs.logits, torch.Tensor)

            # compare tensor outputs
            torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self,
        test_name: str,
        model_arch: str,
        use_cache: bool,
        use_merged: bool,
        num_beams: int,
    ):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=False,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )
            io_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=True,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )

            self.assertFalse(onnx_model.use_io_binding)
            self.assertTrue(io_model.use_io_binding)

            tokenizer = get_preprocessor(model_id)
            tokens = tokenizer("This is a sample output", return_tensors="pt").to("cuda")

            onnx_outputs = onnx_model.generate(**tokens, num_beams=num_beams)
            io_outputs = io_model.generate(**tokens, num_beams=num_beams)

            # compare tensor outputs
            torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForSpeechSeq2SeqIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["whisper", "speech_to_text"]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForSpeechSeq2Seq
    TASK = "automatic-speech-recognition"

    GENERATION_LENGTH = 100

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 18736), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)

        return audio_data

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_transformers_and_save(self, model_arch):
        if "automatic-speech-recognition-with-past" not in TasksManager.get_supported_tasks_for_model_type(
            model_arch, exporter="onnx", library_name="transformers"
        ):
            self.skipTest("Unsupported -with-past export case")

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, use_merged=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            save_path = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir)
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME not in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        task = "automatic-speech-recognition-with-past"

        if task not in TasksManager.get_supported_tasks_for_model_type(model_arch, exporter="onnx"):
            self.skipTest("Unsupported export case", library_name="transformers")

        with tempfile.TemporaryDirectory() as tmpdir:
            main_export(model_id, tmpdir, task=task)

            model = ORTModelForSpeechSeq2Seq.from_pretrained(tmpdir)

            self.assertTrue(model.use_merged)
            self.assertTrue(model.decoder_with_past is None)

            model.save_pretrained(tmpdir + "_save")
            save_path = os.path.join(tmpdir + "_save", ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir + "_save")
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_WITH_PAST_NAME in folder_contents)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name])
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        outputs = model.generate(inputs=features["input_features"])
        res = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        if use_merged is False:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, False)
        else:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, True)

        self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is False:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is True:
            self.assertTrue(onnx_model.decoder_with_past is None)

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        features = {
            "np": processor.feature_extractor(data, return_tensors="np"),
            "pt": processor.feature_extractor(data, return_tensors="pt"),
        }

        decoder_start_token_id = transformers_model.config.decoder_start_token_id
        decoder_inputs = {
            "np": {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id},
            "pt": {"decoder_input_ids": torch.ones((1, 1), dtype=torch.int64) * decoder_start_token_id},
        }

        with torch.no_grad():
            transformers_outputs = transformers_model(**features["pt"], **decoder_inputs["pt"])

        for input_type in ["pt", "np"]:
            onnx_outputs = onnx_model(**features[input_type], **decoder_inputs[input_type])

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        if model_arch == "speech_to_text":
            generation_length = 20
        else:
            generation_length = self.GENERATION_LENGTH

        with torch.no_grad():
            transformers_outputs = transformers_model.generate(
                **features["pt"],
                max_new_tokens=generation_length,
                min_new_tokens=generation_length,
                do_sample=False,
                num_beams=1,
            )

        onnx_outputs = onnx_model.generate(
            **features["pt"],
            max_new_tokens=generation_length,
            min_new_tokens=generation_length,
            do_sample=False,
            num_beams=1,
        )

        torch.testing.assert_close(torch.Tensor(onnx_outputs), transformers_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_speech_recognition(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer, _, feature_extractor = maybe_load_preprocessors(model_id)
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged
        )
        # Speech recogition generation
        pipe = pipeline(
            "automatic-speech-recognition",
            model=onnx_model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        data = self._generate_random_audio_data()
        outputs = pipe(data)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs["text"], str)

        if model_arch == "whisper":
            outputs = pipe(data, return_timestamps=True)
            self.assertTrue("chunks" in outputs)

            outputs = pipe(data, return_timestamps=False)
            self.assertTrue("chunks" not in outputs)

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        processor = get_preprocessor(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=onnx_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0,
        )

        data = self._generate_random_audio_data()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs["text"], str))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        processor = get_preprocessor(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=onnx_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0,
        )

        data = self._generate_random_audio_data()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs["text"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        if model_arch == "speech_to_text":
            generation_length = 20  # maximum length for the model
        else:
            generation_length = self.GENERATION_LENGTH

        model_with_pkv = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        outputs_model_with_pkv = model_with_pkv.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )

        model_without_pkv = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        outputs_model_without_pkv = model_without_pkv.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )

        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_merged_and_not_merged_models_outputs(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name + "_True",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": True,
        }
        self._setup(model_args)
        model_args = {
            "test_name": test_name + "_False",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": False,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        model_not_merged_dir = self.onnx_model_dirs[test_name + "_False"]
        model_merged_dir = self.onnx_model_dirs[test_name + "_True"]

        model_not_merged = ORTModelForSpeechSeq2Seq.from_pretrained(model_not_merged_dir)
        not_merged_onnx_path = Path(model_not_merged_dir, ONNX_DECODER_NAME)
        self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_not_merged.use_merged, False)

        model_merged = ORTModelForSpeechSeq2Seq.from_pretrained(model_merged_dir)
        merged_onnx_path = Path(model_merged_dir, ONNX_DECODER_MERGED_NAME)
        self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_merged.decoder_with_past, None)
        self.assertEqual(model_merged.use_merged, True)

        generation_length = 10

        outputs_model_not_merged = model_not_merged.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )
        outputs_model_merged = model_merged.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )

        torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={
                "cudnn_conv_algo_search": "DEFAULT",
            },
        )
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={
                "cudnn_conv_algo_search": "DEFAULT",
            },
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        inputs = processor([data] * 2, return_tensors="pt").to("cuda")
        inputs["decoder_input_ids"] = torch.ones((2, 1), dtype=torch.long).to("cuda")

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self,
        test_name: str,
        model_arch: str,
        use_cache: bool,
        use_merged: bool,
        num_beams: int,
    ):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        features = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model.generate(**features, num_beams=num_beams)
        io_outputs = io_model.generate(**features, num_beams=num_beams)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForImageToImageIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["swin2sr"]

    ORTMODEL_CLASS = ORTModelForImageToImage

    TASK = "image-to-image"

    def _get_sample_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def _get_preprocessors(self, model_id):
        image_processor = AutoImageProcessor.from_pretrained(model_id)

        return image_processor

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForImageToImage.from_pretrained(MODEL_NAMES["bert"], export=True)
        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        self.assertIsInstance(onnx_model.config, Swin2SRConfig)
        set_seed(SEED)

        transformers_model = AutoModelForImageToImage.from_pretrained(model_id)
        image_processor = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**features)

        onnx_outputs = onnx_model(**features)
        self.assertIsInstance(onnx_outputs, ImageSuperResolutionOutput)
        self.assertTrue("reconstruction" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.reconstruction, torch.Tensor)
        torch.testing.assert_close(
            onnx_outputs.reconstruction, transformers_outputs.reconstruction, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        outputs = onnx_model(**features)
        self.assertIsInstance(outputs, ImageSuperResolutionOutput)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_image_to_image(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-image",
            model=onnx_model,
            image_processor=image_processor,
        )
        data = self._get_sample_image()
        outputs = pipe(data)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs, Image.Image)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-image",
            model=onnx_model,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        self.assertIsInstance(outputs, Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-image",
            model=onnx_model,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        self.assertIsInstance(outputs, Image.Image)


class ORTModelForVision2SeqIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["vision-encoder-decoder", "trocr", "donut"]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForVision2Seq

    TASK = "image-to-text"

    GENERATION_LENGTH = 100

    ATOL = 1e-3
    RTOL = 1e-3

    def _get_sample_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def _get_preprocessors(self, model_id):
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        return image_processor, tokenizer

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForVision2Seq.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        image_processor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        outputs = model.generate(inputs=features["pixel_values"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        if use_merged is False:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, False)
        else:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, True)

        self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is False:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is True:
            self.assertTrue(onnx_model.decoder_with_past is None)

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        image_processor, tokenizer = self._get_preprocessors(model_id)
        transformers_model = AutoModelForVision2Seq.from_pretrained(model_id)

        data = self._get_sample_image()
        inputs = image_processor(data, return_tensors="pt")
        inputs["decoder_input_ids"] = tokenizer("This is a sample output", return_tensors="pt").input_ids

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, use_cache=use_cache)

        for input_type in ["pt", "np"]:
            inputs = image_processor(data, return_tensors=input_type)
            inputs["decoder_input_ids"] = tokenizer("This is a sample output", return_tensors=input_type).input_ids

            onnx_outputs = onnx_model(**inputs, use_cache=use_cache)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

            if use_cache:
                self.assertEqual(
                    len(onnx_outputs["past_key_values"]),
                    len(transformers_outputs["past_key_values"]),
                )
                for i in range(len(onnx_outputs["past_key_values"])):
                    self.assertEqual(
                        len(onnx_outputs["past_key_values"][i]),
                        len(transformers_outputs["past_key_values"][i]),
                    )
                    for j in range(len(onnx_outputs["past_key_values"][i])):
                        torch.testing.assert_close(
                            torch.Tensor(onnx_outputs["past_key_values"][i][j]),
                            transformers_outputs["past_key_values"][i][j],
                            atol=self.ATOL,
                            rtol=self.RTOL,
                        )

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_image_to_text(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        image_processor, tokenizer = self._get_preprocessors(model_id)

        # Speech recogition generation
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            feature_extractor=image_processor,  # for older versions of transformers
        )
        data = self._get_sample_image()
        outputs = pipe(data, max_new_tokens=10)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=False
        )
        image_processor, tokenizer = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=False
        )
        image_processor, tokenizer = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES[:1])
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        image_processor, _ = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        model_with_pkv = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )

        outputs_model_with_pkv = model_with_pkv.generate(
            **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )

        model_without_pkv = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )

        outputs_model_without_pkv = model_without_pkv.generate(
            **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )

        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL)
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + 1)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + 1)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )
        io_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._get_sample_image()
        image_processor, _ = self._get_preprocessors(model_id)
        pixel_values = image_processor([data] * 2, return_tensors="pt").pixel_values.to("cuda")
        decoder_start_token_id = onnx_model.config.decoder.bos_token_id
        decoder_input_ids = torch.full((2, 1), decoder_start_token_id, dtype=torch.long).to("cuda")

        onnx_outputs = onnx_model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        io_outputs = io_model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool, num_beams: int
    ):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._get_sample_image()
        image_processor, _ = self._get_preprocessors(model_id)
        features = image_processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model.generate(**features, num_beams=num_beams)
        io_outputs = io_model.generate(**features, num_beams=num_beams)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForCustomTasksIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "sbert": "optimum/sbert-all-MiniLM-L6-with-pooler",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        for input_type in ["pt", "np"]:
            tokens = tokenizer("This is a sample output", return_tensors=input_type)
            outputs = model(**tokens)
            self.assertIsInstance(outputs.pooler_output, self.TENSOR_ALIAS_TO_TYPE[input_type])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline_ort_model(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pipe.device, onnx_model.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, *args, **kwargs):
        _, model_id = args

        set_seed(SEED)
        onnx_model = ORTModelForCustomTasks.from_pretrained(
            model_id, use_io_binding=False, provider="CUDAExecutionProvider"
        )
        set_seed(SEED)
        io_model = ORTModelForCustomTasks.from_pretrained(
            model_id, use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("pooler_output" in io_outputs)
        self.assertIsInstance(io_outputs.pooler_output, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            onnx_outputs.pooler_output, io_outputs.pooler_output, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class ORTModelForPix2StructTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["pix2struct"]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForPix2Struct
    TASK = "image-to-text"  # is it fine as well with visual-question-answering?

    GENERATION_LENGTH = 100

    IMAGE = Image.open(
        requests.get(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
            stream=True,
        ).raw
    )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForPix2Struct.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_transformers_and_save(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForPix2Struct.from_pretrained(model_id, export=True, use_merged=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            save_path = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir)
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME not in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        task = "image-to-text-with-past"

        with tempfile.TemporaryDirectory() as tmpdir:
            main_export(model_id, tmpdir, task=task)

            model = ORTModelForPix2Struct.from_pretrained(tmpdir)

            self.assertTrue(model.use_merged)
            self.assertTrue(model.decoder_with_past is None)

            model.save_pretrained(tmpdir + "_save")
            save_path = os.path.join(tmpdir + "_save", ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir + "_save")
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_WITH_PAST_NAME in folder_contents)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        if use_merged is False:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
            self.assertFalse(onnx_model.use_merged)
        else:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
            self.assertTrue(onnx_model.use_merged)

        self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
        if use_cache is True and use_merged is False:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
        if use_cache is True and use_merged is True:
            self.assertTrue(onnx_model.decoder_with_past is None)

        set_seed(SEED)
        transformers_model = Pix2StructForConditionalGeneration.from_pretrained(model_id)

        preprocessor = get_preprocessor(model_id)
        questions = [
            "Who am I?",
            "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud and this is long long very long and super long my dear",
        ]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=questions, padding=True, return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(
                images=[self.IMAGE, self.IMAGE], text=questions, padding=True, return_tensors=input_type
            )

            onnx_outputs = onnx_model(**inputs)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_with_pkv = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        model_without_pkv = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )

        model_id = MODEL_NAMES[model_arch]
        preprocessor = get_preprocessor(model_id)
        question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")

        outputs_model_with_pkv = model_with_pkv.generate(
            **inputs, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )
        outputs_model_without_pkv = model_without_pkv.generate(
            **inputs, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )

        self.assertEqual(
            (outputs_model_with_pkv.shape[1], outputs_model_without_pkv.shape[1]),
            (
                inputs["decoder_input_ids"].shape[1] + self.GENERATION_LENGTH + 1,
                inputs["decoder_input_ids"].shape[1] + self.GENERATION_LENGTH + 1,
            ),
        )

        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_merged_and_not_merged_models_outputs(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name + "_True",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": True,
        }
        self._setup(model_args)
        model_args = {
            "test_name": test_name + "_False",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": False,
        }
        self._setup(model_args)

        model_not_merged = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name + "_False"])
        not_merged_onnx_path = Path(self.onnx_model_dirs[test_name + "_False"], ONNX_DECODER_NAME)
        self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_not_merged.use_merged, False)

        model_merged = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name + "_True"])
        merged_onnx_path = Path(self.onnx_model_dirs[test_name + "_True"], ONNX_DECODER_MERGED_NAME)
        self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_merged.decoder_with_past, None)
        self.assertEqual(model_merged.use_merged, True)

        model_id = MODEL_NAMES[model_arch]
        preprocessor = get_preprocessor(model_id)
        question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")

        outputs_model_not_merged = model_not_merged.generate(
            **inputs, max_new_tokens=self.GENERATION_LENGTH, min_new_tokens=self.GENERATION_LENGTH
        )
        outputs_model_merged = model_merged.generate(
            **inputs, max_new_tokens=self.GENERATION_LENGTH, min_new_tokens=self.GENERATION_LENGTH
        )

        torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        question = ["What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud", "Who are you?"]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=question, padding=True, return_tensors="pt").to(
            "cuda"
        )

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertTrue("encoder_last_hidden_state" in io_outputs)

        self.assertIsInstance(io_outputs.logits, torch.Tensor)
        self.assertIsInstance(io_outputs.encoder_last_hidden_state, torch.Tensor)

        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool, num_beams: int
    ):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        question = ["What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud", "Who are you?"]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=question, padding=True, return_tensors="pt").to(
            "cuda"
        )

        onnx_outputs = onnx_model.generate(**inputs, num_beams=num_beams)
        io_outputs = io_model.generate(**inputs, num_beams=num_beams)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)


class TestBothExportersORTModel(unittest.TestCase):
    @parameterized.expand(
        [
            ["question-answering", ORTModelForQuestionAnsweringIntegrationTest],
            ["text-classification", ORTModelForSequenceClassificationIntegrationTest],
            ["token-classification", ORTModelForTokenClassificationIntegrationTest],
            ["feature-extraction", ORTModelForFeatureExtractionIntegrationTest],
            ["multiple-choice", ORTModelForMultipleChoiceIntegrationTest],
            ["image-classification", ORTModelForImageClassificationIntegrationTest],
            ["semantic-segmentation", ORTModelForSemanticSegmentationIntegrationTest],
            ["text2text-generation", ORTModelForSeq2SeqLMIntegrationTest],
            ["automatic-speech-recognition", ORTModelForSpeechSeq2SeqIntegrationTest],
            ["audio-classification", ORTModelForAudioClassificationIntegrationTest],
            ["automatic-speech-recognition", ORTModelForCTCIntegrationTest],
            ["audio-xvector", ORTModelForAudioXVectorIntegrationTest],
            ["audio-frame-classification", ORTModelForAudioFrameClassificationIntegrationTest],
            ["image-to-image", ORTModelForImageToImageIntegrationTest],
        ]
    )
    def test_find_untested_architectures(self, task: str, test_class):
        supported_export_models = TasksManager.get_supported_model_type_for_task(task=task, exporter="onnx")
        tested_architectures = set(test_class.SUPPORTED_ARCHITECTURES)

        untested_architectures = set(supported_export_models) - tested_architectures
        if len(untested_architectures) > 0:
            logger.warning(
                f"For the task `{task}`, the ONNX export supports {supported_export_models}, but only {tested_architectures} are tested.\n"
                f"    Missing {untested_architectures}."
            )
