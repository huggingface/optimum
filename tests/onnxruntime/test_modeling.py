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
import time
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime
import pytest
import requests
import timm
import torch
from huggingface_hub.constants import default_cache_path
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageClassification,
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
from transformers.modeling_utils import no_init_weights
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import get_gpu_count, require_torch_gpu, slow
from utils_onnxruntime_tests import MODEL_NAMES, SEED, ORTModelTestMixin

from optimum.exporters import TasksManager
from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
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
    ORTModelForCausalLM,
    ORTModelForCTC,
    ORTModelForCustomTasks,
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
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
    ORTStableDiffusionPipeline,
)
from optimum.onnxruntime.base import ORTDecoderForSeq2Seq, ORTEncoder
from optimum.onnxruntime.modeling_diffusion import (
    ORTModelTextEncoder,
    ORTModelUnet,
    ORTModelVaeDecoder,
    ORTModelVaeEncoder,
)
from optimum.onnxruntime.modeling_ort import ORTModel
from optimum.pipelines import pipeline
from optimum.utils import (
    CONFIG_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
    logging,
)
from optimum.utils.testing_utils import grid_parameters, remove_directory, require_hf_token, require_ort_rocm


logger = logging.get_logger()


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class ORTModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.LOCAL_MODEL_PATH = "assets/onnx"
        self.ONNX_MODEL_ID = "philschmid/distilbert-onnx"
        self.TINY_ONNX_MODEL_ID = "fxmarty/resnet-tiny-beans"
        self.FAIL_ONNX_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.ONNX_SEQ2SEQ_MODEL_ID = "optimum/t5-small"
        self.LARGE_ONNX_SEQ2SEQ_MODEL_ID = "facebook/mbart-large-en-ro"
        self.TINY_ONNX_SEQ2SEQ_MODEL_ID = "fxmarty/sshleifer-tiny-mbart-onnx"
        self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID = "hf-internal-testing/tiny-random-OnnxStableDiffusionPipeline"

    def test_load_model_from_local_path(self):
        model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        # does not pass with ORTModel as it does not have export_feature attribute
        model = ORTModelForSequenceClassification.from_pretrained(
            "fxmarty/tiny-bert-sst2-distilled-subfolder", subfolder="my_subfolder", export=True
        )
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = ORTModel.from_pretrained("fxmarty/tiny-bert-sst2-distilled-onnx-subfolder", subfolder="my_subfolder")
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_from_hub_subfolder(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(
            "fxmarty/tiny-mbart-subfolder", subfolder="my_folder", export=True
        )
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

    def test_load_stable_diffusion_model_from_cache(self):
        _ = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)  # caching

        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, local_files_only=True
        )

        self.assertIsInstance(model.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(model.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(model.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(model.unet, ORTModelUnet)
        self.assertIsInstance(model.config, Dict)

    def test_load_stable_diffusion_model_from_empty_cache(self):
        dirpath = os.path.join(
            default_cache_path, "models--" + self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID.replace("/", "--")
        )
        remove_directory(dirpath)

        with self.assertRaises(Exception):
            _ = ORTStableDiffusionPipeline.from_pretrained(
                self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, local_files_only=True
            )

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_load_model_cuda_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
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

    def test_load_stable_diffusion_model_from_hub(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        self.assertIsInstance(model.text_encoder, ORTModelTextEncoder)
        self.assertIsInstance(model.vae_decoder, ORTModelVaeDecoder)
        self.assertIsInstance(model.vae_encoder, ORTModelVaeEncoder)
        self.assertIsInstance(model.unet, ORTModelUnet)
        self.assertIsInstance(model.config, Dict)

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_load_stable_diffusion_model_cuda_provider(self):
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="CUDAExecutionProvider"
        )
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.unet.session.get_providers(), model.providers)
        self.assertListEqual(model.text_encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_decoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_encoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_load_stable_diffusion_model_rocm_provider(self):
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="ROCMExecutionProvider"
        )
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.unet.session.get_providers(), model.providers)
        self.assertListEqual(model.text_encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_decoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_encoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    def test_load_stable_diffusion_model_cpu_provider(self):
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="CPUExecutionProvider"
        )
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])
        self.assertListEqual(model.unet.session.get_providers(), model.providers)
        self.assertListEqual(model.text_encoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_decoder.session.get_providers(), model.providers)
        self.assertListEqual(model.vae_encoder.session.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cpu"))

    def test_load_stable_diffusion_model_unknown_provider(self):
        with self.assertRaises(ValueError):
            ORTStableDiffusionPipeline.from_pretrained(
                self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="FooExecutionProvider"
            )

    def test_load_model_from_hub_without_onnx_model(self):
        with self.assertRaises(FileNotFoundError):
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

    def test_passing_session_options_stable_diffusion(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 3
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, session_options=options
        )
        self.assertEqual(model.unet.session.get_session_options().intra_op_num_threads, 3)
        self.assertEqual(model.text_encoder.session.get_session_options().intra_op_num_threads, 3)
        self.assertEqual(model.vae_decoder.session.get_session_options().intra_op_num_threads, 3)
        self.assertEqual(model.vae_encoder.session.get_session_options().intra_op_num_threads, 3)

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

    # test string device input for to()
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

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_passing_provider_options_stable_diffusion(self):
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID, provider="CUDAExecutionProvider"
        )
        self.assertEqual(
            model.unet.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        self.assertEqual(
            model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"],
            "1",
        )
        self.assertEqual(
            model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        self.assertEqual(
            model.vae_encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1"
        )
        model = ORTStableDiffusionPipeline.from_pretrained(
            self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID,
            provider="CUDAExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
        )
        self.assertEqual(
            model.unet.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0"
        )
        self.assertEqual(
            model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"],
            "0",
        )
        self.assertEqual(
            model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0"
        )
        self.assertEqual(
            model.vae_encoder.session.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0"
        )

    def test_stable_diffusion_model_on_cpu(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.unet.device, cpu)
        self.assertEqual(model.text_encoder.device, cpu)
        self.assertEqual(model.vae_decoder.device, cpu)
        self.assertEqual(model.vae_encoder.device, cpu)
        self.assertEqual(model.unet.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.vae_encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    # test string device input for to()
    def test_stable_diffusion_model_on_cpu_str(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        cpu = torch.device("cpu")
        model.to("cpu")
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.unet.device, cpu)
        self.assertEqual(model.text_encoder.device, cpu)
        self.assertEqual(model.vae_decoder.device, cpu)
        self.assertEqual(model.vae_encoder.device, cpu)
        self.assertEqual(model.unet.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.vae_encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_stable_diffusion_model_on_gpu(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.device, torch.device("cuda:0"))
        self.assertEqual(model.text_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.vae_encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_stable_diffusion_model_on_rocm_ep(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.device, torch.device("cuda:0"))
        self.assertEqual(model.text_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.vae_encoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    @unittest.skipIf(get_gpu_count() <= 1, "this test requires multi-gpu")
    def test_stable_diffusion_model_on_gpu_id(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to(torch.device("cuda:1"))
        self.assertEqual(model.unet.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to(1)
        self.assertEqual(model.unet.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to("cuda:1")
        self.assertEqual(model.unet.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.text_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_decoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")
        self.assertEqual(model.vae_encoder.session.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

    # test string device input for to()
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_stable_diffusion_model_on_gpu_str(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.device, torch.device("cuda:0"))
        self.assertEqual(model.text_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.vae_encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_stable_diffusion_model_on_rocm_ep_str(self):
        model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.device, torch.device("cuda:0"))
        self.assertEqual(model.text_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_decoder.device, torch.device("cuda:0"))
        self.assertEqual(model.vae_encoder.device, torch.device("cuda:0"))
        self.assertEqual(model.unet.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.text_encoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.vae_decoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertEqual(model.vae_encoder.session.get_providers()[0], "ROCMExecutionProvider")
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    def test_load_model_from_hub_private(self):
        token = os.environ.get("HF_HUB_READ_TOKEN", None)

        if token is None:
            self.skipTest("Test requires a token for fxmartyclone in the environment variable `HF_HUB_READ_TOKEN`.")

        model = ORTModelForCustomTasks.from_pretrained("optimum-internal-testing/tiny-random-phi-private", token=token)

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

    def test_save_stable_diffusion_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTStableDiffusionPipeline.from_pretrained(self.TINY_ONNX_STABLE_DIFFUSION_MODEL_ID)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(model.config_name, folder_contents)
            for subfoler in {
                DIFFUSION_MODEL_UNET_SUBFOLDER,
                DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            }:
                folder_contents = os.listdir(os.path.join(tmpdirname, subfoler))
                self.assertIn(ONNX_WEIGHTS_NAME, folder_contents)

    def test_save_load_ort_model_with_external_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], export=True)
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(ONNX_WEIGHTS_NAME, folder_contents)
            self.assertIn(ONNX_WEIGHTS_NAME + "_data", folder_contents)
            # verify loading from local folder works
            model = ORTModelForSequenceClassification.from_pretrained(tmpdirname, export=False)
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")
            remove_directory(tmpdirname)

    @parameterized.expand([(False,), (True,)])
    @pytest.mark.run_slow
    @slow
    def test_save_load_decoder_model_with_external_data(self, use_cache: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForCausalLM.from_pretrained(
                "gpt2-large", use_cache=use_cache, export=True, use_merged=False, use_io_binding=False
            )
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(ONNX_WEIGHTS_NAME + "_data" in folder_contents)
            self.assertFalse(use_cache ^ model.use_cache)

            # verify loading from local folder works
            model = ORTModelForCausalLM.from_pretrained(
                tmpdirname, use_cache=use_cache, export=False, use_io_binding=False
            )
            remove_directory(tmpdirname)

    @parameterized.expand([(False,), (True,)])
    def test_save_load_seq2seq_model_with_external_data(self, use_cache: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAMES["t5"], use_cache=use_cache, export=True)
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
            model = ORTModelForSeq2SeqLM.from_pretrained(tmpdirname, use_cache=use_cache, export=False)
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")
            remove_directory(tmpdirname)

    def test_save_load_stable_diffusion_model_with_external_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTStableDiffusionPipeline.from_pretrained(MODEL_NAMES["stable-diffusion"], export=True)
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            for subfoler in {
                DIFFUSION_MODEL_UNET_SUBFOLDER,
                DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            }:
                folder_contents = os.listdir(os.path.join(tmpdirname, subfoler))
                self.assertIn(ONNX_WEIGHTS_NAME, folder_contents)
                self.assertIn(ONNX_WEIGHTS_NAME + "_data", folder_contents)

            # verify loading from local folder works
            model = ORTStableDiffusionPipeline.from_pretrained(tmpdirname, export=False)
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")
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
    def test_push_decoder_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForCausalLM.from_pretrained(MODEL_NAMES["gpt2"], export=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["gpt2"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTModelForCausalLM.from_pretrained(
                MODEL_NAMES["gpt2"] + "-onnx",
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

    @require_hf_token
    def test_push_stable_diffusion_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTStableDiffusionPipeline.from_pretrained(MODEL_NAMES["stable-diffusion"], export=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["stable-diffusion"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTStableDiffusionPipeline.from_pretrained(
                MODEL_NAMES["stable-diffusion"] + "-onnx",
                export=False,
                token=os.environ.get("HF_AUTH_TOKEN", None),
            )
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    def test_trust_remote_code(self):
        model_id = "fxmarty/tiny-testing-gpt2-remote-code"
        ort_model = ORTModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inputs = tokenizer("My name is", return_tensors="pt")

        input_shape = inputs["input_ids"].shape
        inputs["position_ids"] = (
            torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        )

        with torch.inference_mode():
            pt_logits = pt_model(**inputs).logits

        ort_logits = ort_model(**inputs).logits

        self.assertTrue(
            torch.allclose(pt_logits, ort_logits, atol=1e-4), f" Maxdiff: {torch.abs(pt_logits - ort_logits).max()}"
        )


class ORTModelForQuestionAnsweringIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bart",
        "bert",
        # "big_bird",
        # "bigbird_pegasus",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
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
        "xlm_qa",
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForQuestionAnswering
    TASK = "question-answering"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForQuestionAnswering.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(
                torch.allclose(torch.Tensor(onnx_outputs.start_logits), transformers_outputs.start_logits, atol=1e-4)
            )
            self.assertTrue(
                torch.allclose(torch.Tensor(onnx_outputs.end_logits), transformers_outputs.end_logits, atol=1e-4)
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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForQuestionAnswering.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("start_logits" in io_outputs)
        self.assertTrue("end_logits" in io_outputs)
        self.assertIsInstance(io_outputs.start_logits, torch.Tensor)
        self.assertIsInstance(io_outputs.end_logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.start_logits, io_outputs.start_logits))
        self.assertTrue(torch.equal(onnx_outputs.end_logits, io_outputs.end_logits))

        gc.collect()


class ORTModelForMaskedLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        # "big_bird",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
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
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForMaskedLM
    TASK = "fill-mask"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForMaskedLM.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

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
        text = "The capital of France is [MASK]."
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
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch], use_io_binding=False).to(
            "cuda"
        )
        io_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch], use_io_binding=True).to(
            "cuda"
        )

        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token
        tokens = tokenizer([f"The capital of France is {MASK_TOKEN}."] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()


class ORTModelForSequenceClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bart",
        "bert",
        # "big_bird",
        # "bigbird_pegasus",
        "bloom",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
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
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForSequenceClassification
    TASK = "text-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()


class ORTModelForTokenClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        # "big_bird",
        "bloom",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
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
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForTokenClassification
    TASK = "token-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForTokenClassification.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

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
        "xlm_roberta",
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
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("last_hidden_state", onnx_outputs)
            self.assertIsInstance(onnx_outputs.last_hidden_state, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(
                torch.allclose(
                    torch.Tensor(onnx_outputs.last_hidden_state), transformers_outputs.last_hidden_state, atol=1e-4
                )
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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("last_hidden_state" in io_outputs)
        self.assertIsInstance(io_outputs.last_hidden_state, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.last_hidden_state, io_outputs.last_hidden_state))

        gc.collect()


class ORTModelForMultipleChoiceIntegrationTest(ORTModelTestMixin):
    # Multiple Choice tests are conducted on different models due to mismatch size in model's classifier
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        # "big_bird",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta_v2",
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
        "xlm_roberta",
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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMultipleChoice.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForMultipleChoice.from_pretrained(self.onnx_model_dirs[model_arch], use_io_binding=True).to(
            "cuda"
        )

        tokenizer = get_preprocessor(model_id)
        num_choices = 4
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        start = "The color of the sky is"
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

        inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()


class ORTModelForCausalLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "bloom",
        "codegen",
        "falcon",
        "gemma",
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "gptj",
        "llama",
        "mistral",
        "mpt",
        "phi3",
        "qwen2",
    ]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForCausalLM
    TASK = "text-generation"

    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    @parameterized.expand([(False,), (True,)])
    @pytest.mark.run_in_series
    # TODO: still gotta find out why this needs to be ran in series / why it fails in parallel
    # my guess is that the model surgery is happening in parallel and that's causing the issue
    def test_inference_old_onnx_model(self, use_cache):
        tokenizer = get_preprocessor("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        onnx_model = ORTModelForCausalLM.from_pretrained("optimum/gpt2", use_cache=use_cache, use_io_binding=use_cache)

        self.assertEqual(onnx_model.use_cache, use_cache)
        self.assertEqual(onnx_model.model_path.name, ONNX_DECODER_WITH_PAST_NAME if use_cache else ONNX_DECODER_NAME)

        text = "The capital of France is"
        tokens = tokenizer(text, return_tensors="pt")

        onnx_outputs = onnx_model.generate(
            **tokens, num_beams=1, do_sample=False, min_new_tokens=30, max_new_tokens=30
        )
        outputs = model.generate(**tokens, num_beams=1, do_sample=False, min_new_tokens=30, max_new_tokens=30)
        onnx_text_outputs = tokenizer.decode(onnx_outputs[0], skip_special_tokens=True)
        text_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertEqual(onnx_text_outputs, text_outputs)

    def test_load_model_from_hub_onnx(self):
        model = ORTModelForCausalLM.from_pretrained("fxmarty/onnx-tiny-random-gpt2-without-merge")

        self.assertFalse(model.use_merged)
        self.assertTrue(model.use_cache)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertEqual(model.onnx_paths[0].name, ONNX_DECODER_WITH_PAST_NAME)

        model = ORTModelForCausalLM.from_pretrained("fxmarty/onnx-tiny-random-gpt2-with-merge")

        self.assertTrue(model.use_merged)
        self.assertTrue(model.use_cache)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertEqual(model.onnx_paths[0].name, ONNX_DECODER_MERGED_NAME)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForCausalLM.from_pretrained(MODEL_NAMES["vit"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        task = "text-generation-with-past"

        if task not in TasksManager.get_supported_tasks_for_model_type(
            model_arch.replace("_", "-"), exporter="onnx", library_name="transformers"
        ):
            self.skipTest("Unsupported export case")

        with tempfile.TemporaryDirectory() as tmpdir:
            main_export(model_id, tmpdir, task=task, legacy=True)

            model = ORTModelForCausalLM.from_pretrained(tmpdir)

            self.assertTrue(model.use_merged)
            self.assertIsInstance(model.model, onnxruntime.InferenceSession)
            model.save_pretrained(tmpdir + "_save")
            save_path = os.path.join(tmpdir + "_save", ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir + "_save")
            self.assertNotIn(ONNX_DECODER_NAME, folder_contents)
            self.assertNotIn(ONNX_DECODER_WITH_PAST_NAME, folder_contents)
            self.assertNotIn(ONNX_WEIGHTS_NAME, folder_contents)

    @parameterized.expand(grid_parameters({**FULL_GRID, "num_beams": [1, 3]}))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, num_beams: int):
        use_io_binding = None
        if use_cache is False:
            use_io_binding = False

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_cache=use_cache,
            use_io_binding=use_io_binding,
        )

        model_path = Path(self.onnx_model_dirs[test_name], ONNX_WEIGHTS_NAME)
        self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
        self.assertFalse(onnx_model.use_merged)
        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        transformers_model = transformers_model.eval()
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        position_ids = None
        if model_arch.replace("_", "-") in MODEL_TYPES_REQUIRING_POSITION_IDS:
            input_shape = tokens["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        onnx_outputs = onnx_model(**tokens, position_ids=position_ids)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(
            torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4),
            f"Maxdiff: {(onnx_outputs.logits - transformers_outputs.logits).abs()}",
        )

        # Compare batched generation.
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokens = tokenizer(["Today is a nice day and I am longer", "This is me"], return_tensors="pt", padding=True)
        onnx_model.generation_config.eos_token_id = None
        transformers_model.generation_config.eos_token_id = None
        onnx_model.config.eos_token_id = None
        transformers_model.config.eos_token_id = None

        new_tokens = 30
        if model_arch == "falcon":
            # TODO: remove once https://github.com/huggingface/transformers/pull/26873 is released, falcon is broken in transformers
            new_tokens = 5

        onnx_outputs = onnx_model.generate(
            **tokens,
            num_beams=num_beams,
            do_sample=False,
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            eos_token_id=None,
        )

        transformers_outputs = transformers_model.generate(
            **tokens,
            num_beams=num_beams,
            do_sample=False,
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            eos_token_id=None,
        )

        self.assertTrue(torch.allclose(onnx_outputs, transformers_outputs))

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_ort_model(self, test_name: str, model_arch: str, use_cache: bool):
        use_io_binding = None
        if use_cache is False:
            use_io_binding = False

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_cache=use_cache,
            use_io_binding=use_io_binding,
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("text-generation")
        text = "My Name is Philipp and i live"
        outputs = pipe(text)

        # compare model output class
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(self.onnx_model_dirs[test_name])

        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live"
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(self.onnx_model_dirs[test_name])

        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live"
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        gc.collect()

    # TRT EP compile time can be long, so we don't test all archs
    @parameterized.expand(grid_parameters({"model_arch": ["gpt2"], "use_cache": [True, False]}))
    @require_torch_gpu
    @pytest.mark.trt_ep_test
    def test_pipeline_on_trt_execution_provider(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        with tempfile.TemporaryDirectory() as engine_cache_dir:
            provider_options = {"trt_engine_cache_enable": True, "trt_engine_cache_path": engine_cache_dir}

            model_id = MODEL_NAMES[model_arch]
            onnx_model = ORTModelForCausalLM.from_pretrained(
                self.onnx_model_dirs[test_name],
                provider="TensorrtExecutionProvider",
                provider_options=provider_options,
                use_cache=use_cache,
            )

            tokenizer = get_preprocessor(model_id)
            # build engine for a short sequence
            text = ["short"]
            encoded_input = tokenizer(
                text, return_tensors="pt", return_token_type_ids=False if model_arch == "llama" else None
            ).to("cuda")
            _ = onnx_model(**encoded_input)

            # build engine for a long sequence
            text = [" a very long input just for demo purpose, this is very long" * 10]
            encoded_input = tokenizer(
                text, return_tensors="pt", return_token_type_ids=False if model_arch == "llama" else None
            ).to("cuda")
            _ = onnx_model(**encoded_input)

            pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer, device=0)
            text = "My Name is Philipp and i live"

            outputs = pipe(text)
            # check model device
            self.assertEqual(pipe.model.device.type.lower(), "cuda")
            # compare model output class
            self.assertTrue(isinstance(outputs[0]["generated_text"], str))
            self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

            encoded_input = tokenizer(
                ["Replace me by any text you'd like."],
                return_tensors="pt",
                return_token_type_ids=False if model_arch == "llama" else None,
            ).to("cuda")
            _ = onnx_model.generate(**encoded_input)

            gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.cuda_ep_test  # mark as GPU test as well to run the without/with cache timing test on the slow tests
    def test_compare_with_and_without_past_key_values(self, model_arch):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        text = "My Name is Philipp and i live"
        tokens = tokenizer(text, return_tensors="pt", return_token_type_ids=False if model_arch == "llama" else None)

        model_with_pkv = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True, use_io_binding=False
        )
        _ = model_with_pkv.generate(**tokens)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False, use_io_binding=False
        )
        _ = model_without_pkv.generate(**tokens)  # warmup
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], tokens["input_ids"].shape[1] + self.GENERATION_LENGTH)
        self.assertEqual(outputs_model_without_pkv.shape[1], tokens["input_ids"].shape[1] + self.GENERATION_LENGTH)

        if os.environ.get("TEST_LEVEL", 0) == "1":
            self.assertTrue(
                without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
                f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
                f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
            )

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_merged_and_not_merged_models_outputs(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name + "_False",
            "model_arch": model_arch,
            "use_cache": use_cache,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        text = "My Name is Philipp and i live"
        tokens = tokenizer(text, return_tensors="pt", return_token_type_ids=False if model_arch == "llama" else None)
        model_not_merged_dir = self.onnx_model_dirs[test_name + "_False"]

        model_not_merged = ORTModelForCausalLM.from_pretrained(model_not_merged_dir)
        not_merged_onnx_path = Path(model_not_merged_dir, ONNX_WEIGHTS_NAME)
        self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
        self.assertFalse(model_not_merged.use_merged)

        model_merged_dir = Path(Path(model_not_merged_dir).parents[0], "merged")
        task = model_not_merged.export_feature
        if use_cache:
            task += "-with-past"

        main_export(
            model_id,
            output=model_merged_dir,
            task=task,
            no_post_process=False,
            legacy=True,
        )

        model_merged = ORTModelForCausalLM.from_pretrained(model_merged_dir)
        merged_onnx_path = Path(model_merged_dir, ONNX_DECODER_MERGED_NAME)
        self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
        self.assertTrue(model_merged.use_merged)

        outputs_model_not_merged = model_not_merged.generate(**tokens)
        outputs_model_merged = model_merged.generate(**tokens)

        self.assertTrue(torch.equal(outputs_model_merged, outputs_model_not_merged))

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
        onnx_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        position_ids = None
        if model_arch.replace("_", "-") in MODEL_TYPES_REQUIRING_POSITION_IDS:
            input_shape = tokens["input_ids"].shape
            position_ids = (
                torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).expand(2, input_shape[-1]).to("cuda")
            )

        onnx_outputs = onnx_model(**tokens, position_ids=position_ids)
        io_outputs = io_model(**tokens, position_ids=position_ids)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=False).to(
            "cuda"
        )
        io_model = ORTModelForCausalLM.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=True).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
            return_token_type_ids=False if model_arch == "llama" else None,
        ).to("cuda")
        onnx_outputs = onnx_model.generate(**tokens)
        io_outputs = io_model.generate(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs, io_outputs))

        gc.collect()


class ORTModelForImageClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "beit",
        "convnext",
        "convnextv2",
        "data2vec_vision",
        "deit",
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

    TIMM_SUPPORTED_ARCHITECTURES = ["default-timm-config"]

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
            _ = ORTModelForImageClassification.from_pretrained(MODEL_NAMES["t5"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(TIMM_SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @pytest.mark.timm_test
    @slow
    def test_compare_to_timm(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            onnx_model = ORTModelForImageClassification.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, model_arch)
            )

            self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
            self.assertIsInstance(onnx_model.config, PretrainedConfig)

            set_seed(SEED)
            timm_model = timm.create_model(model_id, pretrained=True)
            timm_model = timm_model.eval()

            # get model specific transforms (normalization, resize)
            data_config = timm.data.resolve_model_data_config(timm_model)
            transforms = timm.data.create_transform(**data_config, is_training=False)

            url = (
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
            image = Image.open(requests.get(url, stream=True).raw)
            inputs = transforms(image).unsqueeze(0)

            with torch.no_grad():
                timm_outputs = timm_model(inputs)

            for input_type in ["pt", "np"]:
                if input_type == "np":
                    inputs = inputs.cpu().detach().numpy()
                onnx_outputs = onnx_model(inputs)

                self.assertIn("logits", onnx_outputs)
                self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

                # compare tensor outputs
                self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), timm_outputs, atol=1e-4))

        gc.collect()

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), trtfs_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor)
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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(
            torch.allclose(onnx_outputs.logits, io_outputs.logits, atol=1e-4),
            f" Maxdiff: {torch.abs(onnx_outputs.logits - io_outputs.logits).max()}",
        )

        gc.collect()


class ORTModelForSemanticSegmentationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ("segformer", "dpt")

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForSemanticSegmentation
    TASK = "semantic-segmentation"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSemanticSegmentation.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), trtfs_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(self.onnx_model_dirs[model_arch])
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-segmentation", model=onnx_model, feature_extractor=preprocessor)
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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(
            torch.allclose(onnx_outputs.logits, io_outputs.logits, atol=1e-4),
            f" Maxdiff: {torch.abs(onnx_outputs.logits - io_outputs.logits).max()}",
        )

        gc.collect()


class ORTModelForAudioClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "audio_spectrogram_transformer",
        "data2vec_audio",
        "hubert",
        "sew",
        "sew_d",
        "unispeech",
        "unispeech_sat",
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
            _ = ORTModelForAudioClassification.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        processor = AutoFeatureExtractor.from_pretrained(model_id)
        data = self._generate_random_audio_data()

        input_values = processor(data, return_tensors="pt")
        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, io_outputs.logits, atol=1e-4))

        gc.collect()


class ORTModelForCTCIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "data2vec_audio",
        "hubert",
        "sew",
        "sew_d",
        "unispeech",
        "unispeech_sat",
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
            _ = ORTModelForCTC.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

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
        ).to("cuda")
        onnx_model.use_io_binding = False
        io_model = ORTModelForCTC.from_pretrained(self.onnx_model_dirs[model_arch], use_io_binding=True).to("cuda")

        processor = AutoFeatureExtractor.from_pretrained(model_id)
        data = self._generate_random_audio_data()
        input_values = processor(data, return_tensors="pt")
        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), io_outputs.logits, atol=1e-1))

        gc.collect()


class ORTModelForAudioXVectorIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "data2vec_audio",
        "unispeech_sat",
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
            _ = ORTModelForAudioXVector.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(
                torch.allclose(torch.Tensor(onnx_outputs.embeddings), transformers_outputs.embeddings, atol=1e-4)
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
            self.onnx_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForAudioXVector.from_pretrained(self.onnx_model_dirs[model_arch], use_io_binding=True).to(
            "cuda"
        )

        processor = AutoFeatureExtractor.from_pretrained(model_id)
        data = self._generate_random_audio_data()

        input_values = processor(data, return_tensors="pt")
        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)
        self.assertIsInstance(io_outputs.embeddings, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, io_outputs.logits, atol=1e-4))
        self.assertTrue(torch.allclose(onnx_outputs.embeddings, io_outputs.embeddings, atol=1e-4))
        gc.collect()


class ORTModelForAudioFrameClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "data2vec_audio",
        "unispeech_sat",
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
            _ = ORTModelForAudioFrameClassification.from_pretrained(MODEL_NAMES["t5"], export=True)

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
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()


class ORTModelForSeq2SeqLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "bart",
        # "bigbird_pegasus",
        "blenderbot",
        "blenderbot_small",
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
    SPEEDUP_CACHE = 1.1

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

    @pytest.mark.run_in_series
    def test_inference_old_onnx_model(self):
        tokenizer = get_preprocessor("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained("optimum/t5-small")

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")

        outputs = model.generate(**tokens, num_beams=1, do_sample=False, min_new_tokens=30, max_new_tokens=30)
        onnx_outputs = onnx_model.generate(
            **tokens, num_beams=1, do_sample=False, min_new_tokens=30, max_new_tokens=30
        )
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
            model_arch.replace("_", "-"), exporter="onnx", library_name="transformers"
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

        if task not in TasksManager.get_supported_tasks_for_model_type(model_arch.replace("_", "-"), exporter="onnx"):
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
            tokens = tokenizer("This is a sample output", return_tensors="pt")
            decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id

            decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens, **decoder_inputs)

            for input_type in ["pt", "np"]:
                tokens = tokenizer("This is a sample output", return_tensors=input_type)

                if input_type == "np":
                    decoder_inputs = {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id}

                onnx_outputs = onnx_model(**tokens, **decoder_inputs)

                self.assertTrue("logits" in onnx_outputs)
                self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

                # Compare tensor outputs
                self.assertTrue(
                    torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4)
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
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["generated_text"], str)

            # Summarization
            pipe = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["summary_text"], str)

            # Translation
            pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["translation_text"], str)

        gc.collect()

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
    @pytest.mark.cuda_ep_test  # mark as GPU test as well to run the without/with cache timing test on the slow tests
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        if model_arch == "m2m_100":
            self.skipTest("m2m_100 comparison with/without pkv fail or is not supported")
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
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
            text = "This is a sample output"
            tokens = tokenizer(text, return_tensors="pt")
            model_with_pkv = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, model_arch + "_True"), use_cache=True
            )

            _ = model_with_pkv.generate(**tokens)  # warmup
            with Timer() as with_pkv_timer:
                outputs_model_with_pkv = model_with_pkv.generate(
                    **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
                )

            model_without_pkv = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, model_arch + "_False"), use_cache=False
            )
            _ = model_without_pkv.generate(**tokens)  # warmup
            with Timer() as without_pkv_timer:
                outputs_model_without_pkv = model_without_pkv.generate(
                    **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
                )

            self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
            self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + 1)
            self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + 1)

            if os.environ.get("TEST_LEVEL", 0) == "1":
                self.assertTrue(
                    without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
                    f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
                    f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
                )

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

            self.assertTrue(torch.equal(outputs_model_merged, outputs_model_not_merged))

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
                self._get_onnx_model_dir(model_id, model_arch, test_name), use_io_binding=False, use_cache=use_cache
            ).to("cuda")
            io_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name), use_io_binding=True, use_cache=use_cache
            ).to("cuda")

            self.assertFalse(onnx_model.use_io_binding)
            self.assertTrue(io_model.use_io_binding)

            tokenizer = get_preprocessor(model_id)
            tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")
            decoder_start_token_id = onnx_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id

            decoder_inputs = {"decoder_input_ids": torch.ones((2, 1), dtype=torch.long) * decoder_start_token_id}

            onnx_outputs = onnx_model(**tokens, **decoder_inputs)
            io_outputs = io_model(**tokens, **decoder_inputs)

            self.assertTrue("logits" in io_outputs)
            self.assertIsInstance(io_outputs.logits, torch.Tensor)

            # compare tensor outputs
            self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

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
                self._get_onnx_model_dir(model_id, model_arch, test_name), use_io_binding=False, use_cache=use_cache
            ).to("cuda")
            io_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name), use_io_binding=True, use_cache=use_cache
            ).to("cuda")

            tokenizer = get_preprocessor(model_id)
            tokens = tokenizer("This is a sample output", return_tensors="pt").to("cuda")
            onnx_outputs = onnx_model.generate(**tokens, num_beams=num_beams)
            io_outputs = io_model.generate(**tokens, num_beams=num_beams)

            # compare tensor outputs
            self.assertTrue(torch.equal(onnx_outputs, io_outputs))

        gc.collect()


class ORTModelForSpeechSeq2SeqIntegrationTest(ORTModelTestMixin):
    # TODO: speech_to_text should be tested
    SUPPORTED_ARCHITECTURES = ["whisper", "speech_to_text"]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForSpeechSeq2Seq
    TASK = "automatic-speech-recognition"

    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 18736), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)

        return audio_data

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_transformers_and_save(self, model_arch):
        if "automatic-speech-recognition-with-past" not in TasksManager.get_supported_tasks_for_model_type(
            model_arch.replace("_", "-"), exporter="onnx", library_name="transformers"
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

        if task not in TasksManager.get_supported_tasks_for_model_type(model_arch.replace("_", "-"), exporter="onnx"):
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

        features = processor.feature_extractor(data, return_tensors="pt")

        decoder_start_token_id = transformers_model.config.decoder_start_token_id
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

        with torch.no_grad():
            transformers_outputs = transformers_model(**features, **decoder_inputs)

        for input_type in ["pt", "np"]:
            features = processor.feature_extractor(data, return_tensors=input_type)

            if input_type == "np":
                decoder_inputs = {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id}

            onnx_outputs = onnx_model(**features, **decoder_inputs)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_speech_recognition(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache, "use_merged": bool}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        processor = get_preprocessor(model_id)

        # Speech recogition generation
        pipe = pipeline(
            "automatic-speech-recognition",
            model=onnx_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
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
    @pytest.mark.cuda_ep_test  # mark as GPU test as well to run the without/with cache timing test on the slow tests
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        model_with_pkv = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )

        generation_length = self.GENERATION_LENGTH
        self.GENERATION_LENGTH = 10
        _ = model_with_pkv.generate(**features)  # warpup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        _ = model_without_pkv.generate(**features)  # warpup
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + 1)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + 1)
        self.GENERATION_LENGTH = generation_length
        if os.environ.get("TEST_LEVEL", 0) == "1":
            self.assertTrue(
                without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
                f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
                f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
            )

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

        generation_length = self.GENERATION_LENGTH
        self.GENERATION_LENGTH = 10

        outputs_model_not_merged = model_not_merged.generate(
            **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )
        outputs_model_merged = model_merged.generate(
            **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )

        self.GENERATION_LENGTH = generation_length

        self.assertTrue(torch.equal(outputs_model_merged, outputs_model_not_merged))

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

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=True).to(
            "cuda"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor([data] * 2, return_tensors="pt").to("cuda")

        decoder_start_token_id = onnx_model.config.decoder_start_token_id
        decoder_inputs = {"decoder_input_ids": torch.ones((2, 1), dtype=torch.long) * decoder_start_token_id}

        onnx_outputs = onnx_model(**features, **decoder_inputs)
        io_outputs = io_model(**features, **decoder_inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 5],
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
            self.onnx_model_dirs[test_name], use_io_binding=False
        ).to("cuda")
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=True).to(
            "cuda"
        )

        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model.generate(**features, num_beams=num_beams)
        io_outputs = io_model.generate(**features, num_beams=num_beams)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs, io_outputs))

        gc.collect()


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
    SPEEDUP_CACHE = 1.1

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
        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = feature_extractor(data, return_tensors="pt")

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
        transformers_model = AutoModelForVision2Seq.from_pretrained(model_id)
        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()

        start_token = "<s>"
        decoder_start_token_id = tokenizer.encode(start_token)[0]

        extra_inputs = [{}, {}]

        for extra_inps in extra_inputs:
            features = feature_extractor(data, return_tensors="pt")
            decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

            with torch.no_grad():
                transformers_outputs = transformers_model(**features, **decoder_inputs, **extra_inps, use_cache=True)
            for input_type in ["pt", "np"]:
                features = feature_extractor(data, return_tensors=input_type)

                if input_type == "np":
                    decoder_inputs = {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id}

                    if "past_key_values" in extra_inps:
                        del extra_inps["past_key_values"]  # test only with pytorch

                onnx_outputs = onnx_model(**features, **decoder_inputs, **extra_inps)

                self.assertTrue("logits" in onnx_outputs)
                self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])
                self.assertTrue(
                    torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-3)
                )

                if use_cache:
                    self.assertEqual(
                        len(onnx_outputs["past_key_values"]), len(transformers_outputs["past_key_values"])
                    )
                    self.assertEqual(
                        len(onnx_outputs["past_key_values"][0]), len(transformers_outputs["past_key_values"][0])
                    )
                    for i in range(len(onnx_outputs["past_key_values"])):
                        print(onnx_outputs["past_key_values"][i])
                        for ort_pkv, trfs_pkv in zip(
                            onnx_outputs["past_key_values"][i], transformers_outputs["past_key_values"][i]
                        ):
                            ort_pkv = torch.Tensor(ort_pkv)
                            self.assertTrue(
                                torch.allclose(ort_pkv, trfs_pkv, atol=1e-3),
                                f" Maxdiff: {torch.abs(ort_pkv - trfs_pkv).max()}",
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
        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        # Speech recogition generation
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )
        data = self._get_sample_image()
        outputs = pipe(data)
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
        feature_extractor, tokenizer = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
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
        feature_extractor, tokenizer = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES[:1])
    @pytest.mark.cuda_ep_test  # mark as GPU test as well to run the without/with cache timing test on the slow tests
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = feature_extractor(data, return_tensors="pt")

        model_with_pkv = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        _ = model_with_pkv.generate(**features)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        _ = model_without_pkv.generate(**features)  # warmup
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + 1)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + 1)

        if os.environ.get("TEST_LEVEL", 0) == "1":
            self.assertTrue(
                without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
                f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
                f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
            )

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

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=False).to(
            "cuda"
        )
        io_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=True).to(
            "cuda"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        pixel_values = feature_extractor([data] * 2, return_tensors="pt").pixel_values.to("cuda")

        decoder_start_token_id = onnx_model.config.decoder.bos_token_id
        decoder_input_ids = torch.full((2, 1), decoder_start_token_id, dtype=torch.long).to("cuda")

        onnx_outputs = onnx_model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        io_outputs = io_model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

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
        onnx_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=False).to(
            "cuda"
        )
        io_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=True).to(
            "cuda"
        )

        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = feature_extractor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model.generate(**features, num_beams=num_beams)
        io_outputs = io_model.generate(**features, num_beams=num_beams)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs, io_outputs))

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
        model_arch, model_id = args
        set_seed(SEED)
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id, use_io_binding=False).to("cuda")
        set_seed(SEED)
        io_model = ORTModelForCustomTasks.from_pretrained(model_id, use_io_binding=True).to("cuda")
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("pooler_output" in io_outputs)
        self.assertIsInstance(io_outputs.pooler_output, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.pooler_output, io_outputs.pooler_output))

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
    SPEEDUP_CACHE = 1.1

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

        if use_cache is False:
            self.skipTest("skip")

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
        questions = [
            "Who am I?",
            "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud and this is long long very long and super long my dear",
        ]

        transformers_model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)

        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=questions, padding=True, return_tensors="pt")
        del inputs["decoder_attention_mask"]
        del inputs["decoder_input_ids"]

        decoder_start_token_id = transformers_model.config.decoder_start_token_id
        decoder_inputs = {
            "decoder_input_ids": torch.ones((2, 1), dtype=torch.long) * decoder_start_token_id,
            "decoder_attention_mask": torch.ones((2, 1), dtype=torch.int64),
        }

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, **decoder_inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(
                images=[self.IMAGE, self.IMAGE], text=questions, padding=True, return_tensors=input_type
            )
            del inputs["decoder_attention_mask"]
            del inputs["decoder_input_ids"]

            if input_type == "np":
                decoder_inputs = {
                    "decoder_input_ids": np.ones((2, 1), dtype=np.int64) * decoder_start_token_id,
                    "decoder_attention_mask": np.ones((2, 1), dtype=np.int64),
                }

            onnx_outputs = onnx_model(**inputs, **decoder_inputs)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.cuda_ep_test  # mark as GPU test as well to run the without/with cache timing test on the slow tests
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        if model_arch == "m2m_100":
            return  # TODO: this test is failing for m2m_100
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        preprocessor = get_preprocessor(model_id)

        question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")
        del inputs["decoder_attention_mask"]
        del inputs["decoder_input_ids"]

        model_with_pkv = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )

        _ = model_with_pkv.generate(**inputs)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **inputs, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        _ = model_without_pkv.generate(**inputs)  # warmup
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **inputs, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + 1)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + 1)

        if os.environ.get("TEST_LEVEL", 0) == "1":
            self.assertTrue(
                without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
                f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
                f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
            )

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
        preprocessor = get_preprocessor(model_id)

        question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")
        del inputs["decoder_attention_mask"]
        del inputs["decoder_input_ids"]

        model_not_merged_dir = self.onnx_model_dirs[test_name + "_False"]
        model_merged_dir = self.onnx_model_dirs[test_name + "_True"]

        model_not_merged = ORTModelForPix2Struct.from_pretrained(model_not_merged_dir)
        not_merged_onnx_path = Path(model_not_merged_dir, ONNX_DECODER_NAME)
        self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_not_merged.use_merged, False)

        model_merged = ORTModelForPix2Struct.from_pretrained(model_merged_dir)
        merged_onnx_path = Path(model_merged_dir, ONNX_DECODER_MERGED_NAME)
        self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_merged.decoder_with_past, None)
        self.assertEqual(model_merged.use_merged, True)

        outputs_model_not_merged = model_not_merged.generate(**inputs)
        outputs_model_merged = model_merged.generate(**inputs)

        self.assertTrue(torch.equal(outputs_model_merged, outputs_model_not_merged))

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
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

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=False)
        io_model = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=True)

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)

        question = [
            "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud and this is even longer and longer and longer and longer and hey",
            "Who are you?",
        ]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=question, padding=True, return_tensors="pt")
        del inputs["decoder_attention_mask"]
        del inputs["decoder_input_ids"]
        decoder_start_token_id = onnx_model.config.decoder_start_token_id
        decoder_inputs = {
            "decoder_input_ids": torch.ones((2, 1), dtype=torch.long) * decoder_start_token_id,
            "decoder_attention_mask": torch.ones((2, 1), dtype=torch.int64),
        }

        onnx_outputs = onnx_model(**inputs, **decoder_inputs)
        io_outputs = io_model(**inputs, **decoder_inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        self.assertTrue(torch.allclose(onnx_outputs.logits, io_outputs.logits, atol=1e-4))

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
        onnx_model = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=False)
        io_model = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name], use_io_binding=True)

        preprocessor = get_preprocessor(model_id)

        question = ["What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud", "Who are you?"]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=question, padding=True, return_tensors="pt")
        del inputs["decoder_attention_mask"]
        del inputs["decoder_input_ids"]
        onnx_outputs = onnx_model.generate(**inputs, num_beams=num_beams)
        io_outputs = io_model.generate(**inputs, num_beams=num_beams)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs, io_outputs))

        gc.collect()


class TestBothExportersORTModel(unittest.TestCase):
    @parameterized.expand(
        [
            ["question-answering", ORTModelForQuestionAnsweringIntegrationTest],
            ["text-classification", ORTModelForSequenceClassificationIntegrationTest],
            ["token-classification", ORTModelForTokenClassificationIntegrationTest],
            ["feature-extraction", ORTModelForFeatureExtractionIntegrationTest],
            ["multiple-choice", ORTModelForMultipleChoiceIntegrationTest],
            ["text-generation", ORTModelForCausalLMIntegrationTest],
            ["image-classification", ORTModelForImageClassificationIntegrationTest],
            ["semantic-segmentation", ORTModelForSemanticSegmentationIntegrationTest],
            ["text2text-generation", ORTModelForSeq2SeqLMIntegrationTest],
            ["automatic-speech-recognition", ORTModelForSpeechSeq2SeqIntegrationTest],
            ["audio-classification", ORTModelForAudioClassificationIntegrationTest],
            ["automatic-speech-recognition", ORTModelForCTCIntegrationTest],
            ["audio-xvector", ORTModelForAudioXVectorIntegrationTest],
            ["audio-frame-classification", ORTModelForAudioFrameClassificationIntegrationTest],
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
