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
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from typing import Dict

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTokenClassification,
    AutoTokenizer,
    MBartForConditionalGeneration,
    PretrainedConfig,
    set_seed,
)
from transformers.modeling_utils import no_init_weights
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import get_gpu_count, require_torch_gpu

import onnx
import onnxruntime
import requests
from huggingface_hub.constants import default_cache_path
from optimum.exporters import TasksManager
from optimum.onnxruntime import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    ONNX_WEIGHTS_NAME,
    ORTModelForCausalLM,
    ORTModelForCustomTasks,
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForMultipleChoice,
    ORTModelForQuestionAnswering,
    ORTModelForSemanticSegmentation,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTModelForSpeechSeq2Seq,
    ORTModelForTokenClassification,
)
from optimum.onnxruntime.modeling_decoder import ORTDecoder
from optimum.onnxruntime.modeling_ort import ORTModel
from optimum.onnxruntime.modeling_seq2seq import ORTDecoder as ORTSeq2SeqDecoder
from optimum.onnxruntime.modeling_seq2seq import ORTEncoder
from optimum.pipelines import pipeline
from optimum.utils import CONFIG_NAME, logging
from optimum.utils.testing_utils import grid_parameters, require_hf_token
from parameterized import parameterized


logger = logging.get_logger()

MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "bart": "hf-internal-testing/tiny-random-bart",
    "big_bird": "hf-internal-testing/tiny-random-BigBirdModel",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot_small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "clip": "hf-internal-testing/tiny-random-CLIPModel",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    "codegen": "hf-internal-testing/tiny-random-CodeGenModel",
    "data2vec_text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec_vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec_audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",
    "deberta_v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "detr": "hf-internal-testing/tiny-random-detr",
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "groupvit": "hf-internal-testing/tiny-random-groupvit",
    "ibert": "hf-internal-testing/tiny-random-IBertModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "layoutlm": "hf-internal-testing/tiny-random-LayoutLMModel",
    "layoutlmv3": "hf-internal-testing/tiny-random-LayoutLMv3Model",
    "longt5": "hf-internal-testing/tiny-random-LongT5Model",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "marian": "sshleifer/tiny-marian-en-de",  # hf-internal-testing ones are broken
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mt5": "lewtun/tiny-random-mt5",
    "pegasus": "hf-internal-testing/tiny-random-PegasusModel",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "squeezebert": "hf-internal-testing/tiny-random-SqueezeBertModel",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "t5": "hf-internal-testing/tiny-random-t5",
    "vit": "hf-internal-testing/tiny-random-vit",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
    "whisper": "openai/whisper-tiny.en",  # hf-internal-testing ones are broken
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "wavlm": "hf-internal-testing/tiny-random-wavlm",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew_d": "hf-internal-testing/tiny-random-SEWDModel",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech_sat": "hf-internal-testing/tiny-random-unispeech-sat",
    "audio_spectrogram_transformer": "Ericwang/tiny-random-ast",
    "speech_to_text": "hf-internal-testing/tiny-random-Speech2TextModel",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm_roberta": "hf-internal-testing/tiny-xlm-roberta",
}

SEED = 42


class ORTModelTestMixin(unittest.TestCase):
    ARCH_MODEL_MAP = {}

    @classmethod
    def setUpClass(cls):
        cls.onnx_model_dirs = {}

    def _setup(self, model_args: Dict):
        """
        Exports the PyTorch models to ONNX ahead of time to avoid multiple exports during the tests.
        We don't use unittest setUpClass, in order to still be able to run individual tests.
        """
        model_arch = model_args["model_arch"]
        model_arch_and_params = model_args["test_name"]

        if model_arch_and_params not in self.onnx_model_dirs:
            # model_args will contain kwargs to pass to ORTModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")

            model_id = (
                self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
            )
            set_seed(SEED)
            onnx_model = self.ORTMODEL_CLASS.from_pretrained(model_id, **model_args, from_transformers=True)

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            onnx_model.save_pretrained(model_dir)
            self.onnx_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.onnx_model_dirs.items():
            shutil.rmtree(dir_path)


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

    def test_load_model_from_local_path(self):
        model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        # does not pass with ORTModel as it does not have export_feature attribute
        model = ORTModelForSequenceClassification.from_pretrained(
            "fxmarty/tiny-bert-sst2-distilled-subfolder", subfolder="my_subfolder", from_transformers=True
        )
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = ORTModel.from_pretrained("fxmarty/tiny-bert-sst2-distilled-onnx-subfolder", subfolder="my_subfolder")
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_from_hub_subfolder(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(
            "fxmarty/tiny-mbart-subfolder", subfolder="my_folder", from_transformers=True
        )
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.decoder_with_past, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = ORTModelForSeq2SeqLM.from_pretrained("fxmarty/tiny-mbart-onnx-subfolder", subfolder="my_folder")
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.decoder_with_past, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_cache(self):
        _ = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID)  # caching

        model = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)

        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.TINY_ONNX_MODEL_ID.replace("/", "--"))

        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with self.assertRaises(Exception):
            _ = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)

    def test_load_seq2seq_model_from_cache(self):
        _ = ORTModelForSeq2SeqLM.from_pretrained(self.TINY_ONNX_SEQ2SEQ_MODEL_ID)  # caching

        model = ORTModelForSeq2SeqLM.from_pretrained(self.TINY_ONNX_SEQ2SEQ_MODEL_ID, local_files_only=True)

        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.decoder_with_past, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.TINY_ONNX_SEQ2SEQ_MODEL_ID.replace("/", "--"))

        print("dirpath", dirpath)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with self.assertRaises(Exception):
            _ = ORTModelForSeq2SeqLM.from_pretrained(self.TINY_ONNX_SEQ2SEQ_MODEL_ID, local_files_only=True)

    @require_torch_gpu
    def test_load_model_cuda_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
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
        self.assertIsInstance(model.decoder, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.decoder_with_past, ORTSeq2SeqDecoder)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_without_past_from_hub(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=False)
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTSeq2SeqDecoder)
        self.assertTrue(model.decoder_with_past is None)
        self.assertIsInstance(model.config, PretrainedConfig)

    @require_torch_gpu
    def test_load_seq2seq_model_cuda_provider(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
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
            model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="ThisProviderDoesNotExist")

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
                with self.assertRaises(ImportError) as cm:
                    _ = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider=provider)

                self.assertTrue(
                    f"Asked to use {provider}, but `onnxruntime-gpu` package was not found." in str(cm.exception)
                )
        else:
            logger.info("Skipping CUDAExecutionProvider/TensorrtExecutionProvider without `onnxruntime-gpu` test")

        # need to install first onnxruntime-gpu, then onnxruntime for this test to pass,
        # thus overwritting onnxruntime/capi/_ld_preload.py
        if is_onnxruntime_installed and is_onnxruntime_gpu_installed:
            for provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
                with self.assertRaises(ImportError) as cm:
                    _ = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider=provider)

                self.assertTrue(
                    "`onnxruntime-gpu` is installed, but GPU dependencies are not loaded." in str(cm.exception)
                )
        else:
            logger.info("Skipping double onnxruntime + onnxruntime-gpu install test")

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

    @require_torch_gpu
    def test_model_on_gpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    # test string device input for to()
    @require_torch_gpu
    def test_model_on_gpu_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

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

    def test_seq2seq_model_on_cpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.encoder._device, cpu)
        self.assertEqual(model.decoder._device, cpu)
        self.assertEqual(model.decoder_with_past._device, cpu)
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
        self.assertEqual(model.encoder._device, cpu)
        self.assertEqual(model.decoder._device, cpu)
        self.assertEqual(model.decoder_with_past._device, cpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    @require_torch_gpu
    def test_seq2seq_model_on_gpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder._device, torch.device("cuda:0"))
        self.assertEqual(model.decoder._device, torch.device("cuda:0"))
        self.assertEqual(model.decoder_with_past._device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

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
    def test_seq2seq_model_on_gpu_str(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertEqual(model.encoder._device, torch.device("cuda:0"))
        self.assertEqual(model.decoder._device, torch.device("cuda:0"))
        self.assertEqual(model.decoder_with_past._device, torch.device("cuda:0"))
        self.assertEqual(model.encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_hf_token
    def test_load_model_from_hub_private(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None))
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
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

    def test_save_model_with_different_name(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_model_name = "model-test.onnx"
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)

            # save two models to simulate a optimization
            model.save_pretrained(tmpdirname)
            model.save_pretrained(tmpdirname, file_name=test_model_name)

            model = ORTModel.from_pretrained(tmpdirname, file_name=test_model_name)

            self.assertEqual(model.model_name, test_model_name)

    def test_save_load_ort_model_with_external_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], from_transformers=True)
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(ONNX_WEIGHTS_NAME + "_data" in folder_contents)

            # verify loading from local folder works
            model = ORTModelForSequenceClassification.from_pretrained(tmpdirname, from_transformers=False)
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    @parameterized.expand([(False,), (True,)])
    def test_save_load_decoder_model_with_external_data(self, use_cache: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForCausalLM.from_pretrained(
                MODEL_NAMES["gpt2"], use_cache=use_cache, from_transformers=True
            )
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME + "_data" in folder_contents)

            if use_cache:
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME in folder_contents)
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME + "_data" in folder_contents)

            # verify loading from local folder works
            model = ORTModelForCausalLM.from_pretrained(tmpdirname, use_cache=use_cache, from_transformers=False)
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    @parameterized.expand([(False,), (True,)])
    def test_save_load_seq2seq_model_with_external_data(self, use_cache: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["t5"], use_cache=use_cache, from_transformers=True
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
            model = ORTModelForSeq2SeqLM.from_pretrained(tmpdirname, use_cache=use_cache, from_transformers=False)
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

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

            model = ORTModelForSeq2SeqLM.from_pretrained(tmpdirname, use_cache=use_cache, from_transformers=True)
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
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id=self.HUB_REPOSITORY,
                private=True,
            )

    @require_hf_token
    def test_push_ort_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], from_transformers=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["bert"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTModelForSequenceClassification.from_pretrained(
                MODEL_NAMES["bert"] + "-onnx",
                from_transformers=False,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
            )
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    @require_hf_token
    def test_push_decoder_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForCausalLM.from_pretrained(MODEL_NAMES["gpt2"], from_transformers=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["gpt2"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTModelForCausalLM.from_pretrained(
                MODEL_NAMES["gpt2"] + "-onnx",
                from_transformers=False,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
            )
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    @require_hf_token
    def test_push_seq2seq_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAMES["mbart"], from_transformers=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["mbart"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["mbart"] + "-onnx",
                from_transformers=False,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
            )
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    def test_trust_remote_code(self):
        model_id = "fxmarty/tiny-testing-gpt2-remote-code"
        ort_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inputs = tokenizer("My name is", return_tensors="pt")

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
        "big_bird",
        "bigbird_pegasus",
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
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForQuestionAnswering
    TASK = "question-answering"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForQuestionAnswering.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("start_logits" in onnx_outputs)
        self.assertTrue("end_logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.start_logits, torch.Tensor)
        self.assertIsInstance(onnx_outputs.end_logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.start_logits, transformers_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(onnx_outputs.end_logits, transformers_outputs.end_logits, atol=1e-4))

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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch])
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
        "data2vec_text",
        "deberta",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "gpt2",
        "gpt_neo",
        "gptj",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in ORTModel
        # "layoutlm"
        # "layoutlmv3",
        "mbart",
        "mobilebert",
        # "perceiver",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm_roberta",
    ]

    ARCH_MODEL_MAP = {
        # TODO: fix non passing test
        # "perceiver": "hf-internal-testing/tiny-random-language_perceiver",
    }

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForSequenceClassification
    TASK = "sequence-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(self.onnx_model_dirs[model_arch])
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
            "typeform/distilbert-base-uncased-mnli", from_transformers=True
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
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

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
        "big_bird",
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
            _ = ORTModelForTokenClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(self.onnx_model_dirs[model_arch])
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
    SUPPORTED_ARCHITECTURES = ["albert", "bert", "camembert", "distilbert", "electra", "roberta", "xlm_roberta"]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForFeatureExtraction
    TASK = "default"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("last_hidden_state" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.last_hidden_state, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(
            torch.allclose(onnx_outputs.last_hidden_state, transformers_outputs.last_hidden_state, atol=1e-4)
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])
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
        "big_bird",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "mobilebert",
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

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
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

        inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
        onnx_outputs = onnx_model(**inputs)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Compare tensor outputs
        print(torch.max(torch.abs(onnx_outputs.logits - transformers_outputs.logits)))
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
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
        "gpt2",
        "gpt_neo",
        "gptj",
    ]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForCausalLM
    TASK = "causal-lm"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForCausalLM.from_pretrained(MODEL_NAMES["vit"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForCausalLM.from_pretrained(self.onnx_model_dirs[test_name])
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")

        # General case
        outputs = model.generate(**tokens)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)
        self.assertTrue(len(res[0]) > len(text))

        # With input ids
        tokens = tokenizer(text, return_tensors="pt")
        outputs = model.generate(input_ids=tokens["input_ids"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)
        self.assertTrue(len(res[0]) > len(text))

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.decoder, ORTDecoder)
        if onnx_model.use_cache is True:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoder)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_ort_model(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values_model_outputs(self, model_arch):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        text = "My Name is Philipp and i live"
        tokens = tokenizer(text, return_tensors="pt")
        model_with_pkv = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        outputs_model_with_pkv = model_with_pkv.generate(**tokens)
        model_without_pkv = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        outputs_model_without_pkv = model_without_pkv.generate(**tokens)
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    def test_compare_generation_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt").to("cuda")
        onnx_outputs = onnx_model.generate(**tokens)
        io_outputs = io_model.generate(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs, io_outputs))

        gc.collect()


class ORTModelForImageClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "beit",
        "convnext",
        "data2vec_vision",
        "deit",
        "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        # "perceiver",
        "poolformer",
        "resnet",
        "segformer",
        "swin",
        "vit",
    ]

    ARCH_MODEL_MAP = {
        # TODO: fix non passing test
        # "perceiver": "hf-internal-testing/tiny-random-vision_perceiver_conv",
    }

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForImageClassification
    TASK = "image-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForImageClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch] if model_arch in MODEL_NAMES else self.ARCH_MODEL_MAP[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        onnx_outputs = onnx_model(**inputs)

        self.assertTrue("logits" in onnx_outputs)
        self.assertTrue(isinstance(onnx_outputs.logits, torch.Tensor))

        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, trtfs_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(self.onnx_model_dirs[model_arch])
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
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()


class ORTModelForSemanticSegmentationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ("segformer",)

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    ORTMODEL_CLASS = ORTModelForSemanticSegmentation
    TASK = "semantic-segmentation"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSemanticSegmentation.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        onnx_outputs = onnx_model(**inputs)

        self.assertTrue("logits" in onnx_outputs)
        self.assertTrue(isinstance(onnx_outputs.logits, torch.Tensor))

        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, trtfs_outputs.logits, atol=1e-4))

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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(self.onnx_model_dirs[model_arch])
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

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt")
        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()


class ORTModelForSeq2SeqLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "bart",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot_small",
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
    }

    ORTMODEL_CLASS = ORTModelForSeq2SeqLM
    TASK = "seq2seq-lm"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAMES["bert"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForSeq2SeqLM.from_pretrained(self.onnx_model_dirs[test_name])
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

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        self.assertIsInstance(onnx_model.decoder, ORTSeq2SeqDecoder)
        if onnx_model.use_cache is True:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTSeq2SeqDecoder)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        onnx_outputs = onnx_model(**tokens, **decoder_inputs)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens, **decoder_inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_text_generation(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        tokenizer = get_preprocessor(model_id)

        # Text2Text generation
        pipe = pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        # Summarization
        pipe = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
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
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(self.onnx_model_dirs[test_name])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=False, device=0)
        text = "My Name is Philipp and i live"
        outputs = pipe(text, max_length=2 * len(text) + 1)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["translation_text"], str))

        pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=True, device=0)

        outputs = pipe(text, min_length=len(text) + 1, max_length=2 * len(text) + 1)
        self.assertTrue(isinstance(outputs[0]["translation_token_ids"], torch.Tensor))
        self.assertTrue(len(outputs[0]["translation_token_ids"]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values_model_outputs(self, model_arch: str):
        if model_arch == "m2m_100":
            return  # TODO: this test is failing for m2m_100
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        model_with_pkv = ORTModelForSeq2SeqLM.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        outputs_model_with_pkv = model_with_pkv.generate(**tokens)
        model_without_pkv = ORTModelForSeq2SeqLM.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        outputs_model_without_pkv = model_without_pkv.generate(**tokens)
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSeq2SeqLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")
        decoder_start_token_id = onnx_model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((2, 1), dtype=torch.long) * decoder_start_token_id}

        onnx_outputs = onnx_model(**tokens, **decoder_inputs)
        io_outputs = io_model(**tokens, **decoder_inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.logits, io_outputs.logits))

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    def test_compare_generation_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSeq2SeqLM.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt").to("cuda")
        onnx_outputs = onnx_model.generate(**tokens, num_beams=5)
        io_outputs = io_model.generate(**tokens, num_beams=5)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs, io_outputs))

        gc.collect()


class ORTModelForSpeechSeq2SeqIntegrationTest(ORTModelTestMixin):
    # TODO: speech_to_text should be tested
    SUPPORTED_ARCHITECTURES = ["whisper"]

    FULL_GRID = {
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForSpeechSeq2Seq
    TASK = "speech2seq-lm"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_NAMES["bert"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

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
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        self.assertIsInstance(onnx_model.decoder, ORTSeq2SeqDecoder)
        if onnx_model.use_cache is True:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTSeq2SeqDecoder)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        decoder_start_token_id = transformers_model.config.decoder_start_token_id
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        onnx_outputs = onnx_model(**features, **decoder_inputs)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**features, **decoder_inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_speech_recognition(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
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

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values_model_outputs(self, model_arch: str):
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
        outputs_model_with_pkv = model_with_pkv.generate(**features)
        model_without_pkv = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        outputs_model_without_pkv = model_without_pkv.generate(**features)

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

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

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    def test_compare_generation_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model.generate(**features, num_beams=5)
        io_outputs = io_model.generate(**features, num_beams=5)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs, io_outputs))

        gc.collect()


class ORTModelForCustomTasksIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "sbert": "optimum/sbert-all-MiniLM-L6-with-pooler",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        outputs = model(**tokens)
        self.assertIsInstance(outputs.pooler_output, torch.Tensor)

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
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pipe.device, onnx_model.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_compare_to_io_binding(self, *args, **kwargs):
        model_arch, model_id = args
        set_seed(SEED)
        onnx_model = ORTModelForCustomTasks.from_pretrained(
            model_id, use_io_binding=False, provider="CUDAExecutionProvider"
        )
        set_seed(SEED)
        io_model = ORTModelForCustomTasks.from_pretrained(
            model_id, use_io_binding=True, provider="CUDAExecutionProvider"
        )
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("pooler_output" in io_outputs)
        self.assertIsInstance(io_outputs.pooler_output, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(onnx_outputs.pooler_output, io_outputs.pooler_output))

        gc.collect()


class TestBothExportersORTModel(unittest.TestCase):
    @parameterized.expand(
        [
            ["question-answering", ORTModelForQuestionAnsweringIntegrationTest],
            ["sequence-classification", ORTModelForSequenceClassificationIntegrationTest],
            ["token-classification", ORTModelForTokenClassificationIntegrationTest],
            ["default", ORTModelForFeatureExtractionIntegrationTest],
            ["multiple-choice", ORTModelForMultipleChoiceIntegrationTest],
            ["causal-lm", ORTModelForCausalLMIntegrationTest],
            ["image-classification", ORTModelForImageClassificationIntegrationTest],
            ["semantic-segmentation", ORTModelForSemanticSegmentationIntegrationTest],
            ["seq2seq-lm", ORTModelForSeq2SeqLMIntegrationTest],
            ["speech2seq-lm", ORTModelForSpeechSeq2SeqIntegrationTest],
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
