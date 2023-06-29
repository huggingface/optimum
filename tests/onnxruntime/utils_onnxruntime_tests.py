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

import shutil
import tempfile
import unittest
from typing import Dict

import numpy as np
import torch
from transformers import set_seed

from optimum.exporters import TasksManager


MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "audio_spectrogram_transformer": "Ericwang/tiny-random-ast",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "bart": "hf-internal-testing/tiny-random-bart",
    # "big_bird": "hf-internal-testing/tiny-random-BigBirdModel",
    # "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
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
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "groupvit": "hf-internal-testing/tiny-random-groupvit",
    "ibert": "hf-internal-testing/tiny-random-IBertModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "layoutlm": "hf-internal-testing/tiny-random-LayoutLMModel",
    "layoutlmv3": "hf-internal-testing/tiny-random-LayoutLMv3Model",
    "longt5": "hf-internal-testing/tiny-random-LongT5Model",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "marian": "sshleifer/tiny-marian-en-de",  # hf-internal-testing ones are broken
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mt5": "lewtun/tiny-random-mt5",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "pegasus": "hf-internal-testing/tiny-random-PegasusModel",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "squeezebert": "hf-internal-testing/tiny-random-SqueezeBertModel",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "t5": "hf-internal-testing/tiny-random-t5",
    "vit": "hf-internal-testing/tiny-random-vit",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
    "whisper": "openai/whisper-tiny.en",  # hf-internal-testing ones are broken
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "wavlm": "hf-internal-testing/tiny-random-WavlmModel",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew_d": "hf-internal-testing/tiny-random-SEWDModel",
    "speech_to_text": "hf-internal-testing/tiny-random-Speech2TextModel",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech_sat": "hf-internal-testing/tiny-random-UnispeechSatModel",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm_roberta": "hf-internal-testing/tiny-xlm-roberta",
    "vision-encoder-decoder": "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
    "trocr": "microsoft/trocr-small-handwritten",
}

SEED = 42


class ORTModelTestMixin(unittest.TestCase):
    ARCH_MODEL_MAP = {}

    TENSOR_ALIAS_TO_TYPE = {
        "pt": torch.Tensor,
        "np": np.ndarray,
    }

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

        # TODO: this should actually be checked in ORTModel!
        task = self.TASK
        if "use_cache" in model_args and model_args["use_cache"] is True:
            task = task + "-with-past"

        if "use_cache" in model_args and task not in TasksManager.get_supported_tasks_for_model_type(
            model_arch.replace("_", "-"), exporter="onnx"
        ):
            self.skipTest("Unsupported export case")

        if model_arch_and_params not in self.onnx_model_dirs:
            # model_args will contain kwargs to pass to ORTModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")

            model_id = (
                self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
            )
            set_seed(SEED)
            onnx_model = self.ORTMODEL_CLASS.from_pretrained(model_id, **model_args, use_io_binding=False, export=True)

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            onnx_model.save_pretrained(model_dir)
            self.onnx_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.onnx_model_dirs.items():
            shutil.rmtree(dir_path)
