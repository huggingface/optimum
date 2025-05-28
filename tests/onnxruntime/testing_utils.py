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


SEED = 42

MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "audio-spectrogram-transformer": "Ericwang/tiny-random-ast",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "bart": "hf-internal-testing/tiny-random-bart",
    "big_bird": "hf-internal-testing/tiny-random-BigBirdModel",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "clip": "hf-internal-testing/tiny-random-CLIPModel",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "convnextv2": "hf-internal-testing/tiny-random-ConvNextV2Model",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "data2vec-text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec-vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec-audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "donut": "fxmarty/tiny-doc-qa-vision-encoder-decoder",
    "detr": "hf-internal-testing/tiny-random-detr",
    "dinov2": "hf-internal-testing/tiny-random-Dinov2Model",
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "dpt": "hf-internal-testing/tiny-random-DPTForSemanticSegmentation",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "encoder-decoder": {
        "hf-internal-testing/tiny-random-EncoderDecoderModel-bert-bert": ["text2text-generation"],
        "mohitsha/tiny-random-testing-bert2gpt2": ["text2text-generation", "text2text-generation-with-past"],
    },
    "efficientnet": "hf-internal-testing/tiny-random-EfficientNetForImageClassification",
    "falcon": "fxmarty/really-tiny-falcon-testing",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "flux": "optimum-internal-testing/tiny-random-flux",
    "gemma": "fxmarty/tiny-random-GemmaForCausalLM",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "groupvit": "hf-internal-testing/tiny-random-groupvit",
    "hiera": "hf-internal-testing/tiny-random-HieraForImageClassification",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "ibert": "hf-internal-testing/tiny-random-IBertModel",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
    "layoutlm": "hf-internal-testing/tiny-random-LayoutLMModel",
    "layoutlmv3": "hf-internal-testing/tiny-random-LayoutLMv3Model",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "longt5": "hf-internal-testing/tiny-random-LongT5Model",
    "llama": "optimum-internal-testing/tiny-random-llama",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "marian": "echarlaix/tiny-random-marian",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mgp-str": "hf-internal-testing/tiny-random-MgpstrForSceneTextRecognition",
    "mistral": "echarlaix/tiny-random-mistral",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "lewtun/tiny-random-mt5",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "pegasus": "hf-internal-testing/tiny-random-PegasusModel",
    "perceiver_text": "hf-internal-testing/tiny-random-language_perceiver",
    "perceiver_vision": "hf-internal-testing/tiny-random-vision_perceiver_conv",
    "phi3": "Xenova/tiny-random-Phi3ForCausalLM",
    "pix2struct": "fxmarty/pix2struct-tiny-random",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "pvt": "hf-internal-testing/tiny-random-PvtForImageClassification",
    "qwen2": "fxmarty/tiny-dummy-qwen2",
    "rembert": "hf-internal-testing/tiny-random-RemBertModel",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew-d": "asapp/sew-d-tiny-100k-ft-ls100h",
    "siglip": "hf-internal-testing/tiny-random-SiglipModel",
    "squeezebert": "hf-internal-testing/tiny-random-SqueezeBertModel",
    "speech_to_text": "optimum-internal-testing/tiny-random-Speech2TextModel",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-3": "optimum-internal-testing/tiny-random-stable-diffusion-3",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "swinv2": "hf-internal-testing/tiny-random-Swinv2Model",
    "swin-window": "yujiepan/tiny-random-swin-patch4-window7-224",
    "swin2sr": "hf-internal-testing/tiny-random-Swin2SRForImageSuperResolution",
    "t5": "hf-internal-testing/tiny-random-t5",
    "table-transformer": "hf-internal-testing/tiny-random-TableTransformerModel",
    "trocr": "microsoft/trocr-small-handwritten",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech-sat": "hf-internal-testing/tiny-random-UnispeechSatModel",
    "vision-encoder-decoder": "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
    "vit": "hf-internal-testing/tiny-random-vit",
    "whisper": "optimum-internal-testing/tiny-random-whisper",
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "wavlm": "hf-internal-testing/tiny-random-WavlmModel",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm-qa": "hf-internal-testing/tiny-random-XLMForQuestionAnsweringSimple",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
}


class ORTModelTestMixin(unittest.TestCase):
    TENSOR_ALIAS_TO_TYPE = {
        "pt": torch.Tensor,
        "np": np.ndarray,
    }

    ATOL = 1e-4
    RTOL = 1e-4

    TASK = None

    ORTMODEL_CLASS = None
    AUTOMODEL_CLASS = None

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

        model_ids = MODEL_NAMES[model_arch]
        if isinstance(model_ids, dict):
            model_ids = list(model_ids.keys())
        else:
            model_ids = [model_ids]

        # TODO: this should actually be checked in ORTModel!
        task = self.TASK
        if "use_cache" in model_args and model_args["use_cache"] is True:
            task = task + "-with-past"

        library_name = TasksManager.infer_library_from_model(model_ids[0])

        if "use_cache" in model_args and task not in TasksManager.get_supported_tasks_for_model_type(
            model_arch, exporter="onnx", library_name=library_name
        ):
            self.skipTest("Unsupported export case")

        if model_arch_and_params not in self.onnx_model_dirs:
            self.onnx_model_dirs[model_arch_and_params] = {}

            # model_args will contain kwargs to pass to ORTModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")

            for idx, model_id in enumerate(model_ids):
                if model_arch == "encoder-decoder" and task not in MODEL_NAMES[model_arch][model_id]:
                    # The model with use_cache=True is not supported for bert as a decoder")
                    continue

                set_seed(SEED)
                onnx_model = self.ORTMODEL_CLASS.from_pretrained(
                    model_id, **model_args, use_io_binding=False, export=True
                )

                model_dir = tempfile.mkdtemp(
                    prefix=f"{model_arch_and_params}_{self.TASK}_{model_id.replace('/', '_')}"
                )
                onnx_model.save_pretrained(model_dir)
                if isinstance(MODEL_NAMES[model_arch], dict):
                    self.onnx_model_dirs[model_arch_and_params][model_id] = model_dir
                else:
                    self.onnx_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.onnx_model_dirs.items():
            if isinstance(dir_path, dict):
                for _, sec_dir_path in dir_path.items():
                    shutil.rmtree(sec_dir_path)
            else:
                shutil.rmtree(dir_path)
