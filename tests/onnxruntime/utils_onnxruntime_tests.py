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
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "convnextv2": "hf-internal-testing/tiny-random-ConvNextV2Model",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "data2vec_text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec_vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec_audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",
    "deberta_v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "default-timm-config": {
        "timm/inception_v3.tf_adv_in1k": ["image-classification"],
        "timm/tf_efficientnet_b0.in1k": ["image-classification"],
        "timm/resnetv2_50x1_bit.goog_distilled_in1k": ["image-classification"],
        "timm/cspdarknet53.ra_in1k": ["image-classification"],
        "timm/cspresnet50.ra_in1k": ["image-classification"],
        "timm/cspresnext50.ra_in1k": ["image-classification"],
        "timm/densenet121.ra_in1k": ["image-classification"],
        "timm/dla102.in1k": ["image-classification"],
        "timm/dpn107.mx_in1k": ["image-classification"],
        "timm/ecaresnet101d.miil_in1k": ["image-classification"],
        "timm/efficientnet_b1_pruned.in1k": ["image-classification"],
        "timm/inception_resnet_v2.tf_ens_adv_in1k": ["image-classification"],
        "timm/fbnetc_100.rmsp_in1k": ["image-classification"],
        "timm/xception41.tf_in1k": ["image-classification"],
        "timm/senet154.gluon_in1k": ["image-classification"],
        "timm/seresnext26d_32x4d.bt_in1k": ["image-classification"],
        "timm/hrnet_w18.ms_aug_in1k": ["image-classification"],
        "timm/inception_v3.gluon_in1k": ["image-classification"],
        "timm/inception_v4.tf_in1k": ["image-classification"],
        "timm/mixnet_s.ft_in1k": ["image-classification"],
        "timm/mnasnet_100.rmsp_in1k": ["image-classification"],
        "timm/mobilenetv2_100.ra_in1k": ["image-classification"],
        "timm/mobilenetv3_small_050.lamb_in1k": ["image-classification"],
        "timm/nasnetalarge.tf_in1k": ["image-classification"],
        "timm/tf_efficientnet_b0.ns_jft_in1k": ["image-classification"],
        "timm/pnasnet5large.tf_in1k": ["image-classification"],
        "timm/regnetx_002.pycls_in1k": ["image-classification"],
        "timm/regnety_002.pycls_in1k": ["image-classification"],
        "timm/res2net101_26w_4s.in1k": ["image-classification"],
        "timm/res2next50.in1k": ["image-classification"],
        "timm/resnest101e.in1k": ["image-classification"],
        "timm/spnasnet_100.rmsp_in1k": ["image-classification"],
        "timm/resnet18.fb_swsl_ig1b_ft_in1k": ["image-classification"],
        "timm/wide_resnet101_2.tv_in1k": ["image-classification"],
        "timm/tresnet_l.miil_in1k": ["image-classification"],
    },
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "donut": "fxmarty/tiny-doc-qa-vision-encoder-decoder",
    "detr": "hf-internal-testing/tiny-random-detr",
    "dpt": "hf-internal-testing/tiny-random-DPTModel",
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "encoder-decoder": {
        "hf-internal-testing/tiny-random-EncoderDecoderModel-bert-bert": [
            "text2text-generation",
        ],
        "mohitsha/tiny-random-testing-bert2gpt2": ["text2text-generation", "text2text-generation-with-past"],
    },
    "falcon": "fxmarty/really-tiny-falcon-testing",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gemma": "fxmarty/tiny-random-GemmaForCausalLM",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "groupvit": "hf-internal-testing/tiny-random-groupvit",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "ibert": "hf-internal-testing/tiny-random-IBertModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
    "layoutlm": "hf-internal-testing/tiny-random-LayoutLMModel",
    "layoutlmv3": "hf-internal-testing/tiny-random-LayoutLMv3Model",
    "longt5": "hf-internal-testing/tiny-random-LongT5Model",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "marian": "sshleifer/tiny-marian-en-de",  # hf-internal-testing ones are broken
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mistral": "echarlaix/tiny-random-mistral",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "lewtun/tiny-random-mt5",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "pegasus": "hf-internal-testing/tiny-random-PegasusModel",
    "perceiver_text": "hf-internal-testing/tiny-random-language_perceiver",
    "perceiver_vision": "hf-internal-testing/tiny-random-vision_perceiver_conv",
    "phi3": "Xenova/tiny-random-Phi3ForCausalLM",
    "pix2struct": "fxmarty/pix2struct-tiny-random",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "qwen2": "fxmarty/tiny-dummy-qwen2",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew_d": "asapp/sew-d-tiny-100k-ft-ls100h",
    "squeezebert": "hf-internal-testing/tiny-random-SqueezeBertModel",
    "speech_to_text": "hf-internal-testing/tiny-random-Speech2TextModel",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "swin-window": "yujiepan/tiny-random-swin-patch4-window7-224",
    "t5": "hf-internal-testing/tiny-random-t5",
    "table-transformer": "hf-internal-testing/tiny-random-TableTransformerModel",
    "trocr": "microsoft/trocr-small-handwritten",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech_sat": "hf-internal-testing/tiny-random-UnispeechSatModel",
    "vision-encoder-decoder": "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
    "vit": "hf-internal-testing/tiny-random-vit",
    "whisper": "openai/whisper-tiny.en",  # hf-internal-testing ones are broken
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "wavlm": "hf-internal-testing/tiny-random-WavlmModel",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm_qa": "hf-internal-testing/tiny-random-XLMForQuestionAnsweringSimple",  # issue with default hf-internal-testing in transformers QA pipeline post-processing
    "xlm_roberta": "hf-internal-testing/tiny-xlm-roberta",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
}

SEED = 42


class ORTModelTestMixin(unittest.TestCase):
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
            model_arch.replace("_", "-"), exporter="onnx", library_name=library_name
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
