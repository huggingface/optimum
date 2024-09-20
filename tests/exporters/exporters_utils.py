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

VALIDATE_EXPORT_ON_SHAPES_SLOW = {
    "batch_size": [1, 3, 5],
    "sequence_length": [8, 33, 96, 154],
    "num_choices": [2, 4],
    "audio_sequence_length": [1000, 2000],
    "point_batch_size": [1, 5],
    "nb_points_per_image": [1, 3],
}

VALIDATE_EXPORT_ON_SHAPES_FAST = {
    "batch_size": [4],
    "sequence_length": [19],
    "num_choices": [4],
}

NO_DYNAMIC_AXES_EXPORT_SHAPES_TRANSFORMERS = {
    "batch_size": [1, 3, 5],
    "num_choices": [2, 4],
    "sequence_length": [8, 33, 96],
}

PYTORCH_EXPORT_MODELS_TINY = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": {
        "hf-internal-testing/tiny-random-BertModel": [
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
        ],
        "nreimers/BERT-Tiny_L-2_H-128_A-2": ["feature-extraction"],
    },
    "bart": "hf-internal-testing/tiny-random-bart",
    # "big-bird": "hf-internal-testing/tiny-random-BigBirdModel",
    # "bigbird-pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "clip": "hf-internal-testing/tiny-random-CLIPModel",
    "clip-vision-model": "fxmarty/clip-vision-model-tiny",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "convnextv2": "hf-internal-testing/tiny-random-ConvNextV2Model",
    "codegen": "hf-internal-testing/tiny-random-CodeGenModel",
    "cvt": "hf-internal-testing/tiny-random-CvTModel",
    "data2vec-text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec-vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec-audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "donut": "fxmarty/tiny-doc-qa-vision-encoder-decoder",
    "donut-swin": "hf-internal-testing/tiny-random-DonutSwinModel",
    "detr": "hf-internal-testing/tiny-random-DetrModel",  # hf-internal-testing/tiny-random-detr is larger
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "dpt": "hf-internal-testing/tiny-random-DPTModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "encoder-decoder": {
        "hf-internal-testing/tiny-random-EncoderDecoderModel-bert-bert": [
            "text2text-generation",
        ],
        "mohitsha/tiny-random-testing-bert2gpt2": ["text2text-generation", "text2text-generation-with-past"],
    },
    "esm": "hf-internal-testing/tiny-random-EsmModel",
    "falcon": {
        "fxmarty/really-tiny-falcon-testing": [
            "feature-extraction",
            "feature-extraction-with-past",
            "question-answering",
            "text-generation",
            "text-generation-with-past",
            "token-classification",
        ],
        "fxmarty/tiny-testing-falcon-alibi": ["text-generation", "text-generation-with-past"],
    },
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gemma": "fxmarty/tiny-random-GemmaForCausalLM",
    "glpn": "hf-internal-testing/tiny-random-GLPNModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt-bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt-neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt-neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "groupvit": "hf-internal-testing/tiny-random-groupvit",
    "ibert": "hf-internal-testing/tiny-random-IBertModel",
    "imagegpt": "hf-internal-testing/tiny-random-ImageGPTModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "layoutlm": "hf-internal-testing/tiny-random-LayoutLMModel",
    "layoutlmv3": "hf-internal-testing/tiny-random-LayoutLMv3Model",
    "lilt": "hf-internal-testing/tiny-random-LiltModel",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "longt5": "fxmarty/tiny-random-working-LongT5Model",
    # "longformer": "allenai/longformer-base-4096",
    "m2m-100": "hf-internal-testing/tiny-random-m2m_100",
    "marian": "sshleifer/tiny-marian-en-de",  # hf-internal-testing ones are broken
    "markuplm": "hf-internal-testing/tiny-random-MarkupLMModel",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mistral": "echarlaix/tiny-random-mistral",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet-v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilenet-v1": "google/mobilenet_v1_0.75_192",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "lewtun/tiny-random-mt5",
    "musicgen": "hf-internal-testing/tiny-random-MusicgenForConditionalGeneration",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "owlv2": "hf-internal-testing/tiny-random-Owlv2Model",
    "owlvit": "hf-tiny-model-private/tiny-random-OwlViTModel",
    "pegasus": "hf-internal-testing/tiny-random-PegasusModel",
    "perceiver": {
        "hf-internal-testing/tiny-random-language_perceiver": ["fill-mask", "text-classification"],
        "hf-internal-testing/tiny-random-vision_perceiver_conv": ["image-classification"],
    },
    "phi": "echarlaix/tiny-random-PhiForCausalLM",
    "phi3": "Xenova/tiny-random-Phi3ForCausalLM",
    "pix2struct": "fxmarty/pix2struct-tiny-random",
    # "rembert": "google/rembert",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "qwen2": "fxmarty/tiny-dummy-qwen2",
    "regnet": "hf-internal-testing/tiny-random-RegNetModel",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "sam": "fxmarty/sam-vit-tiny-random",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "splinter": "hf-internal-testing/tiny-random-SplinterModel",
    "squeezebert": "hf-internal-testing/tiny-random-SqueezeBertModel",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "swin2sr": "hf-internal-testing/tiny-random-Swin2SRModel",
    "t5": "hf-internal-testing/tiny-random-t5",
    "table-transformer": "hf-internal-testing/tiny-random-TableTransformerModel",
    "vit": "hf-internal-testing/tiny-random-vit",
    "vits": "echarlaix/tiny-random-vits",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
    "whisper": "openai/whisper-tiny.en",  # hf-internal-testing ones are broken
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "wavlm": {
        "hf-internal-testing/tiny-random-wavlm": [
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
        ],
        "hf-internal-testing/tiny-random-WavLMForCTC": ["audio-frame-classification"],
        "hf-internal-testing/tiny-random-WavLMForXVector": ["audio-xvector"],
    },
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew-d": "hf-internal-testing/tiny-random-SEWDModel",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech-sat": {
        "hf-internal-testing/tiny-random-unispeech-sat": [
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
        ],
        "hf-internal-testing/tiny-random-UniSpeechSatForPreTraining": ["audio-frame-classification"],
        "hf-internal-testing/tiny-random-UniSpeechSatForXVector": ["audio-xvector"],
    },
    "audio-spectrogram-transformer": "Ericwang/tiny-random-ast",
    # Disabled for now because some operator seems to not be supported by ONNX.
    # "mctct": "hf-internal-testing/tiny-random-MCTCTModel",
    "speech-to-text": "hf-internal-testing/tiny-random-Speech2TextModel",
    "speecht5": "hf-internal-testing/tiny-random-SpeechT5ForTextToSpeech",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
    "vision-encoder-decoder": {
        "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2": [
            "image-to-text",
            "image-to-text-with-past",
        ],
        "microsoft/trocr-small-handwritten": ["image-to-text", "image-to-text-with-past"],
        "fxmarty/tiny-doc-qa-vision-encoder-decoder": [
            "document-question-answering",
            "document-question-answering-with-past",
        ],
    },
}


PYTORCH_EXPORT_MODELS_LARGE = {
    "albert": "albert-base-v2",
    "beit": "microsoft/beit-base-patch16-224",
    "bert": "bert-base-cased",
    "bart": "facebook/bart-base",
    # "big-bird": "google/bigbird-roberta-base",
    # "bigbird-pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot-small": "facebook/blenderbot_small-90M",
    "blenderbot": "facebook/blenderbot-90M",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",  # Not using bigscience/bloom-560m because it goes OOM.
    "camembert": "camembert-base",
    "clip": "openai/clip-vit-base-patch32",
    "convbert": "YituTech/conv-bert-base",
    "convnext": "facebook/convnext-tiny-224",
    "codegen": "hf-internal-testing/tiny-random-CodeGenModel",  # Not using Salesforce/codegen-350M-multi because it takes too much time for testing.
    "data2vec-text": "facebook/data2vec-text-base",
    "data2vec-vision": "facebook/data2vec-vision-base",
    "data2vec-audio": "facebook/data2vec-audio-base",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",  # Not using microsoft/deberta-base because it takes too much time for testing.
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",  # Not using microsoft/deberta-v2-xlarge because it takes too much time for testing.
    "deit": "facebook/deit-small-patch16-224",
    "detr": "hf-internal-testing/tiny-random-detr",  # Not using facebook/detr-resnet-50 because it takes too much time for testing.
    "distilbert": "distilbert-base-cased",
    "electra": "google/electra-base-generator",
    "encoder-decoder": "patrickvonplaten/bert2bert_cnn_daily_mail",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",  # TODO
    "gemma": "google/gemma-2b",
    "gpt2": "gpt2",
    "gpt-neo": "EleutherAI/gpt-neo-125M",
    "gpt-neox": "EleutherAI/gpt-neox-20b",
    "gptj": "anton-l/gpt-j-tiny-random",  # TODO
    "groupvit": "nvidia/groupvit-gcc-yfcc",
    "ibert": "kssteven/ibert-roberta-base",
    "imagegpt": "openai/imagegpt-small",
    "levit": "facebook/levit-128S",
    "layoutlm": "microsoft/layoutlm-base-uncased",
    "layoutlmv3": "microsoft/layoutlmv3-base",
    "lilt": "SCUT-DLVCLab/lilt-roberta-en-base",
    "llama": "decapoda-research/llama-65b-hf",
    "longt5": "fxmarty/tiny-random-working-LongT5Model",  # Not using google/long-t5-local-base because it takes too much time for testing.
    # "longformer": "allenai/longformer-base-4096",
    "m2m-100": "hf-internal-testing/tiny-random-m2m_100",  # Not using facebook/m2m100_418M because it takes too much time for testing.
    "marian": "Helsinki-NLP/opus-mt-en-de",
    "markuplm": "hf-internal-testing/tiny-random-MarkupLMModel",
    "mbart": "sshleifer/tiny-mbart",
    "mobilebert": "google/mobilebert-uncased",
    # "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    # "mobilenet_v2": "google/mobilenet_v2_0.35_96",
    "mobilevit": "apple/mobilevit-small",
    "mpt": "mosaicml/mpt-7b",
    "mt5": "lewtun/tiny-random-mt5",  # Not using google/mt5-small because it takes too much time for testing.
    "musicgen": "facebook/musicgen-small",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "owlv2": "google/owlv2-base-patch16",
    "owlvit": "google/owlvit-base-patch32",
    "perceiver": "hf-internal-testing/tiny-random-PerceiverModel",  # Not using deepmind/language-perceiver because it takes too much time for testing.
    # "rembert": "google/rembert",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "regnet": "facebook/regnet-y-040",
    "resnet": "microsoft/resnet-50",
    "roberta": "roberta-base",
    "roformer": "junnyu/roformer_chinese_base",
    "sam": "facebook/sam-vit-base",
    "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",
    "splinter": "hf-internal-testing/tiny-random-SplinterModel",
    "squeezebert": "squeezebert/squeezebert-uncased",
    "swin": "microsoft/swin-tiny-patch4-window7-224",
    "t5": "t5-small",
    "table-transformer": "microsoft/table-transformer-detection",
    "vit": "google/vit-base-patch16-224",
    "yolos": "hustvl/yolos-tiny",
    "whisper": "openai/whisper-tiny.en",
    "hubert": "facebook/hubert-base-ls960",
    "wav2vec2": "facebook/wav2vec2-base-960h",
    "wav2vec2-conformer": "facebook/wav2vec2-conformer-rel-pos-large-960h-ft",
    "wavlm": "microsoft/wavlm-base-plus-sv",
    "sew": "asapp/sew-tiny-100k",
    "sew-d": "asapp/sew-d-tiny-100k-ft-ls100h",
    "unispeech": "microsoft/unispeech-1350-en-353-fr-ft-1h",
    "unispeech-sat": "microsoft/unispeech-sat-base",
    "audio-spectrogram-transformer": "nielsr/audio-spectogram-transformer-finetuned-audioset-10-10-0.4593",
    # Disabled for now because some operator seems to not be supported by ONNX.
    # "mctct": "speechbrain/m-ctc-t-large",
    "speech-to-text": "codenamewei/speech-to-text",
    "xlm": "xlm-clm-ende-1024",
    "xlm-roberta": "Unbabel/xlm-roberta-comet-small",
}

TENSORFLOW_EXPORT_MODELS = {
    "albert": "hf-internal-testing/tiny-albert",
    "bert": "bert-base-cased",
    "camembert": "camembert-base",
    "distilbert": "distilbert-base-cased",
    "roberta": "roberta-base",
}

PYTORCH_DIFFUSION_MODEL = {
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
}

PYTORCH_TIMM_MODEL = {
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
    }
}

PYTORCH_SENTENCE_TRANSFORMERS_MODEL = {
    "clip": "sentence-transformers/clip-ViT-B-32",
    "transformer": {
        "sentence-transformers/all-MiniLM-L6-v2": ["feature-extraction", "sentence-similarity"],
        "fxmarty/tiny-dummy-mistral-sentence-transformer": ["feature-extraction", "sentence-similarity"],
    },
}


PYTORCH_TRANSFORMERS_MODEL_NO_DYNAMIC_AXES = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
}


PYTORCH_TIMM_MODEL_NO_DYNAMIC_AXES = {
    "default-timm-config": {
        "timm/ese_vovnet39b.ra_in1k": ["image-classification"],
        "timm/ese_vovnet19b_dw.ra_in1k": ["image-classification"],
    }
}
