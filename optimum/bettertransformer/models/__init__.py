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
import warnings

from .encoder_models import (
    AlbertLayerBetterTransformer,
    BartEncoderLayerBetterTransformer,
    BertLayerBetterTransformer,
    CLIPLayerBetterTransformer,
    DistilBertLayerBetterTransformer,
    FSMTEncoderLayerBetterTransformer,
    MBartEncoderLayerBetterTransformer,
    ViltLayerBetterTransformer,
    ViTLayerBetterTransformer,
    Wav2Vec2EncoderLayerBetterTransformer,
    WhisperEncoderLayerBetterTransformer,
)


class BetterTransformerManager:
    BETTER_TRANSFORMER_MODEL_MAPPING = {
        "bert": ("BertLayer", BertLayerBetterTransformer),
        "bert-generation": ("BertGenerationLayer", BertLayerBetterTransformer),
        "bart": ("BertLayer", BertLayerBetterTransformer),
        "camembert": ("CamembertLayer", BertLayerBetterTransformer),
        "data2vec-text": ("Data2VecTextLayer", BertLayerBetterTransformer),
        "electra": ("ElectraLayer", BertLayerBetterTransformer),
        "ernie": ("ErnieLayer", BertLayerBetterTransformer),
        "layoutlm": ("LayoutLMLayer", BertLayerBetterTransformer),
        "markuplm": ("MarkupLMLayer", BertLayerBetterTransformer),
        "rembert": ("RemBertLayer", BertLayerBetterTransformer),
        "roberta": ("RobertaLayer", BertLayerBetterTransformer),
        "splinter": ("SplinterLayer", BertLayerBetterTransformer),
        "tapas": ("TapasLayer", BertLayerBetterTransformer),
        "xlm-roberta": ("XLMRobertaLayer", BertLayerBetterTransformer),
        "albert": ("AlbertLayer", AlbertLayerBetterTransformer),
        "bart": ("BartEncoderLayer", BartEncoderLayerBetterTransformer),
        "mbart": ("MBartEncoderLayer", MBartEncoderLayerBetterTransformer),
        "m2m_100": ("M2M100EncoderLayer", MBartEncoderLayerBetterTransformer),
        "distilbert": ("TransformerBlock", DistilBertLayerBetterTransformer),
        "whisper": ("WhisperEncoderLayer", WhisperEncoderLayerBetterTransformer),
        "hubert": ("HubertEncoderLayer", Wav2Vec2EncoderLayerBetterTransformer),
        "wav2vec2": ("Wav2Vec2EncoderLayer", Wav2Vec2EncoderLayerBetterTransformer),
        "deit": ("DeiTLayer", ViTLayerBetterTransformer),
        "vit": ("ViTLayer", ViTLayerBetterTransformer),
        "vit_mae": ("ViTMAELayer", ViTLayerBetterTransformer),
        "vit_msn": ("ViTMSNLayer", ViTLayerBetterTransformer),
        "yolos": ("YolosLayer", ViTLayerBetterTransformer),
        "fsmt_decoder": ("EncoderLayer", FSMTEncoderLayerBetterTransformer),
        "vilt": ("ViltLayer", ViltLayerBetterTransformer),
        "clip": ("CLIPEncoderLayer", CLIPLayerBetterTransformer),
    }

    EXCLUDE_FROM_TRANSFORM = {
        "clip": [
            "text_model"
        ],  # text model uses causal attention, that is most likely not supported in BetterTransformer
    }

    CAN_NOT_BE_SUPPORTED = {
        "t5": "message here",
    }


class warn_uncompatible_save(object):
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "You are calling `save_pretrained` to a `BetterTransformer` converted model you may likely encounter unexepected behaviors. ",
            UserWarning,
        )
        return self.callback(*args, **kwargs)
