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
    MODEL_MAPPING = {
        "albert": ("AlbertLayer", AlbertLayerBetterTransformer),
        "bart": ("BartEncoderLayer", BartEncoderLayerBetterTransformer),
        "bert": ("BertLayer", BertLayerBetterTransformer),
        "bert-generation": ("BertGenerationLayer", BertLayerBetterTransformer),
        "camembert": ("CamembertLayer", BertLayerBetterTransformer),
        "clip": ("CLIPEncoderLayer", CLIPLayerBetterTransformer),
        "data2vec-text": ("Data2VecTextLayer", BertLayerBetterTransformer),
        "deit": ("DeiTLayer", ViTLayerBetterTransformer),
        "distilbert": ("TransformerBlock", DistilBertLayerBetterTransformer),
        "electra": ("ElectraLayer", BertLayerBetterTransformer),
        "ernie": ("ErnieLayer", BertLayerBetterTransformer),
        "fsmt": ("EncoderLayer", FSMTEncoderLayerBetterTransformer),
        "hubert": ("HubertEncoderLayer", Wav2Vec2EncoderLayerBetterTransformer),
        "layoutlm": ("LayoutLMLayer", BertLayerBetterTransformer),
        "m2m_100": ("M2M100EncoderLayer", MBartEncoderLayerBetterTransformer),
        "marian": ("MarianEncoderLayer", BartEncoderLayerBetterTransformer),
        "markuplm": ("MarkupLMLayer", BertLayerBetterTransformer),
        "mbart": ("MBartEncoderLayer", MBartEncoderLayerBetterTransformer),
        "rembert": ("RemBertLayer", BertLayerBetterTransformer),
        "roberta": ("RobertaLayer", BertLayerBetterTransformer),
        "roc_bert": ("RoCBertLayer", BertLayerBetterTransformer),
        "roformer": ("RoFormerLayer", BertLayerBetterTransformer),
        "splinter": ("SplinterLayer", BertLayerBetterTransformer),
        "tapas": ("TapasLayer", BertLayerBetterTransformer),
        "vilt": ("ViltLayer", ViltLayerBetterTransformer),
        "vit": ("ViTLayer", ViTLayerBetterTransformer),
        "vit_mae": ("ViTMAELayer", ViTLayerBetterTransformer),
        "vit_msn": ("ViTMSNLayer", ViTLayerBetterTransformer),
        "wav2vec2": (
            ["Wav2Vec2EncoderLayer", "Wav2Vec2EncoderLayerStableLayerNorm"],
            Wav2Vec2EncoderLayerBetterTransformer,
        ),
        "whisper": ("WhisperEncoderLayer", WhisperEncoderLayerBetterTransformer),
        "xlm-roberta": ("XLMRobertaLayer", BertLayerBetterTransformer),
        "yolos": ("YolosLayer", ViTLayerBetterTransformer),
    }

    EXCLUDE_FROM_TRANSFORM = {
        # clip's text model uses causal attention, that is most likely not supported in BetterTransformer
        "clip": ["text_model"],
    }

    CAN_NOT_BE_SUPPORTED = {
        "deberta-v2": "DeBERTa v2 does not use a regular attention mechanism, which is not suppored in PyTorch's BetterTransformer.",
        "glpn": "GLPN has a convolutional layer present in the FFN network, which is not suppored in PyTorch's BetterTransformer.",
        "t5": "T5 uses attention bias, which is not suppored in PyTorch's BetterTransformer.",
    }

    @staticmethod
    def cannot_support(model_type: str) -> bool:
        """
        Returns True if a given model type can not be supported by PyTorch's Better Transformer.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type in BetterTransformerManager.CAN_NOT_BE_SUPPORTED

    @staticmethod
    def supports(model_type: str) -> bool:
        """
        Returns True if a given model type is supported by PyTorch's Better Transformer, and integrated in Optimum.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type in BetterTransformerManager.MODEL_MAPPING


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
