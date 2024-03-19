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

from ...utils.import_utils import check_if_transformers_greater
from .decoder_models import (
    BarkAttentionLayerBetterTransformer,
    BartAttentionLayerBetterTransformer,
    BlenderbotAttentionLayerBetterTransformer,
    BloomAttentionLayerBetterTransformer,
    CodegenAttentionLayerBetterTransformer,
    GPT2AttentionLayerBetterTransformer,
    GPTJAttentionLayerBetterTransformer,
    GPTNeoAttentionLayerBetterTransformer,
    GPTNeoXAttentionLayerBetterTransformer,
    M2M100AttentionLayerBetterTransformer,
    MarianAttentionLayerBetterTransformer,
    OPTAttentionLayerBetterTransformer,
    PegasusAttentionLayerBetterTransformer,
    T5AttentionLayerBetterTransformer,
)
from .encoder_models import (
    AlbertLayerBetterTransformer,
    BartEncoderLayerBetterTransformer,
    BertLayerBetterTransformer,
    CLIPLayerBetterTransformer,
    DistilBertLayerBetterTransformer,
    FSMTEncoderLayerBetterTransformer,
    MBartEncoderLayerBetterTransformer,
    ProphetNetEncoderLayerBetterTransformer,
    ViltLayerBetterTransformer,
    ViTLayerBetterTransformer,
    Wav2Vec2EncoderLayerBetterTransformer,
)


class BetterTransformerManager:
    MODEL_MAPPING = {
        "albert": {"AlbertLayer": AlbertLayerBetterTransformer},
        "bark": {"BarkSelfAttention": BarkAttentionLayerBetterTransformer},
        "bart": {
            "BartEncoderLayer": BartEncoderLayerBetterTransformer,
            "BartAttention": BartAttentionLayerBetterTransformer,
        },
        "bert": {"BertLayer": BertLayerBetterTransformer},
        "bert-generation": {"BertGenerationLayer": BertLayerBetterTransformer},
        "blenderbot": {"BlenderbotAttention": BlenderbotAttentionLayerBetterTransformer},
        "bloom": {"BloomAttention": BloomAttentionLayerBetterTransformer},
        "camembert": {"CamembertLayer": BertLayerBetterTransformer},
        "blip-2": {"T5Attention": T5AttentionLayerBetterTransformer},
        "clip": {"CLIPEncoderLayer": CLIPLayerBetterTransformer},
        "codegen": {"CodeGenAttention": CodegenAttentionLayerBetterTransformer},
        "data2vec-text": {"Data2VecTextLayer": BertLayerBetterTransformer},
        "deit": {"DeiTLayer": ViTLayerBetterTransformer},
        "distilbert": {"TransformerBlock": DistilBertLayerBetterTransformer},
        "electra": {"ElectraLayer": BertLayerBetterTransformer},
        "ernie": {"ErnieLayer": BertLayerBetterTransformer},
        "fsmt": {"EncoderLayer": FSMTEncoderLayerBetterTransformer},
        "gpt2": {"GPT2Attention": GPT2AttentionLayerBetterTransformer},
        "gptj": {"GPTJAttention": GPTJAttentionLayerBetterTransformer},
        "gpt_neo": {"GPTNeoSelfAttention": GPTNeoAttentionLayerBetterTransformer},
        "gpt_neox": {"GPTNeoXAttention": GPTNeoXAttentionLayerBetterTransformer},
        "hubert": {"HubertEncoderLayer": Wav2Vec2EncoderLayerBetterTransformer},
        "layoutlm": {"LayoutLMLayer": BertLayerBetterTransformer},
        "m2m_100": {
            "M2M100EncoderLayer": MBartEncoderLayerBetterTransformer,
            "M2M100Attention": M2M100AttentionLayerBetterTransformer,
        },
        "marian": {
            "MarianEncoderLayer": BartEncoderLayerBetterTransformer,
            "MarianAttention": MarianAttentionLayerBetterTransformer,
        },
        "markuplm": {"MarkupLMLayer": BertLayerBetterTransformer},
        "mbart": {"MBartEncoderLayer": MBartEncoderLayerBetterTransformer},
        "opt": {"OPTAttention": OPTAttentionLayerBetterTransformer},
        "pegasus": {"PegasusAttention": PegasusAttentionLayerBetterTransformer},
        "rembert": {"RemBertLayer": BertLayerBetterTransformer},
        "prophetnet": {"ProphetNetEncoderLayer": ProphetNetEncoderLayerBetterTransformer},
        "roberta": {"RobertaLayer": BertLayerBetterTransformer},
        "roc_bert": {"RoCBertLayer": BertLayerBetterTransformer},
        "roformer": {"RoFormerLayer": BertLayerBetterTransformer},
        "splinter": {"SplinterLayer": BertLayerBetterTransformer},
        "tapas": {"TapasLayer": BertLayerBetterTransformer},
        "t5": {"T5Attention": T5AttentionLayerBetterTransformer},
        "vilt": {"ViltLayer": ViltLayerBetterTransformer},
        "vit": {"ViTLayer": ViTLayerBetterTransformer},
        "vit_mae": {"ViTMAELayer": ViTLayerBetterTransformer},
        "vit_msn": {"ViTMSNLayer": ViTLayerBetterTransformer},
        "wav2vec2": {
            "Wav2Vec2EncoderLayer": Wav2Vec2EncoderLayerBetterTransformer,
            "Wav2Vec2EncoderLayerStableLayerNorm": Wav2Vec2EncoderLayerBetterTransformer,
        },
        "xlm-roberta": {"XLMRobertaLayer": BertLayerBetterTransformer},
        "yolos": {"YolosLayer": ViTLayerBetterTransformer},
    }

    OVERWRITE_METHODS = {
        # "llama": {"LlamaModel": ("_prepare_decoder_attention_mask", _llama_prepare_decoder_attention_mask)}
    }

    EXCLUDE_FROM_TRANSFORM = {
        # clip's text model uses causal attention, that is most likely not supported in BetterTransformer
        "clip": ["text_model"],
        # blip-2's Q-former and vision model should not be identified as the last layers of the model
        "blip-2": ["qformer.encoder.layer", "vision_model.encoder.layers"],
        # bark.codec_model.encoder is not supported in BetterTransformer
        "bark": ["codec_model.encoder.layers"],
    }

    CAN_NOT_BE_SUPPORTED = {
        "deberta-v2": "DeBERTa v2 does not use a regular attention mechanism, which is not supported in PyTorch's BetterTransformer.",
        "glpn": "GLPN has a convolutional layer present in the FFN network, which is not supported in PyTorch's BetterTransformer.",
    }

    NOT_REQUIRES_NESTED_TENSOR = {
        "bark",
        "blenderbot",
        "bloom",
        "codegen",
        "gpt2",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "opt",
        "pegasus",
        "t5",
    }

    NOT_REQUIRES_STRICT_VALIDATION = {
        "blenderbot",
        "blip-2",
        "bloom",
        "codegen",
        "gpt2",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "opt",
        "pegasus",
        "t5",
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

    @staticmethod
    def requires_nested_tensor(model_type: str) -> bool:
        """
        Returns True if the BetterTransformer implementation for a given architecture uses nested tensors, False otherwise.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type not in BetterTransformerManager.NOT_REQUIRES_NESTED_TENSOR

    @staticmethod
    def requires_strict_validation(model_type: str) -> bool:
        """
        Returns True if the architecture requires to make sure all conditions of `validate_bettertransformer` are met.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type not in BetterTransformerManager.NOT_REQUIRES_STRICT_VALIDATION


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
