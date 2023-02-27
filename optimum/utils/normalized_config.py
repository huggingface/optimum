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
"""Normalization configuration classes."""

import functools
from typing import Callable, Dict, Type, Union

from transformers import PretrainedConfig


class NormalizedConfig:
    """
    Handles the normalization of [`PretrainedConfig`] attribute names, allowing to access attributes in a general way.

    Attributes:
        config ([`PretrainedConfig`]):
            The config to normalize.
    """

    def __init__(self, config: Union[PretrainedConfig, Dict], allow_new: bool = False, **kwargs):
        self.config = config if isinstance(config, PretrainedConfig) else PretrainedConfig.from_dict(config)
        for key, value in kwargs.items():
            if allow_new or hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                raise AttributeError(
                    f"{self.__class__} has not attribute {key}. Set allow_new=True to add a new attribute."
                )

    @classmethod
    def with_args(cls, allow_new: bool = False, **kwargs) -> Callable[[PretrainedConfig], "NormalizedConfig"]:
        return functools.partial(cls, allow_new=allow_new, **kwargs)

    def __getattr__(self, attr_name):
        attr_name = attr_name.split(".")
        leaf_attr_name = attr_name[-1]
        config = self.config
        for attr in attr_name[:-1]:
            config = getattr(config, attr)

        attr = getattr(config, super().__getattribute__(leaf_attr_name.upper()), None)

        # If the attribute was not specified manually, try to fallback on the attribute_map.
        if attr is None:
            attribute_map = getattr(self.config, "attribute_map", {})
            attr = getattr(self.config, attribute_map.get(leaf_attr_name, ""), None)

        if attr is None:
            raise AttributeError(f'Could not find the attribute named "{leaf_attr_name}" in the normalized config.')
        return attr

    def has_attribute(self, attr_name):
        try:
            self.__getattr__(attr_name)
        except AttributeError:
            return False
        return True


class NormalizedTextConfig(NormalizedConfig):
    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "hidden_size"
    NUM_LAYERS = "num_hidden_layers"
    NUM_ATTENTION_HEADS = "num_attention_heads"
    EOS_TOKEN_ID = "eos_token_id"


class NormalizedSeq2SeqConfig(NormalizedTextConfig):
    ENCODER_NUM_LAYERS = NormalizedTextConfig.NUM_LAYERS
    DECODER_NUM_LAYERS = NormalizedTextConfig.NUM_LAYERS
    ENCODER_NUM_ATTENTION_HEADS = NormalizedTextConfig.NUM_ATTENTION_HEADS
    DECODER_NUM_ATTENTION_HEADS = NormalizedTextConfig.NUM_ATTENTION_HEADS


class NormalizedVisionConfig(NormalizedConfig):
    IMAGE_SIZE = "image_size"
    NUM_CHANNELS = "num_channels"


class NormalizedTextAndVisionConfig(NormalizedTextConfig, NormalizedVisionConfig):
    TEXT_CONFIG = None
    VISION_CONFIG = None

    def __getattr__(self, attr_name):
        if self.TEXT_CONFIG is not None and attr_name.upper() in dir(NormalizedTextConfig):
            attr_name = f"{self.TEXT_CONFIG}.{attr_name}"
        elif self.VISION_CONFIG is not None and attr_name.upper() in dir(NormalizedVisionConfig):
            attr_name = f"{self.VISION_CONFIG}.{attr_name}"
        return super().__getattr__(attr_name)


class NormalizedEncoderDecoderConfig(NormalizedConfig):
    ENCODER_NORMALIZED_CONFIG_CLASS = None
    DECODER_NORMALIZED_CONFIG_CLASS = None

    def __getattr__(self, attr_name):
        if self.ENCODER_NORMALIZED_CONFIG_CLASS is not None and attr_name.upper() in dir(
            self.ENCODER_NORMALIZED_CONFIG_CLASS
        ):
            return self.ENCODER_NORMALIZED_CONFIG_CLASS.__getattr__(attr_name)
        if self.DECODER_NORMALIZED_CONFIG_CLASS is not None and attr_name.upper() in dir(
            self.DECODER_NORMALIZED_CONFIG_CLASS
        ):
            return self.DECODER_NORMALIZED_CONFIG_CLASS.__getattr__(attr_name)

        return super().__getattr__(attr_name)


# TODO: this config is bug prone, as `encoder_attention_heads` and `decoder_attention_heads` may be different
BartLikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="encoder_attention_heads",
    hidden_size="d_model",
)

GPT2LikeNormalizedTextConfig = NormalizedTextConfig.with_args(num_attention_heads="n_head", hidden_size="n_embd")
T5LikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="num_heads",
    hidden_size="d_model",
)
WhisperLikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    hidden_size="d_model",
)

TrOCRLikeNormalizedTextConfig = NormalizedSeq2SeqConfig.with_args(
    decoder_num_layers="decoder_layers",
    num_layers="decoder_layers",
    decoder_num_attention_heads="decoder_attention_heads",
    hidden_size="cross_attention_hidden_size",
)

SpeechToTextLikeNormalizedTextConfig = NormalizedSeq2SeqConfig.with_args(
    decoder_num_layers="decoder_layers",
    num_layers="decoder_layers",
    input_features_per_channel="input_feat_per_channel",
    allow_new=True,
)


class NormalizedConfigManager:
    """
    A class that contains all the information needed by ONNX Runtime optimization for a given model type.
    Attributes:
        _conf (`Dict[str, tuple]`):
            A dictionary mapping each supported model type to a tuple containing the number of attention heads
            and the hidden size model config attribute names as well as the corresponding ONNX Runtime model type.
    """

    """
    TODO: missing normalized configs (currently not useful)
        ['beit',
        'clip',
        'convbert',
        'convnext',
        'data2vec-text',
        'data2vec-vision',
        'detr',
        'flaubert',
        'groupvit',
        'ibert',
        'layoutlm',
        'layoutlmv3',
        'levit',
        'mobilebert',
        'mobilevit',
        'owlvit',
        'perceiver',
        'roformer',
        'segformer',
        'squeezebert',
    """

    # Contribution note: Please add new models in alphabetical order
    _conf = {
        "albert": NormalizedTextConfig,
        "bart": BartLikeNormalizedTextConfig,
        "bert": NormalizedTextConfig,
        # "big_bird": NormalizedTextConfig,
        # "bigbird_pegasus": BartLikeNormalizedTextConfig,
        "blenderbot": BartLikeNormalizedTextConfig,
        "blenderbot_small": BartLikeNormalizedTextConfig,
        "bloom": NormalizedTextConfig.with_args(num_layers="n_layer"),
        "camembert": NormalizedTextConfig,
        "codegen": GPT2LikeNormalizedTextConfig,
        "deberta": NormalizedTextConfig,
        "deberta-v2": NormalizedTextConfig,
        "deit": NormalizedVisionConfig,
        "distilbert": NormalizedTextConfig.with_args(num_attention_heads="n_heads", hidden_size="dim"),
        "electra": NormalizedTextConfig,
        "gpt2": GPT2LikeNormalizedTextConfig,
        "gpt_neo": NormalizedTextConfig.with_args(num_attention_heads="num_heads"),
        "gpt_neox": NormalizedTextConfig,
        "gptj": GPT2LikeNormalizedTextConfig,
        "longt5": T5LikeNormalizedTextConfig,
        "marian": BartLikeNormalizedTextConfig,
        "mbart": BartLikeNormalizedTextConfig,
        "mt5": T5LikeNormalizedTextConfig,
        "m2m_100": BartLikeNormalizedTextConfig,
        "nystromformer": NormalizedTextConfig,
        "pegasus": BartLikeNormalizedTextConfig,
        "poolformer": NormalizedVisionConfig,
        "resnet": NormalizedVisionConfig,
        "roberta": NormalizedTextConfig,
        "speech_to_text": SpeechToTextLikeNormalizedTextConfig,
        "splinter": NormalizedTextConfig,
        "t5": T5LikeNormalizedTextConfig,
        "trocr": TrOCRLikeNormalizedTextConfig,
        "whisper": WhisperLikeNormalizedTextConfig,
        "vision-encoder-decoder": NormalizedEncoderDecoderConfig,
        "vit": NormalizedVisionConfig,
        "xlm-roberta": NormalizedTextConfig,
        "yolos": NormalizedVisionConfig,
    }

    @classmethod
    def check_supported_model(cls, model_type: str):
        if model_type not in cls._conf:
            model_types = ", ".join(cls._conf.keys())
            raise KeyError(
                f"{model_type} model type is not supported yet in NormalizedConfig. Only {model_types} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )

    @classmethod
    def get_normalized_config_class(cls, model_type: str) -> Type:
        cls.check_supported_model(model_type)
        return cls._conf[model_type]
