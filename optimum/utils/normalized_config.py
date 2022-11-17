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
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class NormalizedConfig:
    """
    Handles the normalization of [`PretrainedConfig`] attribute names, allowing to access attributes in a general way.

    Attributes:
        config ([`PretrainedConfig`]):
            The config to normalize.
    """

    def __init__(self, config: "PretrainedConfig", allow_new: bool = False, **kwargs):
        self.config = config
        for key, value in kwargs.items():
            if allow_new or hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                raise AttributeError(f"{self.__class__} has not attribute {key}.")

    @classmethod
    def with_args(cls, allow_new: bool = False, **kwargs) -> Callable[["PretrainedConfig"], "NormalizedConfig"]:
        return functools.partial(cls, allow_new=allow_new, **kwargs)

    def __getattr__(self, attr_name):
        attr_name = attr_name.split(".")
        leaf_attr_name = attr_name[-1]
        config = self.config
        for attr in attr_name[:-1]:
            config = getattr(config, attr)
        attr = getattr(config, super().__getattribute__(leaf_attr_name.upper()), None)
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
