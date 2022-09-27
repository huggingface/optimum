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
"""Dummy input generation classes."""

import functools
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from transformers.utils import is_tf_available, is_torch_available


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class NormalizedConfig:
    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "hidden_size"
    NUM_LAYERS = "num_hidden_layers"
    NUM_ATTENTION_HEADS = "num_attention_heads"

    def __init__(self, config: "PretrainedConfig", **kwargs):
        self.config = config
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                raise AttributeError(f"{self.__class__} has not attribute {key}.")

    @classmethod
    def with_args(cls, **kwargs) -> Callable[["PretrainedConfig"], "NormalizedConfig"]:
        return functools.partial(cls, **kwargs)

    def __getattribute__(self, attr_name):
        # if attr_name == attr_name.upper():
        #     return super().__getattr__(self, attr_name)
        if attr_name.startswith("__") or not attr_name.upper() in dir(self.__class__):
            return super().__getattribute__(attr_name)
        else:
            attr = getattr(self.config, super().__getattribute__(attr_name.upper()), None)
            if attr is None:
                raise AttributeError(
                    f'Could not find the attribute named "{attr_name.upper()}" in the normalized config.'
                )
            return attr


class NormalizedSeq2SeqConfig(NormalizedConfig):
    DECODER_NUM_LAYERS = NormalizedConfig.NUM_LAYERS
    ENCODER_NUM_ATTENTION_HEADS = NormalizedConfig.NUM_ATTENTION_HEADS
    DECODER_NUM_ATTENTION_HEADS = NormalizedConfig.NUM_ATTENTION_HEADS


def check_framework_is_available(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        framework = kwargs.get("framework", "pt")
        pt_asked_but_not_available = framework == "pt" and not is_torch_available()
        tf_asked_but_not_available = framework == "tf" and not is_tf_available()
        if pt_asked_but_not_available or tf_asked_but_not_available:
            framework_name = "PyTorch" if framework == "pt" else "TensorFlow"
            raise RuntimeError(f"Requested the {framework_name} framework, but it does not seem installed.")
        return func(*args, **kwargs)

    return wrapper


class DummyInputGenerator(ABC):
    SUPPORTED_INPUT_NAMES = ()

    def supports_input(self, input_name: str) -> bool:
        return any(input_name.startswith(supported_input_name) for supported_input_name in self.SUPPORTED_INPUT_NAMES)

    @abstractmethod
    def generate(self, input_name: str, framework: Optional[str] = "pt"):
        raise NotImplementedError

    @staticmethod
    @check_framework_is_available
    def random_int_tensor(shape: List[int], max_value: int, min_value: int = 0, framework: Optional[str] = "pt"):
        if framework == "pt":
            return torch.randint(low=min_value, high=max_value, size=shape)
        return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tf.int32)

    @staticmethod
    @check_framework_is_available
    def random_float_tensor(shape: List[int], max_value: float, min_value: float = 0, framework: Optional[str] = "pt"):
        if framework == "pt":
            tensor = torch.empty(shape, dtype=torch.float32).uniform_(min_value, max_value)
            return tensor
        return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tf.float32)

    @staticmethod
    @check_framework_is_available
    def constant_tensor(
        shape: List[int], value: Union[int, float] = 1, dtype: Optional[Any] = None, framework: Optional[str] = "pt"
    ):
        if framework == "pt":
            return torch.full(shape, value, dtype=dtype)
        return tf.constant(value, dtype=dtype, shape=shape)


class DummyTextInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_choices: Optional[int] = None,
    ):
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        self.batch_size = random.randint(2, 4) if batch_size is None else batch_size
        self.sequence_length = random.randint(128, 384) if sequence_length is None else sequence_length
        self.num_choices = random.randint(2, 4) if num_choices is None else num_choices

    def generate(self, input_name: str, framework: Optional[str] = "pt"):
        min_value = 0
        max_value = 2 if input_name != "input_ids" else self.vocab_size
        shape = [self.batch_size, self.sequence_length]
        if self.task == "multiple_choice":
            # TODO: check that.
            shape = [self.num_choices, self.batch_size, self.sequence_length]
        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework)


class DummyDecoderTextInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
    )


class DummyPastKeyValuesGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("past_key_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
    ):
        self.num_layers = normalized_config.num_layers
        self.num_attention_heads = normalized_config.num_attention_heads
        self.hidden_size = normalized_config.hidden_size
        self.batch_size = random.randint(2, 4) if batch_size is None else batch_size
        self.sequence_length = random.randint(128, 384) if sequence_length is None else sequence_length

    def generate(self, input_name: str, framework: Optional[str] = "pt"):
        shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework),
                self.random_float_tensor(shape, framework=framework),
            )
            for _ in range(self.num_layers)
        ]


class DummySeq2SeqPastKeyValuesGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("past_key_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedSeq2SeqConfig,
        batch_size: Optional[int] = None,
        encoder_sequence_length: Optional[int] = None,
        sequence_length: Optional[int] = None,
    ):
        self.normalized_config = normalized_config
        self.batch_size = random.randint(2, 4) if batch_size is None else batch_size
        self.sequence_length = random.randint(128, 384) if sequence_length is None else sequence_length
        self.encoder_sequence_length = (
            self.sequence_length if encoder_sequence_length is None else encoder_sequence_length
        )

    def generate(self, input_name: str, framework: Optional[str] = "pt"):
        encoder_shape = (
            self.batch_size,
            self.normalized_config.encoder_num_attention_heads,
            self.encoder_sequence_length,
            self.normalized_config.hidden_size // self.normalized_config.encoder_num_attention_heads,
        )
        decoder_shape = (
            self.batch_size,
            self.normalized_config.decoder_num_attention_heads,
            self.sequence_length,
            self.normalized_config.hidden_size // self.normalized_config.decoder_num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(decoder_shape, framework=framework),
                self.random_float_tensor(decoder_shape, framework=framework),
                self.random_float_tensor(encoder_shape, framework=framework),
                self.random_float_tensor(encoder_shape, framework=framework),
            )
            for _ in range(self.normalized_config.decoder_num_layers)
        ]
