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
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

from transformers.utils import is_tf_available, is_torch_available


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class NormalizedConfig:
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

    def __getattribute__(self, attr_name):
        if attr_name.startswith("__") or not attr_name.upper() in dir(self.__class__):
            return super().__getattribute__(attr_name)
        else:
            attr = getattr(self.config, super().__getattribute__(attr_name.upper()), None)
            if attr is None:
                raise AttributeError(
                    f'Could not find the attribute named "{attr_name.upper()}" in the normalized config.'
                )
            return attr

    def has_attribute(self, attr_name):
        return getattr(self.config, super().__getattribute__(attr_name.upper()), None) is not None


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
    def generate(self, input_name: str, framework: str = "pt"):
        raise NotImplementedError

    @staticmethod
    @check_framework_is_available
    def random_int_tensor(shape: List[int], max_value: int, min_value: int = 0, framework: str = "pt"):
        if framework == "pt":
            return torch.randint(low=min_value, high=max_value, size=shape)
        return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tf.int32)

    @staticmethod
    @check_framework_is_available
    def random_float_tensor(shape: List[int], min_value: float = 0, max_value: float = 1, framework: str = "pt"):
        if framework == "pt":
            tensor = torch.empty(shape, dtype=torch.float32).uniform_(min_value, max_value)
            return tensor
        return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tf.float32)

    @staticmethod
    @check_framework_is_available
    def constant_tensor(
        shape: List[int], value: Union[int, float] = 1, dtype: Optional[Any] = None, framework: str = "pt"
    ):
        if framework == "pt":
            return torch.full(shape, value, dtype=dtype)
        return tf.constant(value, dtype=dtype, shape=shape)

    @staticmethod
    def _infer_framework_from_input(input_) -> str:
        framework = None
        if is_torch_available() and isinstance(input_, torch.Tensor):
            framework = "pt"
        elif is_tf_available() and isinstance(input_, tf.Tensor):
            framework = "tf"
        else:
            raise RuntimeError(f"Could not infer the framework from {input_}")
        return framework

    @classmethod
    def concat_inputs(cls, inputs, dim: int):
        if not inputs:
            raise ValueError("You did not provide any inputs to concat")
        framework = cls._infer_framework_from_input(inputs[0])
        if framework == "pt":
            return torch.cat(inputs, dim=dim)
        return tf.concat(inputs, axis=dim)

    @classmethod
    def pad_input_on_dim(cls, input_, dim: int, desired_length: Optional[int] = None, padding_length: Optional[int] = None, value: Union[int, float] = 1, dtype: Optional[Any] = None):
        if (desired_length is None and padding_length is None) or (desired_length is not None and padding_length is not None):
            raise ValueError("You need to provide either `desired_length` or `padding_length`")
        framework = cls._infer_framework_from_input(input_)
        shape = input_.shape
        padding_shape = list(shape)
        diff = desired_length - shape[dim] if desired_length else padding_length
        if diff <= 0:
            return input_
        padding_shape[dim] = diff
        return cls.concat_inputs([input_, cls.constant_tensor(padding_shape, value=value, dtype=dtype, framework=framework)], dim=dim)


class DummyTextInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = 2,
        sequence_length: int = 16,
        num_choices: int = 4,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
    ):
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length
        if random_num_choices_range:
            low, high = random_num_choices_range
            self.num_choices = random.randint(low, high)
        else:
            self.num_choices = num_choices

    def generate(self, input_name: str, framework: str = "pt"):
        min_value = 0
        max_value = 2 if input_name != "input_ids" else self.vocab_size
        shape = [self.batch_size, self.sequence_length]
        if self.task == "multiple-choice":
            shape = [self.batch_size, self.num_choices, self.sequence_length]
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
        normalized_config: NormalizedTextConfig,
        batch_size: int = 2,
        sequence_length: int = 16,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
    ):
        self.num_layers = normalized_config.num_layers
        self.num_attention_heads = normalized_config.num_attention_heads
        self.hidden_size = normalized_config.hidden_size
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length

    def generate(self, input_name: str, framework: str = "pt"):
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
        batch_size: int = 2,
        sequence_length: int = 16,
        encoder_sequence_length: Optional[int] = None,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
    ):
        self.normalized_config = normalized_config
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length
        self.encoder_sequence_length = (
            self.sequence_length if encoder_sequence_length is None else encoder_sequence_length
        )

    def generate(self, input_name: str, framework: str = "pt"):
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


class DummyVisionInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = 2,
        num_channels: int = 3,
        width: int = 224,
        height: int = 224,
    ):
        self.task = task
        # Some vision models can take any input sizes, in this case we use the values provided as parameters.
        if normalized_config.has_attribute("image_size"):
            self.image_size = normalized_config.image_size
        else:
            self.image_size = (width, height)
        if normalized_config.has_attribute("num_channels"):
            self.num_channels = normalized_config.num_channels
        else:
            self.num_channels = num_channels

        if not isinstance(self.image_size, (tuple, list)):
            self.image_size = (self.image_size, self.image_size)
        self.batch_size = batch_size
        self.height, self.width = self.image_size

    def generate(self, input_name: str, framework: str = "pt"):
        if input_name == "pixel_mask":
            shape = [self.batch_size, self.height, self.width]
            return self.random_int_tensor(shape, max_value=1, framework=framework)
        shape = [self.batch_size, self.num_channels, self.height, self.width]
        return self.random_float_tensor(shape, framework=framework)
