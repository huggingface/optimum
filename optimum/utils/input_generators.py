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
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from transformers.utils import is_tf_available, is_torch_available

from .normalized_config import NormalizedConfig, NormalizedSeq2SeqConfig, NormalizedTextConfig, NormalizedVisionConfig


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


def check_framework_is_available(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        framework = kwargs.get("framework", "pt")
        pt_asked_but_not_available = framework == "pt" and not is_torch_available()
        tf_asked_but_not_available = framework == "tf" and not is_tf_available()
        if (pt_asked_but_not_available or tf_asked_but_not_available) and framework != "np":
            framework_name = "PyTorch" if framework == "pt" else "TensorFlow"
            raise RuntimeError(f"Requested the {framework_name} framework, but it does not seem installed.")
        return func(*args, **kwargs)

    return wrapper


DEFAULT_DUMMY_SHAPES = {
    "batch_size": 2,
    "sequence_length": 16,
    "num_choices": 4,
    # image
    "width": 64,
    "height": 64,
    "num_channels": 3,
    # audio
    "feature_size": 80,
    "nb_max_frames": 3000,
    "audio_sequence_length": 16000,
}


class DummyInputGenerator(ABC):
    """
    Generates dummy inputs for the supported input names, in the requested framework.
    """

    SUPPORTED_INPUT_NAMES = ()

    def supports_input(self, input_name: str) -> bool:
        """
        Checks whether the `DummyInputGenerator` supports the generation of the requested input.

        Args:
            input_name (`str`):
                The name of the input to generate.

        Returns:
            `bool`: A boolean specifying whether the input is supported.

        """
        return any(input_name.startswith(supported_input_name) for supported_input_name in self.SUPPORTED_INPUT_NAMES)

    @abstractmethod
    def generate(self, input_name: str, framework: str = "pt"):
        """
        Generates the dummy input matching `input_name` for the requested framework.

        Args:
            input_name (`str`):
                The name of the input to generate.
            framework (`str`, *optional*, defaults to `"pt"`):
                The requested framework.

        Returns:
            A tensor in the requested framework of the input.
        """
        raise NotImplementedError

    @staticmethod
    @check_framework_is_available
    def random_int_tensor(shape: List[int], max_value: int, min_value: int = 0, framework: str = "pt"):
        """
        Generates a tensor of random integers in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            max_value (`int`):
                The maximum value allowed.
            min_value (`int`, *optional*, defaults to 0):
                The minimum value allowed.
            framework (`str`, *optional*, defaults to `"pt"`):
                The requested framework.

        Returns:
            A random tensor in the requested framework.
        """
        if framework == "pt":
            return torch.randint(low=min_value, high=max_value, size=shape)
        elif framework == "tf":
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tf.int64)
        else:
            return np.random.randint(min_value, high=max_value, size=shape, dtype=np.int64)

    @staticmethod
    @check_framework_is_available
    def random_float_tensor(shape: List[int], min_value: float = 0, max_value: float = 1, framework: str = "pt"):
        """
        Generates a tensor of random floats in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            min_value (`float`, *optional*, defaults to 0):
                The minimum value allowed.
            max_value (`float`, *optional*, defaults to 1):
                The maximum value allowed.
            framework (`str`, *optional*, defaults to `"pt"`):
                The requested framework.

        Returns:
            A random tensor in the requested framework.
        """
        if framework == "pt":
            tensor = torch.empty(shape, dtype=torch.float32).uniform_(min_value, max_value)
            return tensor
        elif framework == "tf":
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tf.float32)
        else:
            return np.random.uniform(low=min_value, high=max_value, size=shape).astype(np.float32)

    @staticmethod
    @check_framework_is_available
    def constant_tensor(
        shape: List[int], value: Union[int, float] = 1, dtype: Optional[Any] = None, framework: str = "pt"
    ):
        """
        Generates a constant tensor.

        Args:
            shape (`List[int]`):
                The shape of the constant tensor.
            value (`Union[int, float]`, *optional*, defaults to 1):
                The value to fill the constant tensor with.
            dtype (`Any`, *optional*):
                The dtype of the constant tensor.
            framework (`str`, *optional*, defaults to `"pt"`):
                The requested framework.

        Returns:
            A constant tensor in the requested framework.
        """
        if framework == "pt":
            return torch.full(shape, value, dtype=dtype)
        elif framework == "tf":
            return tf.constant(value, dtype=dtype, shape=shape)
        else:
            return np.full(shape, value, dtype=dtype)

    @staticmethod
    def _infer_framework_from_input(input_) -> str:
        framework = None
        if is_torch_available() and isinstance(input_, torch.Tensor):
            framework = "pt"
        elif is_tf_available() and isinstance(input_, tf.Tensor):
            framework = "tf"
        elif isinstance(input_, np.ndarray):
            framework = "np"
        else:
            raise RuntimeError(f"Could not infer the framework from {input_}")
        return framework

    @classmethod
    def concat_inputs(cls, inputs, dim: int):
        """
        Concatenates inputs together.

        Args:
            inputs:
                The list of tensors in a given framework to concatenate.
            dim (`int`):
                The dimension along which to concatenate.
        Returns:
            The tensor of the concatenation.
        """
        if not inputs:
            raise ValueError("You did not provide any inputs to concat")
        framework = cls._infer_framework_from_input(inputs[0])
        if framework == "pt":
            return torch.cat(inputs, dim=dim)
        elif framework == "tf":
            return tf.concat(inputs, axis=dim)
        else:
            return np.concatenate(inputs, axis=dim)

    @classmethod
    def pad_input_on_dim(
        cls,
        input_,
        dim: int,
        desired_length: Optional[int] = None,
        padding_length: Optional[int] = None,
        value: Union[int, float] = 1,
        dtype: Optional[Any] = None,
    ):
        """
        Pads an input either to the desired length, or by a padding length.

        Args:
            input_:
                The tensor to pad.
            dim (`int`):
                The dimension along which to pad.
            desired_length (`int`, *optional*):
                The desired length along the dimension after padding.
            padding_length (`int`, *optional*):
                The length to pad along the dimension.
            value (`Union[int, float]`, *optional*, defaults to 1):
                The value to use for padding.
            dtype (`Any`, *optional*):
                The dtype of the padding.

        Returns:
            The padded tensor.
        """
        if (desired_length is None and padding_length is None) or (
            desired_length is not None and padding_length is not None
        ):
            raise ValueError("You need to provide either `desired_length` or `padding_length`")
        framework = cls._infer_framework_from_input(input_)
        shape = input_.shape
        padding_shape = list(shape)
        diff = desired_length - shape[dim] if desired_length else padding_length
        if diff <= 0:
            return input_
        padding_shape[dim] = diff
        return cls.concat_inputs(
            [input_, cls.constant_tensor(padding_shape, value=value, dtype=dtype, framework=framework)], dim=dim
        )


class DummyTextInputGenerator(DummyInputGenerator):
    """
    Generates dummy encoder text inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        **kwargs,
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
    """
    Generates dummy decoder text inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
    )


class DummySeq2SeqDecoderTextInputGenerator(DummyDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_outputs",
        "encoder_hidden_states",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
        )

        self.hidden_size = normalized_config.hidden_size

    def generate(self, input_name: str, framework: str = "pt"):
        if input_name in ["encoder_outputs", "encoder_hidden_states"]:
            return (
                self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.hidden_size],
                    min_value=0,
                    max_value=1,
                    framework=framework,
                ),
                None,
                None,
            )
        return super().generate(input_name, framework=framework)


class DummyPastKeyValuesGenerator(DummyInputGenerator):
    """
    Generates dummy past_key_values inputs.
    """

    SUPPORTED_INPUT_NAMES = ("past_key_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
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
    """
    Generates dummy past_key_values inputs for seq2seq architectures.
    """

    SUPPORTED_INPUT_NAMES = ("past_key_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedSeq2SeqConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        encoder_sequence_length: Optional[int] = None,
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
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


# TODO: should it just be merged to DummyTextInputGenerator?
class DummyBboxInputGenerator(DummyInputGenerator):
    """
    Generates dummy bbox inputs.
    """

    SUPPORTED_INPUT_NAMES = ("bbox",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        # self.max_2d_position_embeddings = normalized_config.max_2d_position_embeddings
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
        return self.random_int_tensor(
            [self.batch_size, self.sequence_length, 4],
            # TODO: find out why this fails with the commented code.
            1,  # self.max_2d_position_embeddings - 1,
            framework=framework,
        )


class DummyVisionInputGenerator(DummyInputGenerator):
    """
    Generates dummy vision inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
        "sample",
        "latent_sample",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        self.task = task
        # Some vision models can take any input sizes, in this case we use the values provided as parameters.
        if normalized_config.has_attribute("image_size"):
            self.image_size = normalized_config.image_size
        else:
            self.image_size = (height, width)
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
            return self.random_int_tensor(
                shape=[self.batch_size, self.height, self.width], max_value=1, framework=framework
            )
        else:
            return self.random_float_tensor(
                shape=[self.batch_size, self.num_channels, self.height, self.width], framework=framework
            )


class DummyAudioInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input_features", "input_values")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        feature_size: int = DEFAULT_DUMMY_SHAPES["feature_size"],
        nb_max_frames: int = DEFAULT_DUMMY_SHAPES["nb_max_frames"],
        audio_sequence_length: int = DEFAULT_DUMMY_SHAPES["audio_sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.normalized_config = normalized_config

        self.feature_size = feature_size
        self.nb_max_frames = nb_max_frames
        self.batch_size = batch_size
        self.sequence_length = audio_sequence_length

    def generate(self, input_name: str, framework: str = "pt"):
        if input_name == "input_values":  # raw waveform
            return self.random_float_tensor(
                shape=[self.batch_size, self.sequence_length], min_value=-1, max_value=1, framework=framework
            )
        else:
            return self.random_float_tensor(
                shape=[self.batch_size, self.feature_size, self.nb_max_frames],
                min_value=-1,
                max_value=1,
                framework=framework,
            )


class DummyTimestepInputGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """

    SUPPORTED_INPUT_NAMES = ("timestep",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        self.vocab_size = normalized_config.vocab_size

        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size

    def generate(self, input_name: str, framework: str = "pt"):
        shape = [self.batch_size]
        return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework)


class DummyTrainingLabelsInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = ("labels", "start_positions", "end_positions")

    def generate(self, input_name: str, framework: str = "pt"):
        max_value = 1 if self.task != "seq2seq-lm" else self.vocab_size
        shape = [self.batch_size, self.sequence_length]
        if self.task in [
            "default",
            "sequence-classification",
            "multiple-choice",
            "question-answering",
            "image-classification",
        ]:
            shape = [self.batch_size]

        return self.random_int_tensor(shape, max_value=max_value, framework=framework)
