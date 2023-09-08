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
    "point_batch_size": 3,
    "nb_points_per_image": 2,
    # audio
    "feature_size": 80,
    "nb_max_frames": 3000,
    "audio_sequence_length": 16000,
}


class DTYPE_MAPPER:
    @classmethod
    def np(cls, dtype):
        mapping = {
            "fp32": np.float32,
            "fp16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "int8": np.int8,
            "bool": bool,
        }
        return mapping[dtype]

    @classmethod
    def pt(cls, dtype):
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int8": torch.int8,
            "bool": torch.bool,
        }
        return mapping[dtype]

    @classmethod
    def tf(cls, dtype):
        mapping = {
            "fp32": tf.float32,
            "fp16": tf.float16,
            "bf16": tf.bfloat16,
            "int64": tf.int64,
            "int32": tf.int32,
            "int8": tf.int8,
            "bool": tf.bool,
        }
        return mapping[dtype]


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
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """
        Generates the dummy input matching `input_name` for the requested framework.

        Args:
            input_name (`str`):
                The name of the input to generate.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            int_dtype (`str`, defaults to `"int64"`):
                The dtypes of generated integer tensors.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtypes of generated float tensors.

        Returns:
            A tensor in the requested framework of the input.
        """
        raise NotImplementedError

    @staticmethod
    @check_framework_is_available
    def random_int_tensor(
        shape: List[int], max_value: int, min_value: int = 0, framework: str = "pt", dtype: str = "int64"
    ):
        """
        Generates a tensor of random integers in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            max_value (`int`):
                The maximum value allowed.
            min_value (`int`, defaults to 0):
                The minimum value allowed.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"int64"`):
                The dtype of the generated integer tensor. Could be "int64", "int32", "int8".

        Returns:
            A random tensor in the requested framework.
        """
        if framework == "pt":
            return torch.randint(low=min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.pt(dtype))
        elif framework == "tf":
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
        else:
            return np.random.randint(min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.np(dtype))

    @staticmethod
    @check_framework_is_available
    def random_float_tensor(
        shape: List[int], min_value: float = 0, max_value: float = 1, framework: str = "pt", dtype: str = "fp32"
    ):
        """
        Generates a tensor of random floats in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            min_value (`float`, defaults to 0):
                The minimum value allowed.
            max_value (`float`, defaults to 1):
                The maximum value allowed.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"fp32"`):
                The dtype of the generated float tensor. Could be "fp32", "fp16", "bf16".

        Returns:
            A random tensor in the requested framework.
        """
        if framework == "pt":
            tensor = torch.empty(shape, dtype=DTYPE_MAPPER.pt(dtype)).uniform_(min_value, max_value)
            return tensor
        elif framework == "tf":
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
        else:
            return np.random.uniform(low=min_value, high=max_value, size=shape).astype(DTYPE_MAPPER.np(dtype))

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
            value (`Union[int, float]`, defaults to 1):
                The value to fill the constant tensor with.
            dtype (`Optional[Any]`, defaults to `None`):
                The dtype of the constant tensor.
            framework (`str`, defaults to `"pt"`):
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
            desired_length (`Optional[int]`, defaults to `None`):
                The desired length along the dimension after padding.
            padding_length (`Optional[int]`, defaults to `None`):
                The length to pad along the dimension.
            value (`Union[int, float]`, defaults to 1):
                The value to use for padding.
            dtype (`Optional[Any]`, defaults to `None`):
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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        min_value = 0
        max_value = 2 if input_name != "input_ids" else self.vocab_size
        shape = [self.batch_size, self.sequence_length]
        if self.task == "multiple-choice":
            shape = [self.batch_size, self.num_choices, self.sequence_length]
        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["encoder_outputs", "encoder_hidden_states"]:
            return (
                self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.hidden_size],
                    min_value=0,
                    max_value=1,
                    framework=framework,
                    dtype=float_dtype,
                ),
                None,
                None,
            )
        return super().generate(input_name, framework=framework, int_dtype=int_dtype)


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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
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
                self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        return self.random_int_tensor(
            [self.batch_size, self.sequence_length, 4],
            # TODO: find out why this fails with the commented code.
            1,  # self.max_2d_position_embeddings - 1,
            framework=framework,
            dtype=int_dtype,
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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "pixel_mask":
            return self.random_int_tensor(
                shape=[self.batch_size, self.height, self.width],
                max_value=1,
                framework=framework,
                dtype=int_dtype,
            )
        else:
            return self.random_float_tensor(
                shape=[self.batch_size, self.num_channels, self.height, self.width],
                framework=framework,
                dtype=float_dtype,
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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "input_values":  # raw waveform
            return self.random_float_tensor(
                shape=[self.batch_size, self.sequence_length],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        else:
            return self.random_float_tensor(
                shape=[self.batch_size, self.feature_size, self.nb_max_frames],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )


class DummyTimestepInputGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "timestep",
        "text_embeds",
        "time_ids",
    )

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
        self.text_encoder_projection_dim = normalized_config.text_encoder_projection_dim
        self.time_ids = 5 if normalized_config.requires_aesthetics_score else 6
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size]

        if input_name == "timestep":
            return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=int_dtype)

        shape.append(self.text_encoder_projection_dim if input_name == "text_embeds" else self.time_ids)
        return self.random_float_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)


class DummyLabelsGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "labels",
        "start_positions",
        "end_positions",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task

        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size

        self.sequence_length = kwargs.get("sequence_length", None)
        self.num_labels = kwargs.get("num_labels", None)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        max_value = self.num_labels if self.num_labels is not None else 0
        if self.sequence_length is None:
            shape = [self.batch_size]
        else:
            shape = [self.batch_size, self.sequence_length]

        return self.random_int_tensor(shape, max_value=max_value, framework=framework, dtype=int_dtype)


class DummyPointsGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """

    SUPPORTED_INPUT_NAMES = ("input_points",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        point_batch_size: int = DEFAULT_DUMMY_SHAPES["point_batch_size"],
        nb_points_per_image: int = DEFAULT_DUMMY_SHAPES["nb_points_per_image"],
        **kwargs,
    ):
        self.task = task

        self.batch_size = batch_size
        self.point_batch_size = point_batch_size
        self.nb_points_per_image = nb_points_per_image

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image, 2]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class DummyVisionEmbeddingsGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("image_positional_embeddings", "image_embeddings")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        image_embedding_size: Optional[int] = None,
        output_channels: Optional[int] = None,
        **kwargs,
    ):
        self.task = task

        self.batch_size = batch_size
        self.image_embedding_size = (
            image_embedding_size
            if image_embedding_size is not None
            else normalized_config.prompt_encoder_config.image_embedding_size
        )
        self.output_channels = (
            output_channels if output_channels is not None else normalized_config.vision_config.output_channels
        )

    def generate(self, input_name: str, framework: str = "pt"):
        shape = [self.batch_size, self.output_channels, self.image_embedding_size, self.image_embedding_size]
        return self.random_float_tensor(shape, framework=framework)


class DummyPix2StructInputGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """

    SUPPORTED_INPUT_NAMES = ("flattened_patches",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        preprocessors: List[Any],
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        **kwargs,
    ):
        self.task = task

        self.batch_size = batch_size

        # looking for static shapes in Pix2StructProcessor
        patch_height = preprocessors[1].image_processor.patch_size["height"]
        patch_width = preprocessors[1].image_processor.patch_size["width"]
        self.flattened_patch_size = 2 + patch_height * patch_width * num_channels
        self.max_patches = preprocessors[1].image_processor.max_patches

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size, self.max_patches, self.flattened_patch_size]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class GPTBigCodeDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_value_shape = (
            self.batch_size,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads * 2,
        )
        return [
            self.random_float_tensor(past_key_value_shape, framework=framework, dtype=float_dtype)
            for _ in range(self.num_layers)
        ]


class BloomDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (
            self.batch_size * self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.sequence_length,
        )
        past_value_shape = (
            self.batch_size * self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class FalconDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        self.num_kv_heads = 1
        head_dim = self.hidden_size // self.num_attention_heads

        past_key_shape = (
            self.batch_size,
            self.num_kv_heads,
            self.sequence_length,
            head_dim,
        )
        past_value_shape = (
            self.batch_size,
            self.num_kv_heads,
            self.sequence_length,
            head_dim,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]
