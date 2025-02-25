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

from ..utils import is_diffusers_version, is_tf_available, is_torch_available, is_transformers_version
from .normalized_config import (
    NormalizedConfig,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf  # type: ignore


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
    def random_mask_tensor(shape: List[int], padding_side: str = "right", framework: str = "pt", dtype: str = "int64"):
        """
        Generates a mask tensor either right or left padded.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            padding_side (`str`, defaults to "right"):
                The side on which the padding is applied.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"int64"`):
                The dtype of the generated integer tensor. Could be "int64", "int32", "int8".

        Returns:
            A random mask tensor either left padded or right padded in the requested framework.
        """
        shape = tuple(shape)
        mask_length = random.randint(1, shape[-1] - 1)
        if framework == "pt":
            mask_tensor = torch.cat(
                [
                    torch.ones(*shape[:-1], shape[-1] - mask_length, dtype=DTYPE_MAPPER.pt(dtype)),
                    torch.zeros(*shape[:-1], mask_length, dtype=DTYPE_MAPPER.pt(dtype)),
                ],
                dim=-1,
            )
            if padding_side == "left":
                mask_tensor = torch.flip(mask_tensor, [-1])
        elif framework == "tf":
            mask_tensor = tf.concat(
                [
                    tf.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.tf(dtype)),
                    tf.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.tf(dtype)),
                ],
                axis=-1,
            )
            if padding_side == "left":
                mask_tensor = tf.reverse(mask_tensor, [-1])
        else:
            mask_tensor = np.concatenate(
                [
                    np.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.np(dtype)),
                    np.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.np(dtype)),
                ],
                axis=-1,
            )
            if padding_side == "left":
                mask_tensor = np.flip(mask_tensor, [-1])
        return mask_tensor

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
        "encoder_attention_mask",
        "global_attention_mask",
        "token_type_ids",
        "position_ids",
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
        padding_side: str = "right",
        **kwargs,
    ):
        self.task = task

        if isinstance(normalized_config, NormalizedEncoderDecoderConfig):
            self.vocab_size = normalized_config.vocab_size
        else:
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
        self.padding_side = padding_side
        self.normalized_config = normalized_config

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        min_value = 0
        max_value = 2 if input_name != "input_ids" else self.vocab_size

        if self.task == "multiple-choice":
            shape = [self.batch_size, self.num_choices, self.sequence_length]
        else:
            shape = [self.batch_size, self.sequence_length]

        if input_name in ["attention_mask", "encoder_attention_mask"]:
            return self.random_mask_tensor(shape, padding_side=self.padding_side, framework=framework, dtype=int_dtype)
        else:
            return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class LongformerDummyTextInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "global_attention_mask",
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "global_attention_mask":
            attention_mask = super().generate(
                "attention_mask", framework=framework, int_dtype=int_dtype, float_dtype=float_dtype
            )

            if framework == "pt":
                global_attention_mask = torch.zeros_like(attention_mask)
            elif framework == "tf":
                global_attention_mask = tf.zeros_like(attention_mask)
            else:
                global_attention_mask = np.zeros_like(attention_mask)

            return global_attention_mask

        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)


class DummyXPathSeqInputGenerator(DummyTextInputGenerator):
    """
    Generates dummy xpath sequences.
    """

    SUPPORTED_INPUT_NAMES = (
        "xpath_tags_seq",
        "xpath_subs_seq",
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
        padding_side: str = "right",
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
            padding_side=padding_side,
            **kwargs,
        )
        self.max_depth = normalized_config.max_depth
        self.tag_pad_id = normalized_config.tag_pad_id
        self.subs_pad_id = normalized_config.subs_pad_id

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        min_value = 0
        max_value = self.tag_pad_id if input_name == "xpath_tags_seq" else self.subs_pad_id
        shape = [self.batch_size, self.sequence_length, self.max_depth]
        return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class DummyDecoderTextInputGenerator(DummyTextInputGenerator):
    """
    Generates dummy decoder text inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
    )


class DummyDecisionTransformerInputGenerator(DummyTextInputGenerator):
    """
    Generates dummy decision transformer inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "states",
        "actions",
        "timesteps",
        "returns_to_go",
        "attention_mask",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_dim = self.normalized_config.config.act_dim
        self.state_dim = self.normalized_config.config.state_dim
        self.max_ep_len = self.normalized_config.config.max_ep_len

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "states":
            shape = [self.batch_size, self.sequence_length, self.state_dim]
        elif input_name == "actions":
            shape = [self.batch_size, self.sequence_length, self.act_dim]
        elif input_name == "rewards":
            shape = [self.batch_size, self.sequence_length, 1]
        elif input_name == "returns_to_go":
            shape = [self.batch_size, self.sequence_length, 1]
        elif input_name == "attention_mask":
            shape = [self.batch_size, self.sequence_length]
        elif input_name == "timesteps":
            shape = [self.batch_size, self.sequence_length]
            return self.random_int_tensor(shape=shape, max_value=self.max_ep_len, framework=framework, dtype=int_dtype)

        return self.random_float_tensor(shape, min_value=-2.0, max_value=2.0, framework=framework, dtype=float_dtype)


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

        if isinstance(normalized_config, NormalizedEncoderDecoderConfig):
            self.hidden_size = normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS.hidden_size
        else:
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

    SUPPORTED_INPUT_NAMES = ("past_key_values", "cache_position")

    def __init__(
        self,
        task: str,
        normalized_config: Union[NormalizedSeq2SeqConfig, NormalizedEncoderDecoderConfig],
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

        if isinstance(normalized_config, NormalizedEncoderDecoderConfig):
            # encoder_num_attention_heads / decoder_num_attention_heads are bad names, they rather refer to cross / self attention num heads.
            self.encoder_num_attention_heads = (
                self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.encoder_num_attention_heads
            )
            self.decoder_num_attention_heads = (
                self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_attention_heads
            )
            # Same, `encoder_hidden_size` and `decoder_hidden_size` are bad names.
            self.encoder_hidden_size = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.hidden_size
            self.decoder_hidden_size = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.hidden_size
            self.decoder_num_layers = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_layers
        else:
            self.encoder_num_attention_heads = self.normalized_config.encoder_num_attention_heads
            self.decoder_num_attention_heads = self.normalized_config.decoder_num_attention_heads
            self.encoder_hidden_size = self.normalized_config.hidden_size
            self.decoder_hidden_size = self.normalized_config.hidden_size
            self.decoder_num_layers = self.normalized_config.decoder_num_layers

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "past_key_values":
            encoder_shape = (
                self.batch_size,
                self.encoder_num_attention_heads,
                self.encoder_sequence_length,
                self.encoder_hidden_size // self.encoder_num_attention_heads,
            )
            decoder_shape = (
                self.batch_size,
                self.decoder_num_attention_heads,
                self.sequence_length,
                self.decoder_hidden_size // self.decoder_num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.decoder_num_layers)
            ]

        elif input_name == "cache_position":
            return self.random_int_tensor(
                shape=[1],
                max_value=self.sequence_length,
                framework=framework,
                dtype=int_dtype,
            )

        raise ValueError(f"Unsupported input name {input_name}")


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
        if normalized_config.has_attribute("num_channels"):
            self.num_channels = normalized_config.num_channels
        else:
            self.num_channels = num_channels

        if normalized_config.has_attribute("image_size"):
            self.image_size = normalized_config.image_size
        elif normalized_config.has_attribute("input_size"):
            input_size = normalized_config.input_size
            self.num_channels = input_size[0]
            self.image_size = input_size[1:]
        else:
            self.image_size = (height, width)

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

        if hasattr(self.normalized_config, "feature_size"):
            self.feature_size = self.normalized_config.feature_size
        else:
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
        "timestep_cond",
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
        self.text_encoder_projection_dim = getattr(normalized_config, "text_encoder_projection_dim", None)
        self.time_ids = 5 if getattr(normalized_config, "requires_aesthetics_score", False) else 6
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        self.time_cond_proj_dim = getattr(normalized_config.config, "time_cond_proj_dim", None)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "timestep":
            shape = []  # a scalar with no dimension (it can be int or float depending on the sd architecture)
            return self.random_float_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)

        if input_name == "text_embeds":
            if self.text_encoder_projection_dim is None:
                raise ValueError(
                    "Unable to infer the value of `text_encoder_projection_dim` for generating `text_embeds`, please double check the config of your model."
                )
            dim = self.text_encoder_projection_dim
        elif input_name == "timestep_cond":
            if self.time_cond_proj_dim is None:
                raise ValueError(
                    "Unable to infer the value of `time_cond_proj_dim` for generating `timestep_cond`, please double check the config of your model."
                )
            dim = self.time_cond_proj_dim
        else:
            dim = self.time_ids

        shape = [self.batch_size, dim]
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

    SUPPORTED_INPUT_NAMES = ("input_points", "input_labels")

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
        if input_name == "input_points":
            shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image, 2]
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        else:  # input_labels
            shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image]
            return self.random_int_tensor(shape, min_value=0, max_value=1, framework=framework, dtype=int_dtype)


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

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
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
            self.hidden_size // self.num_attention_heads * 2,  # GPT BigCode has a fused KV cache.
        )
        return [
            self.random_float_tensor(past_key_value_shape, framework=framework, dtype=float_dtype)
            for _ in range(self.num_layers)
        ]


class BloomDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if is_transformers_version(">=", "4.44"):
            return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)
        else:
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


class MultiQueryPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            **kwargs,
        )
        self.num_kv_heads = normalized_config.num_kv_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_shape = (
            self.batch_size * self.num_kv_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class FalconDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            **kwargs,
        )
        self.num_kv_heads = self.num_kv_heads = (
            normalized_config.num_kv_heads
            if (normalized_config.new_decoder_architecture or not normalized_config.multi_query)
            else 1
        )
        self.head_dim = self.hidden_size // self.num_attention_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (
            self.batch_size,
            self.num_kv_heads,
            self.sequence_length,
            self.head_dim,
        )
        past_value_shape = (
            self.batch_size,
            self.num_kv_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class MistralDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.num_key_value_heads = normalized_config.num_key_value_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
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


class GemmaDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.num_key_value_heads = normalized_config.num_key_value_heads
        self.head_dim = normalized_config.head_dim

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class DummySpeechT5InputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("output_sequence", "speaker_embeddings", "spectrogram")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = 1  # TODO: SpeechT5 does not support batch inference in Transformers for now.

        self.sequence_length = sequence_length
        self.speaker_embedding_dim = normalized_config.speaker_embedding_dim
        self.num_mel_bins = normalized_config.num_mel_bins

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "output_sequence":
            shape = [self.batch_size, self.sequence_length, self.num_mel_bins]
        elif input_name == "speaker_embeddings":
            shape = [self.batch_size, self.speaker_embedding_dim]
        elif input_name == "spectrogram":
            shape = [20, self.num_mel_bins]  # NOTE: the first axis length is arbitrary and dynamic
        else:
            raise ValueError(f"Unsupported input {input_name} for DummySpeechT5InputGenerator")

        return self.random_float_tensor(
            shape=shape,
            min_value=0,
            max_value=1,
            framework=framework,
            dtype=float_dtype,
        )


class DummyVisionEncoderDecoderPastKeyValuesGenerator(DummySeq2SeqPastKeyValuesGenerator):
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
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            encoder_sequence_length=encoder_sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            **kwargs,
        )
        if normalized_config.model_type == "trocr":
            image_size = normalized_config.encoder.image_size
            patch_size = normalized_config.encoder.patch_size
            self.encoder_sequence_length = (image_size // patch_size) ** 2 + 1

        if isinstance(normalized_config.DECODER_NORMALIZED_CONFIG_CLASS, NormalizedSeq2SeqConfig):
            # Here, the decoder used in the vision-encoder-decoder comes from a seq2seq model.
            self.num_layers = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_layers
            self.use_cross_attention = True
        else:
            self.num_layers = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_layers
            self.use_cross_attention = False

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        decoder_hidden_size = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.hidden_size
        decoder_num_attention_heads = self.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_attention_heads
        decoder_shape = (
            self.batch_size,
            decoder_num_attention_heads,
            self.sequence_length,
            decoder_hidden_size // decoder_num_attention_heads,
        )

        if not self.use_cross_attention:
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]
        else:
            encoder_hidden_size = decoder_hidden_size
            encoder_num_attention_heads = decoder_num_attention_heads

            encoder_shape = (
                self.batch_size,
                encoder_num_attention_heads,
                self.encoder_sequence_length,
                encoder_hidden_size // encoder_num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]


class DummyCodegenDecoderTextInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
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
        self.num_codebooks = normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_codebooks

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["decoder_input_ids"]:
            min_value = 0
            max_value = 2 if input_name != "input_ids" else self.vocab_size
            shape = [self.batch_size * self.num_codebooks, self.sequence_length]
            return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)

        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)


class DummyEncodecInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("audio_codes",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size

        self.num_codebooks = normalized_config.decoder.num_codebooks
        self.sequence_length = sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "audio_codes":
            # Kind of a hack to use `self.sequence_length` here, for Musicgen pad tokens are filtered out, see
            # https://github.com/huggingface/transformers/blob/31c575bcf13c2b85b65d652dd1b5b401f99be999/src/transformers/models/musicgen/modeling_musicgen.py#L2458
            shape = [1, self.batch_size, self.num_codebooks, self.sequence_length]
        else:
            raise ValueError(f"Unsupported input {input_name} for DummyEncodecInputGenerator")

        return self.random_int_tensor(
            shape=shape,
            min_value=0,
            max_value=50,
            framework=framework,
            dtype=int_dtype,
        )


class DummyIntGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "pad_token_id",
        "max_length",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        **kwargs,
    ):
        pass

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        return self.random_int_tensor(shape=(1,), min_value=20, max_value=22, framework=framework, dtype=int_dtype)


class DummyTransformerTimestepInputGenerator(DummyTimestepInputGenerator):
    SUPPORTED_INPUT_NAMES = ("timestep",)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "timestep":
            shape = [self.batch_size]  # With transformer diffusers, timestep is a 1D tensor
            return self.random_float_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)

        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyTransformerVisionInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("hidden_states",)


class DummyTransformerTextInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "encoder_hidden_states",
        "pooled_projection",
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "encoder_hidden_states":
            return super().generate(input_name, framework, int_dtype, float_dtype)[0]

        elif input_name == "pooled_projections":
            return self.random_float_tensor(
                [self.batch_size, self.normalized_config.projection_size], framework=framework, dtype=float_dtype
            )

        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyFluxTransformerVisionInputGenerator(DummyTransformerVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "hidden_states",
        "img_ids",
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "hidden_states":
            shape = [self.batch_size, (self.height // 2) * (self.width // 2), self.num_channels]
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        elif input_name == "img_ids":
            shape = (
                [(self.height // 2) * (self.width // 2), 3]
                if is_diffusers_version(">=", "0.31.0")
                else [self.batch_size, (self.height // 2) * (self.width // 2), 3]
            )
            return self.random_int_tensor(shape, max_value=1, framework=framework, dtype=int_dtype)

        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyFluxTransformerTextInputGenerator(DummyTransformerTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "encoder_hidden_states",
        "pooled_projections",
        "guidance",
        "txt_ids",
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "txt_ids":
            shape = (
                [self.sequence_length, 3]
                if is_diffusers_version(">=", "0.31.0")
                else [self.batch_size, self.sequence_length, 3]
            )
            return self.random_int_tensor(shape, max_value=1, framework=framework, dtype=int_dtype)
        elif input_name == "guidance":
            shape = [self.batch_size]
            return self.random_float_tensor(shape, min_value=0, max_value=1, framework=framework, dtype=float_dtype)

        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyPatchTSTInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("past_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        **kwargs,
    ):
        self.task = task
        self.normalized_config = normalized_config

        self.batch_size = batch_size
        self.context_length = normalized_config.context_length
        self.num_input_channels = normalized_config.num_input_channels

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        return self.random_float_tensor(
            shape=[self.batch_size, self.context_length, self.num_input_channels],
            min_value=-1,
            max_value=1,
            framework=framework,
            dtype=float_dtype,
        )


class MCTCTDummyAudioInputGenerator(DummyAudioInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "input_features":
            shape = [self.batch_size, self.sequence_length, self.normalized_config.input_features_per_channel]
            return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework, dtype=float_dtype)

        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)


class Dinov2DummyInputGenerator(DummyVisionInputGenerator):
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
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            num_channels=num_channels,
            width=width,
            height=height,
            **kwargs,
        )

        from transformers.onnx.utils import get_preprocessor

        preprocessor = get_preprocessor(normalized_config._name_or_path)
        if preprocessor is not None and hasattr(preprocessor, "crop_size"):
            self.height = preprocessor.crop_size.get("height", self.height)
            self.width = preprocessor.crop_size.get("width", self.width)


class DummyVisionStaticInputGenerator(DummyVisionInputGenerator):
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
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            num_channels=num_channels,
            width=width,
            height=height,
            **kwargs,
        )

        from transformers.onnx.utils import get_preprocessor

        preprocessor = get_preprocessor(normalized_config._name_or_path)
        if preprocessor is not None and hasattr(preprocessor, "size"):
            self.height = preprocessor.size.get("height", self.height)
            self.width = preprocessor.size.get("width", self.width)


class PerceiverDummyInputGenerator(DummyVisionStaticInputGenerator):
    pass


class VitPoseDummyInputGenerator(DummyVisionStaticInputGenerator):
    pass
