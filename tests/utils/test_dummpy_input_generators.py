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

from contextlib import nullcontext
from typing import TYPE_CHECKING, Tuple
from unittest import TestCase

import torch
from parameterized import parameterized
from transformers import AutoConfig
from transformers.utils import is_tf_available

from optimum.utils import DTYPE_MAPPER, DummyAudioInputGenerator, DummyTextInputGenerator, DummyVisionInputGenerator
from optimum.utils.normalized_config import NormalizedConfigManager
from optimum.utils.testing_utils import grid_parameters


if TYPE_CHECKING:
    from optimum.utils.input_generators import DummyInputGenerator


TEXT_ENCODER_MODELS = {"distilbert": "hf-internal-testing/tiny-random-DistilBertModel"}

VISION_MODELS = {"resnet": "hf-internal-testing/tiny-random-resnet"}

SEQ2SEQ_MODELS = {"t5": "hf-internal-testing/tiny-random-T5Model"}

AUDIO_MODELS = {"whisper": "optimum-internal-testing/tiny-random-whisper"}

DUMMY_SHAPES = {
    "batch_size": [2, 4],
    "sequence_length": [16, 18],
    "num_choices": [4, 6],
    # image
    "num_channels": [3, 1],
    "width": [224, 150],
    "height": [224, 150],
    # audio
    "feature_size": [40, 10],
    "nb_max_frames": [300, 500],
    "audio_sequence_length": [16000, 8000],
}


class GenerateDummy(TestCase):
    _FRAMEWORK_TO_SHAPE_CLS = {
        "pt": torch.Size,
        "np": tuple,
    }
    if is_tf_available():
        import tensorflow as tf  # type: ignore[import]

        _FRAMEWORK_TO_SHAPE_CLS["tf"] = tf.TensorShape

    def validate_shape_for_framework(
        self, generator: "DummyInputGenerator", input_name: str, framework: str, target_shape: Tuple[int, ...]
    ):
        generated_shape = generator.generate(input_name).shape
        target_shape = self._FRAMEWORK_TO_SHAPE_CLS[framework](target_shape)
        if generated_shape != target_shape:
            raise ValueError(
                f"{input_name} shape is wrong for framework = {framework}. Expected {target_shape} but got "
                f"{generated_shape}"
            )

    def validate_dtype_for_framework(
        self, generator: "DummyInputGenerator", input_name: str, framework: str, target_dtype: str
    ):
        if "int" in target_dtype:
            generated_tenor = generator.generate(input_name=input_name, framework=framework, int_dtype=target_dtype)
        else:
            generated_tenor = generator.generate(input_name=input_name, framework=framework, float_dtype=target_dtype)
        dtype_funcs = {"np": DTYPE_MAPPER.np, "pt": DTYPE_MAPPER.pt, "tf": DTYPE_MAPPER.tf}
        target_dtype = dtype_funcs[framework](target_dtype)
        if generated_tenor.dtype != target_dtype:
            raise ValueError(
                f"{input_name} dtype is wrong for framework = {framework}. Expected {target_dtype} but got "
                f"{generated_tenor.dtype}"
            )

    def validate_shape_for_all_frameworks(
        self, generator: "DummyInputGenerator", input_name: str, target_shape: Tuple[int, ...]
    ):
        for framework in self._FRAMEWORK_TO_SHAPE_CLS:
            self.validate_shape_for_framework(generator, input_name, framework, target_shape)

    @parameterized.expand(
        grid_parameters(
            {
                "model_name": VISION_MODELS.values(),
                "framework": ["pt"],
                "int_dtype": ["int64", "int32", "int8"],
                "float_dtype": ["fp32", "fp16", "bf16"],
            }
        )
    )
    def test_generated_tensor_dtype(
        self, test_name: str, model_name: str, framework: str, int_dtype: str, float_dtype: str
    ):
        config = AutoConfig.from_pretrained(model_name)
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)
        input_generator = DummyVisionInputGenerator(
            task="image-classification",
            normalized_config=normalized_config,
            batch_size=2,
            num_channels=3,
            height=224,
            width=224,
        )
        self.validate_dtype_for_framework(input_generator, "pixel_values", framework, float_dtype)
        self.validate_dtype_for_framework(input_generator, "pixel_mask", framework, int_dtype)

    @parameterized.expand(
        grid_parameters(
            {
                "model_name": TEXT_ENCODER_MODELS.values(),
                "batch_size": DUMMY_SHAPES["batch_size"],
                "num_choices": DUMMY_SHAPES["num_choices"],
                "sequence_length": DUMMY_SHAPES["sequence_length"],
            }
        )
    )
    def test_text_models(
        self, test_name: str, model_name: str, batch_size: int, num_choices: int, sequence_length: int
    ):
        # isn't this very verbose?
        config = AutoConfig.from_pretrained(model_name)
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)

        input_generator = DummyTextInputGenerator(
            task="text-classification",
            normalized_config=normalized_config,
            batch_size=batch_size,
            num_choices=num_choices,
            sequence_length=sequence_length,
        )
        self.validate_shape_for_all_frameworks(input_generator, "input_ids", (batch_size, sequence_length))
        self.validate_shape_for_all_frameworks(input_generator, "attention_mask", (batch_size, sequence_length))

        input_generator = DummyTextInputGenerator(
            task="multiple-choice",
            normalized_config=normalized_config,
            batch_size=batch_size,
            num_choices=num_choices,
            sequence_length=sequence_length,
        )
        self.validate_shape_for_all_frameworks(
            input_generator, "input_ids", (batch_size, num_choices, sequence_length)
        )
        self.validate_shape_for_all_frameworks(
            input_generator, "attention_mask", (batch_size, num_choices, sequence_length)
        )

    @parameterized.expand(
        grid_parameters(
            {
                "model_name": VISION_MODELS.values(),
                "batch_size": DUMMY_SHAPES["batch_size"],
                "num_channels": DUMMY_SHAPES["num_channels"],
                "height": DUMMY_SHAPES["height"],
                "width": DUMMY_SHAPES["width"],
            }
        )
    )
    def test_vision_models(
        self, test_name: str, model_name: str, batch_size: int, num_channels: int, height: int, width: int
    ):
        # isn't this very verbose?
        config = AutoConfig.from_pretrained(model_name)
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)

        input_generator = DummyVisionInputGenerator(
            task="image-classification",
            normalized_config=normalized_config,
            batch_size=batch_size,
            num_channels=num_channels,
            height=height,
            width=width,
        )
        with self.assertRaises(ValueError) if num_channels != normalized_config.num_channels else nullcontext():
            self.validate_shape_for_all_frameworks(
                input_generator, "pixel_values", (batch_size, num_channels, height, width)
            )
            self.validate_shape_for_all_frameworks(input_generator, "pixel_mask", (batch_size, height, width))

    @parameterized.expand(
        grid_parameters(
            {
                "model_name": AUDIO_MODELS.values(),
                "batch_size": DUMMY_SHAPES["batch_size"],
                "feature_size": DUMMY_SHAPES["feature_size"],
                "nb_max_frames": DUMMY_SHAPES["nb_max_frames"],
                "audio_sequence_length": DUMMY_SHAPES["audio_sequence_length"],
            }
        )
    )
    def test_audio_models(
        self,
        test_name: str,
        model_name: str,
        batch_size: int,
        feature_size: int,
        nb_max_frames: int,
        audio_sequence_length: int,
    ):
        # isn't this very verbose?
        config = AutoConfig.from_pretrained(model_name)
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)

        input_generator = DummyAudioInputGenerator(
            task="audio-classification",
            normalized_config=normalized_config,
            batch_size=batch_size,
            feature_size=feature_size,
            nb_max_frames=nb_max_frames,
            audio_sequence_length=audio_sequence_length,
        )
        self.validate_shape_for_all_frameworks(input_generator, "input_values", (batch_size, audio_sequence_length))
        self.validate_shape_for_all_frameworks(
            input_generator, "input_feautres", (batch_size, feature_size, nb_max_frames)
        )
