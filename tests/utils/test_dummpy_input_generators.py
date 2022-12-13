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
from unittest import TestCase

import torch
from transformers import AutoConfig

from optimum.utils import DummyAudioInputGenerator, DummyTextInputGenerator, DummyVisionInputGenerator
from optimum.utils.normalized_config import NormalizedConfigManager
from optimum.utils.testing_utils import grid_parameters
from parameterized import parameterized


TEXT_ENCODER_MODELS = {"distilbert": "distilbert-base-cased"}

VISION_MODELS = {"resnet": "hf-internal-testing/tiny-random-resnet"}

SEQ2SEQ_MODELS = {"t5": "t5-small"}

AUDIO_MODELS = {"whisper": "openai/whisper-tiny.en"}

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
    def test_text_models(self, model_name: str, batch_size: int, num_choices: int, sequence_length: int):
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
        generated_tensor = input_generator.generate("input_ids")
        assert generated_tensor.shape == torch.Size((batch_size, sequence_length))

        generated_tensor = input_generator.generate("attention_mask")
        assert generated_tensor.shape == torch.Size((batch_size, sequence_length))

        input_generator = DummyTextInputGenerator(
            task="multiple-choice",
            normalized_config=normalized_config,
            batch_size=batch_size,
            num_choices=num_choices,
            sequence_length=sequence_length,
        )
        generated_tensor = input_generator.generate("input_ids")
        assert generated_tensor.shape == torch.Size((batch_size, num_choices, sequence_length))

        generated_tensor = input_generator.generate("attention_mask")
        assert generated_tensor.shape == torch.Size((batch_size, num_choices, sequence_length))

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
    def test_vision_models(self, model_name: str, batch_size: int, num_channels: int, height: int, width: int):
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
        with self.assertRaises(AssertionError) if num_channels != normalized_config.num_channels else nullcontext():
            generated_tensor = input_generator.generate("pixel_values")
            assert generated_tensor.shape == torch.Size((batch_size, num_channels, height, width))

            generated_tensor = input_generator.generate("pixel_mask")
            assert generated_tensor.shape == torch.Size((batch_size, height, width))

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
        self, model_name: str, batch_size: int, feature_size: int, nb_max_frames: int, audio_sequence_length: int
    ):
        # isn't this very verbose?
        config = AutoConfig.from_pretrained(model_name)
        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(config.model_type)
        normalized_config = normalized_config_class(config)

        input_generator = DummyAudioInputGenerator(
            task="image-classification",
            normalized_config=normalized_config,
            batch_size=batch_size,
            feature_size=feature_size,
            nb_max_frames=nb_max_frames,
            sequence_length=audio_sequence_length,
        )
        generated_tensor = input_generator.generate("input_values")
        assert generated_tensor.shape == torch.Size((batch_size, audio_sequence_length))

        generated_tensor = input_generator.generate("input_features")
        assert generated_tensor.shape == torch.Size((batch_size, feature_size, nb_max_frames))
