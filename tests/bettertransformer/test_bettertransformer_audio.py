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
import gc
import tempfile
import timeit
import unittest

import torch
import transformers
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, pipeline

from optimum.bettertransformer import BETTER_TRANFORMER_LAYERS_MAPPING_DICT, BetterTransformer
from optimum.utils import is_accelerate_available, is_datasets_available
from optimum.utils.testing_utils import require_datasets


if is_datasets_available():
    from datasets import load_dataset

if is_accelerate_available():
    from accelerate import init_empty_weights

from testing_bettertransformer_utils import BetterTransformersTestMixin


ALL_AUDIO_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-WhisperModel",
]


@require_datasets
class BetterTransformersAudioTest(BetterTransformersTestMixin, unittest.TestCase):
    r""" """
    all_models_to_test = ALL_AUDIO_MODELS_TO_TEST

    def prepare_inputs_for_class(self, model_id):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        input_audio = ds[0]["audio"]["array"]

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        input_features = feature_extractor(input_audio, return_tensors="pt").input_features

        input_dict = {"input_features": input_features, "attention_mask": None}
        return input_dict
