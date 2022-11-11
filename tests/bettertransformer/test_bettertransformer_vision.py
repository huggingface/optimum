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
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel, pipeline

import requests
from optimum.bettertransformer import BETTER_TRANFORMER_LAYERS_MAPPING_DICT, BetterTransformer
from optimum.utils import is_accelerate_available, is_datasets_available
from optimum.utils.testing_utils import (
    convert_to_hf_classes,
    is_torch_greater_than_113,
    require_accelerate,
    require_datasets,
    require_torch_gpu,
)


if is_datasets_available():
    from datasets import load_dataset

if is_accelerate_available():
    from accelerate import init_empty_weights

from testing_bettertransformer_utils import BetterTransformersTestMixin


ALL_AUDIO_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-ViTModel",
    "hf-internal-testing/tiny-random-YolosModel",
    "hf-internal-testing/tiny-random-ViTMAEModel",
    "hf-internal-testing/tiny-random-ViTMSNModel",
]


class BetterTransformersVisionTest(BetterTransformersTestMixin, unittest.TestCase):
    r""" """
    all_models_to_test = ALL_AUDIO_MODELS_TO_TEST

    def prepare_inputs_for_class(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # Use the same feature extractor for everyone
        feature_extractor = AutoFeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-ViTModel")
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

    # TODO: add pipeline test
