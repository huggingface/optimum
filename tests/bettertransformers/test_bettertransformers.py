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
import unittest

import transformers
from transformers import AutoModel, AutoModelForCausalLM

from optimum.bettertransformer import FAST_LAYERS_MAPPING_DICT, BetterTransformer, convert_to_hf_classes
from optimum.utils.testing_utils import is_accelerate_available


if is_accelerate_available():
    from accelerate import init_empty_weights


ALL_MODEL_CLASSES = (
    AutoModel,
    AutoModelForCausalLM,
)


class BetterTransformersTest(unittest.TestCase):
    r"""
    Full testing suite of the `BetterTransformers` integration into Hugging Face
    `transformers` ecosystem.
    """

    def test_dict_consistency(self):
        r"""
        A test to check if the modified dictionnary is consistent (same number of keys + successfully import
        the correct `PreTrainedModel` module).
        """
        for keys in FAST_LAYERS_MAPPING_DICT.keys():
            self.assertTrue("Layer" in keys)

        ALL_SUPPORTED_HF_CLASSES = convert_to_hf_classes(FAST_LAYERS_MAPPING_DICT)
        self.assertEqual(len(ALL_SUPPORTED_HF_CLASSES.keys()), len(FAST_LAYERS_MAPPING_DICT.keys()))

    @unittest.skipIf(not is_accelerate_available(), "Skipping the test since accelerate is not available...")
    @init_empty_weights()
    def test_conversion(self):
        r"""
        This tests if the conversion of a slow model to its `Fast` version
        has been successfull.
        """
        # Step 0: for each model_class
        # Step 1: convert the model
        # Step 2: check if the conversion if successfull
        # Step 3: check also that the class attributes still remains in the model
        # (for eg, `generate`)
        ALL_SUPPORTED_HF_CLASSES = convert_to_hf_classes(FAST_LAYERS_MAPPING_DICT)

        for layer_class in FAST_LAYERS_MAPPING_DICT.keys():
            random_config = getattr(transformers, layer_class[:-5] + "Config")

            hf_random_model = AutoModel.from_config(random_config())
            converted_model = BetterTransformer.transform(hf_random_model)

    def test_logits(self):
        r"""
        This tests if the converted model produces the same logits
        than the original model.
        """
        pass

    def test_inference_speed(self):
        r"""
        The converted models should be at least slightly faster than the native
        model. This test aims to check this.
        """
        pass

    def test_class_functions(self):
        r"""
        This test runs class functions such as `generate` and checks if the
        function works as expected.
        """
        pass

    def test_accelerate_compatibility(self):
        r"""
        This tests if a model loaded with `accelerate` will be successfully converted
        into its BetterTransformers format.
        """
        pass
