# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
from typing import List, Optional

from transformers import is_tf_available, is_torch_available

from parameterized import parameterized


if is_torch_available() or is_tf_available():
    from transformers.onnx.features import FeaturesManager

from transformers.onnx import OnnxConfig
from transformers.testing_utils import require_torch, slow

from optimum.onnx.auto.configuration_onnx_auto import AutoOnnxConfig


PYTORCH_MODELS = {
    ("albert", "hf-internal-testing/tiny-albert"),
    ("bert", "bert-base-cased"),
    ("ibert", "kssteven/ibert-roberta-base"),
    ("camembert", "camembert-base"),
    ("distilbert", "distilbert-base-cased"),
    ("roberta", "roberta-base"),
    ("xlm-roberta", "xlm-roberta-base"),
    ("layoutlm", "microsoft/layoutlm-base-uncased"),
}

PYTORCH_WITH_PAST_MODELS = {
    ("gpt2", "gpt2"),
    ("gpt-neo", "EleutherAI/gpt-neo-125M"),
}

PYTORCH_SEQ2SEQ_WITH_PAST_MODELS = {
    ("bart", "facebook/bart-base"),
    ("mbart", "sshleifer/tiny-mbart"),
    ("t5", "t5-small"),
    ("marian", "Helsinki-NLP/opus-mt-en-de"),
}

PYTORCH_UNSUPPORTED_MODELS = {
    ("deberta", "microsoft/deberta-base", "sequence-classification"),
    ("bert", "bert-base-cased", "unk-task"),
    ("operta", "optimum/operta-base", "default"),
}


def _get_models_to_test(models_list, excluded: Optional[List[str]] = None):
    models_to_test = []
    if is_torch_available() or is_tf_available():
        for (name, model) in models_list:
            for feature, _ in FeaturesManager.get_supported_features_for_model_type(name).items():
                if excluded and any(key in excluded for key in [name, feature]):
                    continue
                models_to_test.append((f"{name}_{feature}", model, feature))
        return sorted(models_to_test)


def _get_invalid_models_to_test(models_list):
    models_to_test = []
    if is_torch_available() or is_tf_available():
        for (name, model, feature) in models_list:
            models_to_test.append((f"{name}_{feature}", model, feature))
        return sorted(models_to_test)


class AutoOnnxConfigTest(unittest.TestCase):
    @parameterized.expand(
        _get_models_to_test(PYTORCH_MODELS, excluded=["next-sentence-prediction"]), skip_on_empty=True
    )
    @require_torch
    def test_config_from_supported_model(self, test_name, model_name, feature):
        config = AutoOnnxConfig.from_pretrained(model_name, task=feature)
        self.assertIsInstance(config, OnnxConfig)

    @parameterized.expand(_get_invalid_models_to_test(PYTORCH_UNSUPPORTED_MODELS))
    @require_torch
    @unittest.skip("Skip intended to fail tests.")
    def test_config_from_unsupported_model(self, test_name, model_name, feature):
        config = AutoOnnxConfig.from_pretrained(model_name, task=feature)
        self.assertIsInstance(config, OnnxConfig)


if __name__ == "__main__":
    unittest.main()
