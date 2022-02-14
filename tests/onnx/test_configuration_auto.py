# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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

# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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

import os
import tempfile
import unittest
import importlib

from transformers.onnx import OnnxConfig

from optimum.onnx.auto.configuration_onnx_auto import AutoOnnxConfig


SAMPLE_ROBERTA_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/dummy-config.json")


class AutoOnnxConfigTest(unittest.TestCase):
    def test_config_from_model_shortcut(self):
        model_names = {"bert-base-cased", "distilbert-base-uncased", "facebook/bart-base", "gpt2", "roberta-base"}
        task_names = {"default", "masked-lm", "sequence-classification"}

        for model_name in model_names:
            for task_name in task_names:
                with self.subTest(model_name=model_name):
                    config = AutoOnnxConfig.from_pretrained(model_name, task=task_name)
                    self.assertIsInstance(config, OnnxConfig)
                    print(type(config))
                    print(config.outputs)


if __name__ == "__main__":
    unittest.main()
