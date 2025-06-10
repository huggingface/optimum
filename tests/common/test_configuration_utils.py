# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
import tempfile
import unittest

from huggingface_hub import login
from transformers.testing_utils import TOKEN, TemporaryHubRepo, is_staging_test

from optimum.configuration_utils import BaseConfig


class FakeConfig(BaseConfig):
    CONFIG_NAME = "fake_config.json"
    FULL_CONFIGURATION_FILE = "fake_config.json"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)


class ConfigTester(unittest.TestCase):
    def test_create_and_test_config_from_and_save_pretrained(self):
        config_first = FakeConfig(attribute=10)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = FakeConfig.from_pretrained(tmpdirname)

        self.assertEqual(config_second.to_dict(), config_first.to_dict())


@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        login(token=TOKEN)

    def test_push_to_hub(self):
        config = FakeConfig(attribute=15)

        with TemporaryHubRepo(token=TOKEN) as tmp_repo:
            config.push_to_hub(tmp_repo.repo_id, token=TOKEN)

            new_config = FakeConfig.from_pretrained(tmp_repo.repo_id, token=TOKEN)
            for k, v in config.to_dict().items():
                if k != "optimum_version" and k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        config = FakeConfig(attribute=15)

        with TemporaryHubRepo(namespace="valid_org", token=TOKEN) as tmp_repo:
            config.push_to_hub(tmp_repo.repo_id, token=TOKEN)
            new_config = FakeConfig.from_pretrained(tmp_repo.repo_id, token=TOKEN)
            for k, v in config.to_dict().items():
                if k != "optimum_version" and k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))
