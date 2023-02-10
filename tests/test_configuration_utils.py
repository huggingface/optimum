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
import os
import tempfile
import unittest

from huggingface_hub import HfFolder, delete_repo, set_access_token
from requests.exceptions import HTTPError
from transformers.testing_utils import TOKEN, USER, is_staging_test

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
        cls._token = TOKEN
        set_access_token(TOKEN)
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="optimum-test-base-config")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/optimum-test-base-config-org")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="optimum-test-base-dynamic-config")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = FakeConfig(attribute=15)
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(
                os.path.join(tmp_dir, "optimum-test-base-config"), push_to_hub=True, use_auth_token=self._token
            )

            new_config = FakeConfig.from_pretrained(f"{USER}/optimum-test-base-config")
            for k, v in config.to_dict().items():
                if k != "optimum_version" and k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        config = FakeConfig(attribute=15)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(
                os.path.join(tmp_dir, "optimum-test-base-config-org"),
                push_to_hub=True,
                use_auth_token=self._token,
                organization="valid_org",
            )

            new_config = FakeConfig.from_pretrained("valid_org/optimum-test-base-config-org")
            for k, v in config.to_dict().items():
                if k != "optimum_version" and k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))
