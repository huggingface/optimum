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
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_from_pretrained_selects_latest_compatible_configuration(self):
        configuration_files = ["fake_config1.9.0.json", "fake_config1.10.0.json", "fake_config99.0.0.json"]
        with tempfile.TemporaryDirectory() as tmpdirname:
            directory = Path(tmpdirname)
            (directory / FakeConfig.CONFIG_NAME).write_text(
                json.dumps({"configuration_files": configuration_files, "attribute": 0}), encoding="utf-8"
            )
            for attribute, filename in enumerate(configuration_files, start=9):
                (directory / filename).write_text(json.dumps({"attribute": attribute}), encoding="utf-8")

            for optimum_version, expected_attribute in [("1.9.0", 9), ("1.10.0", 10)]:
                with self.subTest(optimum_version=optimum_version):
                    with patch("optimum.configuration_utils.__version__", optimum_version):
                        config = FakeConfig.from_pretrained(tmpdirname, local_files_only=True)
                    self.assertEqual(config.attribute, expected_attribute)


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
