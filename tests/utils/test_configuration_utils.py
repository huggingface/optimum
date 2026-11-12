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

from unittest import TestCase
from unittest.mock import patch

from optimum.configuration_utils import BaseConfig


class GetConfigurationFileTest(TestCase):
    def test_unversioned_config_json_does_not_raise(self):
        with patch("optimum.configuration_utils.__version__", "2.4.0"):
            self.assertEqual(BaseConfig.get_configuration_file(["config.json"]), "config.json")

    def test_selects_latest_versioned_file_not_newer_than_installed(self):
        files = ["config.json", "config.1.20.0.json", "config.2.1.0.json", "config.2.5.0.json"]
        with patch("optimum.configuration_utils.__version__", "2.4.0"):
            self.assertEqual(BaseConfig.get_configuration_file(files), "config.2.1.0.json")

    def test_ignores_versioned_files_newer_than_installed(self):
        files = ["config.json", "config.99.0.0.json"]
        with patch("optimum.configuration_utils.__version__", "2.4.0"):
            self.assertEqual(BaseConfig.get_configuration_file(files), "config.json")

    def test_accepts_versioned_file_without_unversioned_sibling(self):
        with patch("optimum.configuration_utils.__version__", "2.4.0"):
            self.assertEqual(BaseConfig.get_configuration_file(["config.1.2.3.json"]), "config.1.2.3.json")

    def test_does_not_treat_similar_prefix_as_versioned_config(self):
        files = ["config.json", "myconfig.1.0.0.json"]
        with patch("optimum.configuration_utils.__version__", "2.4.0"):
            self.assertEqual(BaseConfig.get_configuration_file(files), "config.json")

    def test_matches_versioned_file_by_basename(self):
        files = ["config.json", "subdir/config.1.2.3.json"]
        with patch("optimum.configuration_utils.__version__", "2.4.0"):
            self.assertEqual(BaseConfig.get_configuration_file(files), "subdir/config.1.2.3.json")
