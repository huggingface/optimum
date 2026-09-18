# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import optimum
from optimum.version import __version__ as package_version


class VersionTester(unittest.TestCase):
    def test_dunder_version_matches_package(self):
        self.assertEqual(optimum.__version__, package_version)

    def test_dunder_version_is_nonempty_string(self):
        self.assertIsInstance(optimum.__version__, str)
        self.assertTrue(optimum.__version__)

    def test_package_path_is_searchable(self):
        self.assertTrue(hasattr(optimum, "__path__"))
        self.assertGreater(len(list(optimum.__path__)), 0)
