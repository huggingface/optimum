# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Tests for file utility functions."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from optimum.utils.file_utils import download_file_with_filename, validate_file_exists


class TestDownloadFileWithFilename:
    """Test cases for download_file_with_filename function."""

    @patch("optimum.utils.file_utils.hf_hub_download")
    def test_download_file_with_default_filename(self, mock_hf_hub_download):
        """Test downloading a file without specifying a custom local filename."""
        mock_hf_hub_download.return_value = "/path/to/cached/model.onnx"

        result = download_file_with_filename(
            repo_id="xenova/test-model",
            filename="model.onnx",
            cache_dir="/tmp/cache",
        )

        assert result == "/path/to/cached/model.onnx"
        mock_hf_hub_download.assert_called_once_with(
            repo_id="xenova/test-model",
            filename="model.onnx",
            local_filename=None,
            revision=None,
            cache_dir="/tmp/cache",
            token=None,
            repo_type="model",
        )

    @patch("optimum.utils.file_utils.hf_hub_download")
    def test_download_file_with_custom_local_filename(self, mock_hf_hub_download):
        """Test downloading a file with a custom local filename."""
        mock_hf_hub_download.return_value = "/path/to/cached/custom_model.onnx"

        result = download_file_with_filename(
            repo_id="xenova/test-model",
            filename="model.onnx",
            local_filename="custom_model.onnx",
            cache_dir="/tmp/cache",
        )

        assert result == "/path/to/cached/custom_model.onnx"
        mock_hf_hub_download.assert_called_once_with(
            repo_id="xenova/test-model",
            filename="model.onnx",
            local_filename="custom_model.onnx",
            revision=None,
            cache_dir="/tmp/cache",
            token=None,
            repo_type="model",
        )

    @patch("optimum.utils.file_utils.hf_hub_download")
    def test_download_file_with_subfolder(self, mock_hf_hub_download):
        """Test downloading a file from a subfolder."""
        mock_hf_hub_download.return_value = "/path/to/cached/model.onnx"

        result = download_file_with_filename(
            repo_id="xenova/test-model",
            filename="model.onnx",
            subfolder="onnx",
            cache_dir="/tmp/cache",
        )

        assert result == "/path/to/cached/model.onnx"
        mock_hf_hub_download.assert_called_once_with(
            repo_id="xenova/test-model",
            filename="onnx/model.onnx",
            local_filename=None,
            revision=None,
            cache_dir="/tmp/cache",
            token=None,
            repo_type="model",
        )

    @patch("optimum.utils.file_utils.hf_hub_download")
    def test_download_file_with_revision_and_token(self, mock_hf_hub_download):
        """Test downloading a file with revision and token parameters."""
        mock_hf_hub_download.return_value = "/path/to/cached/model.onnx"

        result = download_file_with_filename(
            repo_id="xenova/test-model",
            filename="model.onnx",
            revision="main",
            token="test_token",
            cache_dir="/tmp/cache",
        )

        assert result == "/path/to/cached/model.onnx"
        mock_hf_hub_download.assert_called_once_with(
            repo_id="xenova/test-model",
            filename="model.onnx",
            local_filename=None,
            revision="main",
            cache_dir="/tmp/cache",
            token="test_token",
            repo_type="model",
        )

    @patch("optimum.utils.file_utils.hf_hub_download")
    def test_download_file_with_dataset_repo_type(self, mock_hf_hub_download):
        """Test downloading a file from a dataset repository."""
        mock_hf_hub_download.return_value = "/path/to/cached/data.json"

        result = download_file_with_filename(
            repo_id="test/dataset",
            filename="data.json",
            repo_type="dataset",
            cache_dir="/tmp/cache",
        )

        assert result == "/path/to/cached/data.json"
        mock_hf_hub_download.assert_called_once_with(
            repo_id="test/dataset",
            filename="data.json",
            local_filename=None,
            revision=None,
            cache_dir="/tmp/cache",
            token=None,
            repo_type="dataset",
        )


class TestValidateFileExists:
    """Test cases for validate_file_exists function."""

    def test_validate_file_exists_local_directory(self):
        """Test validating file existence in a local directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test_file.txt"
            test_file.write_text("test content")

            # Test with file in root
            assert validate_file_exists(tmpdir, "test_file.txt") is True
            assert validate_file_exists(tmpdir, "nonexistent.txt") is False

            # Test with file in subfolder
            subfolder = Path(tmpdir) / "subfolder"
            subfolder.mkdir()
            subfolder_file = subfolder / "sub_file.txt"
            subfolder_file.write_text("test content")

            assert validate_file_exists(tmpdir, "sub_file.txt", subfolder="subfolder") is True
            assert validate_file_exists(tmpdir, "nonexistent.txt", subfolder="subfolder") is False

    @patch("optimum.utils.file_utils.HfApi")
    def test_validate_file_exists_remote_repo(self, mock_hf_api_class):
        """Test validating file existence in a remote repository."""
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # Test file exists
        mock_api.file_exists.return_value = True
        assert validate_file_exists("test/model", "config.json") is True
        mock_api.file_exists.assert_called_with(
            filename="config.json",
            repo_id="test/model",
            revision=None,
            token=None,
        )

        # Test file doesn't exist
        mock_api.file_exists.return_value = False
        assert validate_file_exists("test/model", "nonexistent.json") is False

        # Test with subfolder
        mock_api.file_exists.return_value = True
        assert validate_file_exists("test/model", "model.onnx", subfolder="onnx") is True
        mock_api.file_exists.assert_called_with(
            filename="onnx/model.onnx",
            repo_id="test/model",
            revision=None,
            token=None,
        )

