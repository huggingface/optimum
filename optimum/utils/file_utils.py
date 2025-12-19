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
"""Utility functions related to both local files and files on the Hugging Face Hub."""

import os
import re
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from transformers.utils import http_user_agent


def validate_file_exists(
    model_name_or_path: Union[str, Path],
    filename: str,
    subfolder: str = "",
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
) -> bool:
    """
    Checks that the file called `filename` exists in the `model_name_or_path` directory or model repo.
    """

    if os.path.isdir(model_name_or_path):
        return os.path.isfile(os.path.join(model_name_or_path, subfolder, filename))
    else:
        return HfApi(user_agent=http_user_agent(), token=token).file_exists(
            filename=os.path.join(subfolder, filename),
            repo_id=model_name_or_path,
            revision=revision,
            token=token,
        )


def find_files_matching_pattern(
    model_name_or_path: Union[str, Path],
    pattern: str,
    glob_pattern: str = "**/*",
    subfolder: str = "",
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
) -> List[Path]:
    """
    Scans either a model repo or a local directory to find filenames matching the pattern.

    Args:
        model_name_or_path (`Union[str, Path]`):
            The name of the model repo on the Hugging Face Hub or the path to a local directory.
        pattern (`str`):
            The pattern to use to look for files.
        glob_pattern (`str`, defaults to `"**/*"`):
            The pattern to use to list all the files that need to be checked.
        subfolder (`str`, defaults to `""`):
            In case the model files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        token (`Optional[Union[bool,str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
        token (`Optional[Union[bool, str]]`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`Optional[str]`, defaults to `None`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.

    Returns:
        `List[Path]`
    """

    model_path = str(model_name_or_path) if isinstance(model_name_or_path, Path) else model_name_or_path
    pattern = re.compile(subfolder + pattern)
    if os.path.isdir(model_path):
        files = Path(model_path).glob(glob_pattern)
        files = [p for p in files if re.search(pattern, str(p))]
    else:
        repo_files = HfApi(user_agent=http_user_agent(), token=token).list_repo_files(
            model_path, revision=revision, token=token
        )
        files = [Path(p) for p in repo_files if re.match(pattern, p)]

    return files


def download_file_with_filename(
    repo_id: str,
    filename: str,
    local_filename: Optional[str] = None,
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
    repo_type: Optional[str] = None,
) -> str:
    """
    Download a specific file from a Hugging Face Hub repository with an optional custom local filename.

    This function is useful when you need to download a file (e.g., an ONNX model) and cache it with a specific
    filename, which is particularly important for repositories like xenova that may have specific naming requirements.

    Args:
        repo_id (`str`):
            The repository ID on the Hugging Face Hub.
        filename (`str`):
            The name of the file to download from the repository.
        local_filename (`Optional[str]`, defaults to `None`):
            The local filename to use when caching the file. If `None`, the original filename is used.
            This parameter allows you to specify a custom name for the cached file.
        subfolder (`str`, defaults to `""`):
            In case the file is located inside a subfolder of the repository, you can specify the subfolder name here.
        revision (`Optional[str]`, defaults to `None`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        cache_dir (`Optional[str]`, defaults to `None`):
            Path to a directory in which a downloaded file should be cached if the standard cache should not be used.
        token (`Optional[Union[bool, str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
        repo_type (`Optional[str]`, defaults to `None`):
            The type of repository. Can be `"model"`, `"dataset"`, or `"space"`. Defaults to `"model"`.

    Returns:
        `str`: The path to the downloaded file.

    Example:
        ```python
        # Download an ONNX file with a custom local filename
        file_path = download_file_with_filename(
            repo_id="xenova/model-name",
            filename="model.onnx",
            local_filename="custom_model.onnx",
            cache_dir="./cache"
        )
        ```
    """
    if repo_type is None:
        repo_type = "model"

    # Construct the full filename path including subfolder
    if subfolder:
        full_filename = f"{subfolder}/{filename}"
    else:
        full_filename = filename

    # Use hf_hub_download which supports local_filename parameter
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=full_filename,
        local_filename=local_filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        repo_type=repo_type,
    )

    return downloaded_path
