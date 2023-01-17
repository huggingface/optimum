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

import re
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfApi, HfFolder, get_hf_file_metadata, hf_hub_url


def validate_file_exists(
    model_name_or_path: Union[str, Path], filename: str, subfolder: str = "", revision: Optional[str] = None
) -> bool:
    """
    Checks that the file called `filename` exists in the `model_name_or_path` directory or model repo.
    """
    model_path = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    if model_path.is_dir():
        return (model_path / subfolder / filename).is_file()
    succeeded = True
    try:
        get_hf_file_metadata(hf_hub_url(model_name_or_path, filename, subfolder=subfolder, revision=revision))
    except Exception:
        succeeded = False
    return succeeded


def find_files_matching_pattern(
    model_name_or_path: Union[str, Path],
    pattern: str,
    glob_pattern: str = "**/*",
    subfolder: str = "",
    use_auth_token: Optional[Union[bool, str]] = None,
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
        use_auth_token (`Optional[bool, str]`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`Optional[str]`, defaults to `None`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.

    Returns:
        `List[Path]`
    """
    model_path = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    pattern = re.compile(f"{subfolder}/{pattern}" if subfolder != "" else pattern)
    if model_path.is_dir():
        path = model_path
        files = model_path.glob(glob_pattern)
        files = [p for p in files if re.search(pattern, str(p))]
    else:
        path = model_name_or_path
        if isinstance(use_auth_token, bool):
            token = HfFolder().get_token()
        else:
            token = use_auth_token
        repo_files = map(Path, HfApi().list_repo_files(model_name_or_path, revision=revision, token=token))
        if subfolder != "":
            path = f"{path}/{subfolder}"
        files = [Path(p) for p in repo_files if re.match(pattern, str(p))]

    return files
