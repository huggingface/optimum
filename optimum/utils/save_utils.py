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
"""Utilities related to saving files."""

import logging
from pathlib import Path
from typing import List, Union

from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer


logger = logging.getLogger(__name__)


def maybe_load_preprocessors(
    src_name_or_path: Union[str, Path], subfolder: str = "", trust_remote_code: bool = False
) -> List:
    """Load all available preprocessors (tokenizer, processor, feature extractor, image processor) from a model path or Hub repo.

    Tries to load each preprocessor type in turn and silently skips any that are
    not present.  This is useful when you want to save preprocessors alongside
    an exported model without knowing in advance which types are available.

    Args:
        src_name_or_path (`Union[str, Path]`): Local path or Hugging Face Hub
            model identifier to load the preprocessors from.
        subfolder (`str`, *optional*, defaults to `""`): Subfolder within the
            model directory or Hub repo where the preprocessor files are stored.
        trust_remote_code (`bool`, *optional*, defaults to `False`): Whether to
            allow running arbitrary remote code when loading preprocessors.

    Returns:
        `List`: A list containing the successfully loaded preprocessor objects.
        May be empty if none of the expected preprocessors are found.

    Example:
        ```py
        >>> preprocessors = maybe_load_preprocessors("bert-base-uncased")
        >>> [type(p).__name__ for p in preprocessors]
        ['BertTokenizerFast']
        ```
    """
    preprocessors = []
    try:
        preprocessors.append(
            AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoFeatureExtractor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoImageProcessor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception:
        pass
    return preprocessors


def maybe_save_preprocessors(
    src_name_or_path: Union[str, Path],
    dest_dir: Union[str, Path],
    src_subfolder: str = "",
    trust_remote_code: bool = False,
):
    """
    Saves the tokenizer, the processor and the feature extractor when found in `src_dir` in `dest_dir`.

    Args:
        src_dir (`Union[str, Path]`):
            The source directory from which to copy the files.
        dest_dir (`Union[str, Path]`):
            The destination directory to copy the files to.
        src_subfolder (`str`, defaults to `""`):
            In case the preprocessor files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        trust_remote_code (`bool`, defaults to `False`):
            Whether to allow to save preprocessors that is allowed to run arbitrary code. Use this option at your own risk.
    """
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)

    dest_dir.mkdir(exist_ok=True)
    for preprocessor in maybe_load_preprocessors(
        src_name_or_path, subfolder=src_subfolder, trust_remote_code=trust_remote_code
    ):
        preprocessor.save_pretrained(dest_dir)
