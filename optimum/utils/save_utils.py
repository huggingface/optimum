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

from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


logger = logging.getLogger(__name__)


def maybe_load_preprocessors(src_name_or_path: Union[str, Path], subfolder: str = "") -> List:
    preprocessors = []
    try:
        preprocessors.append(AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder))
    except Exception:
        pass

    try:
        preprocessors.append(AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder))
    except Exception:
        pass

    try:
        preprocessors.append(AutoFeatureExtractor.from_pretrained(src_name_or_path, subfolder=subfolder))
    except Exception:
        pass
    return preprocessors


def maybe_save_preprocessors(src_name_or_path: Union[str, Path], dest_dir: Union[str, Path], src_subfolder: str = ""):
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
    """
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)

    dest_dir.mkdir(exist_ok=True)
    for preprocessor in maybe_load_preprocessors(src_name_or_path, subfolder=src_subfolder):
        preprocessor.save_pretrained(dest_dir)
