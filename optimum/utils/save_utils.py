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
from typing import Union
from pathlib import Path

from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor

logger = logging.getLogger(__name__)


def maybe_save_tokenizer_or_processor_or_feature_extractor(src_dir: Union[str, Path], dest_dir: Union[str, Path]):
    """
    Saves the tokenizer, the processor and the feature extractor when found in `src_dir` in `dest_dir`.

    Args:
        src_dir (`Union[str, Path]`):
            The source directory from which to copy the files.
        dest_dir (`Union[str, Path]`):
            The destination directory to copy the files to.
    """
    if not isinstance(src_dir, Path):
        src_dir = Path(src_dir)
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)

    dest_dir.mkdir(exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(src_dir)
        tokenizer.save_pretrained(dest_dir)
    except Exception:
        pass

    try:
        processor = AutoProcessor.from_pretrained(src_dir)
        processor.save_pretrained(dest_dir)
    except Exception:
        pass

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(src_dir)
        feature_extractor.save_pretrained(dest_dir)
    except Exception:
        pass
