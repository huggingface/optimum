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
"""Import utilities."""

import packaging
from transformers.utils import is_torch_available


MIN_TORCH_VERSION = packaging.version.parse("1.11.0")
TORCH_VERSION = None
if is_torch_available():
    import torch

    TORCH_VERSION = packaging.version.parse(torch.__version__)

_is_torch_onnx_support_available = is_torch_available() and (MIN_TORCH_VERSION.major, MIN_TORCH_VERSION.minor) <= (
    TORCH_VERSION.major,
    TORCH_VERSION.minor,
)


def is_torch_onnx_support_available():
    return _is_torch_onnx_support_available


# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = packaging.version.parse("1.4.0")
