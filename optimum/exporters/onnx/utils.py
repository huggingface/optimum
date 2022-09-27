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
"""Utility functions."""

import packaging
from ctypes import c_float, sizeof
from enum import Enum

from transformers.utils import is_torch_available

MIN_TORCH_VERSION = packaging.version.parse("1.8.0")
TORCH_VERSION = None
if is_torch_available():
    import torch
    TORCH_VERSION = packaging.version.parse(torch.__version__)

_is_torch_onnx_support_available = is_torch_available() and (MIN_TORCH_VERSION.major, MIN_TORCH_VERSION.minor) <= (TORCH_VERSION.major, TORCH_VERSION.minor)


def is_torch_onnx_support_available():
    return _is_torch_onnx_support_available


# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = packaging.version.parse("1.4.0")


def check_onnxruntime_requirements(minimum_version: packaging.version.Version):
    """
    Checks that ONNX Runtime is installed and if version is recent enough.

    Args:
        minimum_version (`packaging.version.Version`):
            The minimum version allowed for the onnxruntime package.

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "ONNX Runtime doesn't seem to be currently installed. "
            "Please install ONNX Runtime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )

    ort_version = packaging.version.parse(onnxruntime.__version__)
    if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
        raise ImportError(
            f"We found an older version of ONNX Runtime ({onnxruntime.__version__}) "
            f"but we require the version to be >= {minimum_version} to enable all the conversions options.\n"
            "Please update ONNX Runtime by running `pip install --upgrade onnxruntime`"
        )


class ParameterFormat(Enum):
    Float = c_float

    @property
    def size(self) -> int:
        """
        Number of byte required for this data type

        Returns:
            Integer > 0
        """
        return sizeof(self.value)


def compute_serialized_parameters_size(num_parameters: int, dtype: ParameterFormat) -> int:
    """
    Compute the size taken by all the parameters in the given the storage format when serializing the model

    Args:
        num_parameters: Number of parameters to be saved
        dtype: The data format each parameter will be saved

    Returns:
        Size (in byte) taken to save all the parameters
    """
    return num_parameters * dtype.size
