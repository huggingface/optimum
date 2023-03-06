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

import importlib.util
import inspect
import sys
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import packaging
from transformers.utils import is_torch_available


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


TORCH_MINIMUM_VERSION = packaging.version.parse("1.11.0")


# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = packaging.version.parse("1.4.0")


_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
_pydantic_available = importlib.util.find_spec("pydantic") is not None
_accelerate_available = importlib.util.find_spec("accelerate") is not None
_diffusers_available = importlib.util.find_spec("diffusers") is not None

torch_version = None
if is_torch_available():
    torch_version = packaging.version.parse(importlib_metadata.version("torch"))


_is_torch_onnx_support_available = is_torch_available() and (
    TORCH_MINIMUM_VERSION.major,
    TORCH_MINIMUM_VERSION.minor,
) <= (
    torch_version.major,
    torch_version.minor,
)


def is_torch_onnx_support_available():
    return _is_torch_onnx_support_available


def is_onnxruntime_available():
    try:
        # Try to import the source file of onnxruntime - if you run the tests from `tests` the function gets
        # confused since there a folder named `onnxruntime` in `tests`. Therefore, `_onnxruntime_available`
        # will be set to `True` even if not installed.
        mod = importlib.import_module("onnxruntime")
        inspect.getsourcefile(mod)
    except Exception:
        return False
    return _onnxruntime_available


def is_pydantic_available():
    return _pydantic_available


def is_accelerate_available():
    return _accelerate_available


def is_diffusers_available():
    return _diffusers_available


@contextmanager
def check_if_pytorch_greater(target_version: str, message: str):
    r"""
    A context manager that does nothing except checking if the PyTorch version is greater than `pt_version`
    """
    import torch

    if not packaging.version.parse(torch.__version__) >= packaging.version.parse(target_version):
        raise ImportError(
            f"Found an incompatible version of PyTorch. Found version {torch.__version__}, but only {target_version} and above are supported. {message}"
        )
    try:
        yield
    finally:
        pass


def check_if_transformers_greater(target_version: str) -> bool:
    """
    Checks whether the current install of transformers is greater than or equal to the target version.

    Args:
        target_version (str): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    import transformers

    return packaging.version.parse(transformers.__version__) >= packaging.version.parse(target_version)


@contextmanager
def require_numpy_strictly_lower(version: str, message: str):
    if not packaging.version.parse(np.__version__) < packaging.version.parse(version):
        raise ImportError(
            f"Found an incompatible version of numpy. Found version {np.__version__}, but expected numpy<{version}. {message}"
        )
    try:
        yield
    finally:
        pass


DIFFUSERS_IMPORT_ERROR = """
{0} requires the diffusers library but it was not found in your environment. You can install it with pip: `pip install
diffusers`. Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("diffusers", (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


# Copied from: https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/utils/import_utils.py#L1041
class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)
