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
from typing import Tuple, Union

import numpy as np
from packaging import version
from transformers.utils import is_torch_available


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


TORCH_MINIMUM_VERSION = version.parse("1.11.0")
TRANSFORMERS_MINIMUM_VERSION = version.parse("4.25.0")
DIFFUSERS_MINIMUM_VERSION = version.parse("0.22.0")
AUTOGPTQ_MINIMUM_VERSION = version.parse("0.4.99")  # Allows 0.5.0.dev0


# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = version.parse("1.4.0")


_onnx_available = _is_package_available("onnx")

# importlib.metadata.version seem to not be robust with the ONNX Runtime extensions (`onnxruntime-gpu`, etc.)
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None

_pydantic_available = _is_package_available("pydantic")
_accelerate_available = _is_package_available("accelerate")
_diffusers_available = _is_package_available("diffusers")
_auto_gptq_available = _is_package_available("auto_gptq")
_timm_available = _is_package_available("timm")
_sentence_transformers_available = _is_package_available("sentence_transformers")

torch_version = None
if is_torch_available():
    torch_version = version.parse(importlib_metadata.version("torch"))

_is_torch_onnx_support_available = is_torch_available() and (
    TORCH_MINIMUM_VERSION.major,
    TORCH_MINIMUM_VERSION.minor,
) <= (
    torch_version.major,
    torch_version.minor,
)


_diffusers_version = None
if _diffusers_available:
    try:
        _diffusers_version = importlib_metadata.version("diffusers")
    except importlib_metadata.PackageNotFoundError:
        _diffusers_available = False


def is_torch_onnx_support_available():
    return _is_torch_onnx_support_available


def is_onnx_available():
    return _onnx_available


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


def is_timm_available():
    return _timm_available


def is_sentence_transformers_available():
    return _sentence_transformers_available


def is_auto_gptq_available():
    if _auto_gptq_available:
        version_autogptq = version.parse(importlib_metadata.version("auto_gptq"))
        if AUTOGPTQ_MINIMUM_VERSION < version_autogptq:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq}, but only version above {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )


@contextmanager
def check_if_pytorch_greater(target_version: str, message: str):
    r"""
    A context manager that does nothing except checking if the PyTorch version is greater than `pt_version`
    """
    import torch

    if not version.parse(torch.__version__) >= version.parse(target_version):
        raise ImportError(
            f"Found an incompatible version of PyTorch. Found version {torch.__version__}, but only {target_version} and above are supported. {message}"
        )
    try:
        yield
    finally:
        pass


def check_if_transformers_greater(target_version: Union[str, version.Version]) -> bool:
    """
    Checks whether the current install of transformers is greater than or equal to the target version.

    Args:
        target_version (`Union[str, packaging.version.Version]`): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    import transformers

    if isinstance(target_version, str):
        target_version = version.parse(target_version)

    return version.parse(transformers.__version__) >= target_version


def check_if_diffusers_greater(target_version: str) -> bool:
    """
    Checks whether the current install of diffusers is greater than or equal to the target version.

    Args:
        target_version (str): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    if not _diffusers_available:
        return False

    return version.parse(_diffusers_version) >= version.parse(target_version)


@contextmanager
def require_numpy_strictly_lower(package_version: str, message: str):
    if not version.parse(np.__version__) < version.parse(package_version):
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

TRANSFORMERS_IMPORT_ERROR = """requires the transformers>={0} library but it was not found in your environment. You can install it with pip: `pip install
-U transformers`. Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("diffusers", (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)),
        (
            "transformers_431",
            (lambda: check_if_transformers_greater("4.31"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.31")),
        ),
        (
            "transformers_432",
            (lambda: check_if_transformers_greater("4.32"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.32")),
        ),
        (
            "transformers_434",
            (lambda: check_if_transformers_greater("4.34"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.34")),
        ),
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
