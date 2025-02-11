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

import importlib.metadata
import importlib.util
import operator as op
from collections import OrderedDict
from contextlib import contextmanager
from logging import getLogger
from typing import List, Optional, Tuple, Union

import numpy as np
from packaging import version


logger = getLogger(__name__)

TORCH_MINIMUM_VERSION = version.parse("1.11.0")
TRANSFORMERS_MINIMUM_VERSION = version.parse("4.25.0")
DIFFUSERS_MINIMUM_VERSION = version.parse("0.22.0")
AUTOGPTQ_MINIMUM_VERSION = version.parse("0.4.99")  # Allows 0.5.0.dev0
GPTQMODEL_MINIMUM_VERSION = version.parse("1.6.0")

# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = version.parse("1.4.0")

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}


def _is_package_available(
    pkg_name: str,
    return_version: bool = False,
    pkg_distributions: Optional[List[str]] = None,
) -> Union[Tuple[bool, str], bool]:
    """
    Check if a package is available in the current environment and not just an importable module by checking its version.
    Optionally return the version of the package.

    Args:
        pkg_name (str): The name of the package to check.
        return_version (bool): Whether to return the version of the package.
        pkg_distributions (Optional[List[str]]): A list of package distributions (e.g. "package-name", "package-name-gpu", etc.) to check for the package.

    Returns:
        Union[Tuple[bool, str], bool]: A tuple of the package availability and the version of the package if `return_version` is `True`.
    """

    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"

    if pkg_distributions is None:
        pkg_distributions = [pkg_name]
    else:
        pkg_distributions.append(pkg_name)

    if package_exists:
        for pkg in pkg_distributions:
            try:
                package_version = importlib.metadata.version(pkg)
                package_exists = True
                break
            except importlib.metadata.PackageNotFoundError:
                package_exists = False
                pass

    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_onnx_available = _is_package_available("onnx")
_pydantic_available = _is_package_available("pydantic")
_accelerate_available = _is_package_available("accelerate")
_auto_gptq_available = _is_package_available("auto_gptq")
_gptqmodel_available = _is_package_available("gptqmodel")
_timm_available = _is_package_available("timm")
_sentence_transformers_available = _is_package_available("sentence_transformers")
_datasets_available = _is_package_available("datasets")
_diffusers_available, _diffusers_version = _is_package_available("diffusers", return_version=True)
_transformers_available, _transformers_version = _is_package_available("transformers", return_version=True)
_torch_available, _torch_version = _is_package_available("torch", return_version=True)
_onnxruntime_available, _onnxruntime_version = _is_package_available(
    "onnxruntime",
    return_version=True,
    pkg_distributions=[
        "onnxruntime-gpu",
        "onnxruntime-rocm",
        "onnxruntime-training",
        # list in https://github.com/microsoft/onnxruntime/blob/main/setup.py#L56C1-L98C91
        "onnxruntime-training-rocm",
        "onnxruntime-training-cpu",
        "onnxruntime-openvino",
        "onnxruntime-vitisai",
        "onnxruntime-armnn",
        "onnxruntime-cann",
        "onnxruntime-dnnl",
        "onnxruntime-acl",
        "onnxruntime-tvm",
        "onnxruntime-qnn",
        "onnxruntime-migraphx",
        "ort-migraphx-nightly",
        "ort-rocm-nightly",
    ],
)
_tf_available, _tf_version = _is_package_available(
    "tensorflow",
    return_version=True,
    pkg_distributions=[
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-gpu",
        "tensorflow-rocm",
        "tensorflow-macos",
        "tensorflow-aarch64",
        "tf-nightly",
        "tf-nightly-cpu",
        "tf-nightly-gpu",
        "tf-nightly-rocm",
        "tf-nightly-macos",
        "intel-tensorflow",
        "intel-tensorflow-avx512",
    ],
)

if _tf_available and version.parse(_tf_version) < version.parse("2"):
    logger.warning(
        "TensorFlow 2.0 or higher is required to use the TensorFlow backend. "
        "Please install the latest version of TensorFlow, or switch to another backend."
    )
    _tf_available = False


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, version.Version], operation: str, requirement_version: str):
    """
    Compare a library version to some requirement using a given operation.

    Arguments:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = version.parse(importlib.metadata.version(library_or_version))
    return operation(library_or_version, version.parse(requirement_version))


def is_transformers_version(operation: str, reference_version: str):
    """
    Compare the current Transformers version to a given reference with an operation.
    """
    if not _transformers_available:
        return False
    return compare_versions(version.parse(_transformers_version), operation, reference_version)


def is_diffusers_version(operation: str, reference_version: str):
    """
    Compare the current diffusers version to a given reference with an operation.
    """
    if not _diffusers_available:
        return False
    return compare_versions(version.parse(_diffusers_version), operation, reference_version)


def is_torch_version(operation: str, reference_version: str):
    """
    Compare the current torch version to a given reference with an operation.
    """
    if not _torch_available:
        return False

    import torch

    return compare_versions(version.parse(version.parse(torch.__version__).base_version), operation, reference_version)


_is_torch_onnx_support_available = _torch_available and is_torch_version(">=", TORCH_MINIMUM_VERSION.base_version)


def is_torch_onnx_support_available():
    return _is_torch_onnx_support_available


def is_onnx_available():
    return _onnx_available


def is_onnxruntime_available():
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


def is_datasets_available():
    return _datasets_available


def is_transformers_available():
    return _transformers_available


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def is_auto_gptq_available():
    if _auto_gptq_available:
        v = version.parse(importlib.metadata.version("auto_gptq"))
        if v >= AUTOGPTQ_MINIMUM_VERSION:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {v}, but only version >= {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )


def is_gptqmodel_available():
    if _gptqmodel_available:
        v = version.parse(importlib.metadata.version("gptqmodel"))
        if v >= GPTQMODEL_MINIMUM_VERSION:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of gptqmodel. Found version {v}, but only version >= {GPTQMODEL_MINIMUM_VERSION} are supported"
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


# TODO : Remove check_if_transformers_greater, check_if_diffusers_greater, check_if_torch_greater
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


def check_if_torch_greater(target_version: str) -> bool:
    """
    Checks whether the current install of torch is greater than or equal to the target version.

    Args:
        target_version (str): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    if not _torch_available:
        return False

    return version.parse(_torch_version) >= version.parse(target_version)


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

DATASETS_IMPORT_ERROR = """
{0} requires the datasets library but it was not found in your environment. You can install it with pip:
`pip install datasets`. Please note that you may need to restart your runtime after installation.
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("diffusers", (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)),
        (
            "transformers_431",
            (lambda: is_transformers_version(">=", "4.31"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.31")),
        ),
        (
            "transformers_432",
            (lambda: is_transformers_version(">=", "4.32"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.32")),
        ),
        (
            "transformers_434",
            (lambda: is_transformers_version(">=", "4.34"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.34")),
        ),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
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
