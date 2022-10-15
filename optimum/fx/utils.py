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
from contextlib import contextmanager
from functools import wraps

import transformers
from packaging import version


_TRANSFORMERS_MIN_VERSION = version.parse("4.20.0.dev0")

transformers_version = version.parse(transformers.__version__)
_fx_features_available = (_TRANSFORMERS_MIN_VERSION.major, _TRANSFORMERS_MIN_VERSION.minor) <= (
    transformers_version.major,
    transformers_version.minor,
)


def are_fx_features_available():
    return _fx_features_available


def check_if_available(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not are_fx_features_available():
            raise ImportError(
                f"Found an incompatible version of transformers. Found version {transformers_version}, but only {_TRANSFORMERS_MIN_VERSION} and above are supported."
            )
        return func(*args, **kwargs)

    return wrapper


@contextmanager
def check_if_pytorch_greater_112():
    r"""
    A context manager that does nothing except checking if the PyTorch version is greater than 1.12.0.
    """
    import torch

    if version.parse(torch.__version__) < version.parse("1.12.0"):
        raise ImportError(
            f"Found an incompatible version of PyTorch. Found version {torch.__version__}, but only 1.12.0 and above are supported."
        )
    try:
        yield
    finally:
        pass
