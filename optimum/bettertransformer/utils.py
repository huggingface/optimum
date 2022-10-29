# Copyright 2022 The HuggingFace Team.  All rights reserved.
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
import importlib
from contextlib import contextmanager

from packaging import version


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


def is_accelerate_available():
    return importlib.util.find_spec("accelerate") is not None
