#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import importlib.util
import inspect
from contextlib import contextmanager

from packaging import version


CONFIG_NAME = "config.json"

_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
_pydantic_available = importlib.util.find_spec("pydantic") is not None
_accelerate_available = importlib.util.find_spec("accelerate") is not None


def is_onnxruntime_available():
    try:
        # Try to import the source file of onnxruntime
        mod = importlib.import_module("onnxruntime")
        inspect.getsourcefile(mod)
    except TypeError:
        return False
    return _onnxruntime_available


def is_pydantic_available():
    return _pydantic_available


def is_accelerate_available():
    return _accelerate_available


def is_pytorch_greater_112():
    import torch

    return version.parse(torch.__version__) >= version.parse("1.12.0")


@contextmanager
def check_if_pytorch_greater_112():
    r"""
    A context manager that does nothing except checking if the PyTorch version is greater than 1.12.0.
    """
    import torch

    if not is_pytorch_greater_112():
        raise ImportError(
            f"Found an incompatible version of PyTorch. Found version {torch.__version__}, but only 1.12.0 and above are supported."
        )
    try:
        yield
    finally:
        pass


from .input_generators import (  # noqa
    DummyBboxInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    NormalizedConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)
