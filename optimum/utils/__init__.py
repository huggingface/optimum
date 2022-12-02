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
        # Try to import the source file of onnxruntime - if you run the tests from `tests` the function gets
        # confused since there a folder named `onnxruntime` in `tests`. Therefore, `_onnxruntime_available`
        # will be set to `True` even if not installed.
        mod = importlib.import_module("onnxruntime")
        inspect.getsourcefile(mod)
    except:
        return False
    return _onnxruntime_available


def is_pydantic_available():
    return _pydantic_available


def is_accelerate_available():
    return _accelerate_available


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


def check_if_transformers_greater(target_version: str) -> bool:
    """
    Checks whether the current install of transformers is grater than or equal to the target version.

    Args:
        target_version (str): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    import transformers

    return version.parse(transformers.__version__) >= version.parse(target_version)


from .input_generators import (  # noqa
    DEFAULT_DUMMY_SHAPES,
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
)
from .normalized_config import (  # noqa
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)
