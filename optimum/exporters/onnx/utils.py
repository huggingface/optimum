# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from ctypes import c_float, sizeof
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # tests_ignore


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


def compute_effective_axis_dimension(dimension: int, fixed_dimension: int, num_token_to_add: int = 0) -> int:
    """

    Args:
        dimension:
        fixed_dimension:
        num_token_to_add:

    Returns:

    """
    # < 0 is possible if using a dynamic axis
    if dimension <= 0:
        dimension = fixed_dimension

    dimension -= num_token_to_add
    return dimension


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


def get_preprocessor(model_name: str) -> Optional[Union["AutoTokenizer", "AutoFeatureExtractor", "AutoProcessor"]]:
    """
    Gets a preprocessor (tokenizer, feature extractor or processor) that is available for `model_name`.

    Args:
        model_name (`str`): Name of the model for which a preprocessor are loaded.

    Returns:
        `Optional[Union[AutoTokenizer, AutoFeatureExtractor, AutoProcessor]]`:
            If a processor is found, it is returned. Otherwise, if a tokenizer or a feature extractor exists, it is
            returned. If both a tokenizer and a feature extractor exist, an error is raised. The function returns
            `None` if no preprocessor is found.
    """
    # Avoid circular imports by only importing this here.
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # tests_ignore

    try:
        return AutoProcessor.from_pretrained(model_name)
    except (ValueError, OSError, KeyError):
        tokenizer, feature_extractor = None, None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except (OSError, KeyError):
            pass
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        except (OSError, KeyError):
            pass

        if tokenizer is not None and feature_extractor is not None:
            raise ValueError(
                f"Couldn't auto-detect preprocessor for {model_name}. Found both a tokenizer and a feature extractor."
            )
        elif tokenizer is None and feature_extractor is None:
            return None
        elif tokenizer is not None:
            return tokenizer
        else:
            return feature_extractor
