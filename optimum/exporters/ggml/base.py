# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""ggml configuration base classes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Tuple, Union

from numpy import ndarray
from torch import Tensor

from ..base import ExportConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class GgmlConfig(ExportConfig, ABC):
    """
    Base class for GGML exportable model.
    """

    STRUCT_HPARAM_KEYS = []
    USE_BYTE_DECODER = True  # TODO this should eventually be always True
    GGML_MEM_ALIGN = 16

    def __init__(self, config: "PretrainedConfig", task: str = "text-generation"):
        self.task = task
        self._config = config

    @abstractmethod
    def get_cpp_name(self, name: str) -> str:
        raise NotImplementedError

    def should_skip(self, name: str) -> bool:
        return False

    @abstractmethod
    def reshape_weights(self, name: str, weights: Union[ndarray, Tensor], hparams: Dict) -> ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_dtype(name: str, data: Union[ndarray, Tensor], ftype: int, n_dims: int) -> Tuple[ndarray, int]:
        return data, ftype


class GgmlConfigWithPast(GgmlConfig, ABC):
    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "text-generation") -> "GgmlConfigWithPast":
        return cls(config, task=task, use_past=True)
