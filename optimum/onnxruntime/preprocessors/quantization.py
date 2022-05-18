#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
from abc import ABC, abstractmethod
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Optional, Set, Tuple, Union

from onnx import ModelProto, load_model
from onnxruntime.transformers.onnx_model import OnnxModel


LOGGER = getLogger("GraphWalker")


class PreprocessorPass(ABC):
    def __init__(self):
        self._logger = LOGGER

    @abstractmethod
    def __call__(self, graph: ModelProto, model: OnnxModel) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        raise NotImplementedError()


class QuantizationPreprocessor:
    __slots__ = ("_passes",)

    def __init__(self):
        self._passes = []

    def from_config(self, config):
        pass

    def register_pass(self, target: PreprocessorPass):
        if target not in self._passes:
            self._passes.append(target)

    def collect(self, model_or_path: Union[str, PathLike, Path, bytes]) -> Tuple[Set[str], Set[str]]:
        global_nodes_to_quantize, global_nodes_to_exclude = set(), set()
        graph = load_model(model_or_path.as_posix() if isinstance(model_or_path, Path) else model_or_path)
        model = OnnxModel(graph)

        for walking_pass in self._passes:
            nodes_to_quantize, nodes_to_exclude = walking_pass(graph, model)

            if nodes_to_quantize is not None:
                global_nodes_to_quantize.update(nodes_to_quantize)

            if nodes_to_exclude is not None:
                global_nodes_to_exclude.update(nodes_to_exclude)

        # Exclude the nodes from quantization when present in both sets
        global_nodes_to_quantize = global_nodes_to_quantize - global_nodes_to_exclude

        return global_nodes_to_quantize, global_nodes_to_exclude
