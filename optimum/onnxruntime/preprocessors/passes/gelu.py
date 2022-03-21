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
from typing import Optional, Set, Tuple

from onnx import ModelProto
from onnxruntime.transformers.onnx_model import OnnxModel
from optimum.onnxruntime.preprocessors import PreprocessorPass


class ExcludeGeLUNodes(PreprocessorPass):

    def __init__(self):
        super().__init__()

    def __call__(self, graph: ModelProto, model: OnnxModel) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        gelu_subgraphs = []
        for mul_node in model.get_nodes_by_op_type("Mul"):
            gelu_components = model.match_parent_path(
                mul_node,
                ["Mul", "Add", "Erf", "Div"],
                [0, 1, 0, 0]
            )

            if gelu_components is not None:
                gelu_components.append(mul_node)
                gelu_subgraphs.append(gelu_components)

        ln_components = (node.name for ln in gelu_subgraphs for node in ln)
        return set(), set(ln_components)

