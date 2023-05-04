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
from typing import Set, Tuple

from onnx import ModelProto

from onnxruntime.transformers.onnx_model import OnnxModel

from .. import PreprocessorPass


class IncludeFullyConnectedNodes(PreprocessorPass):
    def __init__(self):
        super().__init__()

    def __call__(self, graph: ModelProto, model: OnnxModel) -> Tuple[Set[str], Set[str]]:
        fc_subgraphs = []
        for add_node in model.get_nodes_by_op_type("Add"):
            fc_components = model.match_parent_path(add_node, ["MatMul"], [1])
            if fc_components is not None:
                fc_components.append(add_node)
                fc_subgraphs.append(fc_components)
        fc_components = {node.name for fc in fc_subgraphs for node in fc}
        return fc_components, set()
