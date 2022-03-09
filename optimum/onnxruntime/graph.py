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
from typing import List

from onnxruntime.transformers.onnx_model import OnnxModel


def find_fully_connected_layers_nodes(model: OnnxModel) -> List[List[str]]:
    adds = model.get_nodes_by_op_type("Add")
    fc = list(filter(lambda graph: graph[1] is not None, ((add, model.match_parent(add, "MatMul")) for add in adds)))

    return fc
