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
from typing import List, Tuple

from onnx import ModelProto
from onnxruntime.transformers.onnx_model import OnnxModel
from optimum.onnxruntime.walker import WalkerPass


class ExcludeNodeFollowedBy(WalkerPass):
    def __init__(self, operator_type_1: str, operator_type_2: str):
        super().__init__()

        self.operator_type_1 = operator_type_1
        self.operator_type_2 = operator_type_2

    def __call__(self, graph: ModelProto, model: OnnxModel) -> Tuple[List[str], List[str]]:
        op1_nodes, op2_nodes = [], []
        selected_op1_nodes, not_selected_op1_nodes = [], []

        for node in model.graph().node:
            if node.op_type == self.operator_type_1:
                op1_nodes.append(node)

            if node.op_type == self.operator_type_2:
                op2_nodes.append(node)

        for op1_node in op1_nodes:
            for op2_node in op2_nodes:
                if op1_node.output == op2_node.input:
                    selected_op1_nodes.append(op1_node.name)

            if op1_node.name not in selected_op1_nodes:
                not_selected_op1_nodes.append(op1_node.name)

        print(not_selected_op1_nodes)
        return [], not_selected_op1_nodes
