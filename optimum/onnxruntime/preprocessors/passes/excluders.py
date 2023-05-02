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


class ExcludeNodeFollowedBy(PreprocessorPass):
    def __init__(self, operator_type_to_exclude: str, following_operator_type: str):
        super().__init__()

        self.operator_type_to_exclude = operator_type_to_exclude
        self.following_operator_type = following_operator_type

    def __call__(self, _: ModelProto, model: OnnxModel) -> Tuple[Set[str], Set[str]]:
        # Find out the nodes to exclude in the graph
        candidate_nodes_to_exclude = {
            candidate_output: candidate.name
            for candidate in model.get_nodes_by_op_type(self.operator_type_to_exclude)
            for candidate_output in candidate.output
        }

        nodes_of_following_type = {
            node_input: node.name
            for node in model.get_nodes_by_op_type(self.following_operator_type)
            for node_input in node.input
        }

        # Intersection of both are the one we want to remove
        to_exclude = set(candidate_nodes_to_exclude.keys()).intersection(nodes_of_following_type.keys())
        nodes_to_exclude = {candidate_nodes_to_exclude[node] for node in to_exclude}

        return set(), nodes_to_exclude


class ExcludeNodeAfter(PreprocessorPass):
    def __init__(self, parent_operator_type: str, operator_type_to_exclude: str):
        super().__init__()

        self.parent_operator_type = parent_operator_type
        self.operator_type_to_exclude = operator_type_to_exclude

    def __call__(self, graph: ModelProto, model: OnnxModel) -> Tuple[Set[str], Set[str]]:
        # Find out the nodes to exclude in the graph
        candidate_nodes_to_exclude = {
            candidate_input: candidate.name
            for candidate in model.get_nodes_by_op_type(self.operator_type_to_exclude)
            for candidate_input in candidate.input
        }

        parent_node = {
            node_output: node.name
            for node in model.get_nodes_by_op_type(self.parent_operator_type)
            for node_output in node.output
        }

        # Intersection of both are the one we want to remove
        to_exclude = set(candidate_nodes_to_exclude.keys()).intersection(parent_node.keys())
        nodes_to_exclude = {candidate_nodes_to_exclude[node] for node in to_exclude}

        return set(), nodes_to_exclude
