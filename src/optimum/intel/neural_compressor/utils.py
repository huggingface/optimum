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

import logging
from collections import UserDict
from typing import Dict, List, Tuple

import torch
from torch.fx import GraphModule
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


CONFIG_NAME = "best_configure.yaml"
WEIGHTS_NAME = "pytorch_model.bin"


class IncDataLoader(DataLoader):
    @classmethod
    def from_pytorch_dataloader(cls, dataloader: DataLoader):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected a PyTorch DataLoader, got: {type(dataloader)}.")
        inc_dataloader = cls(dataloader.dataset)
        for key, value in dataloader.__dict__.items():
            inc_dataloader.__dict__[key] = value
        return inc_dataloader

    def __iter__(self):
        for input in super().__iter__():
            if not isinstance(input, (dict, tuple, list, UserDict)):
                raise TypeError(f"Model calibration cannot use input of type {type(input)}.")
            label = input.get("labels") if isinstance(input, dict) else None
            yield input, label


def _cfgs_to_fx_cfgs(op_cfgs: Dict, observer_type: str = "post_training_static_quant") -> Dict:
    """Inc function which convert a quantization config to a format that meets the requirements of torch.fx.
    Args:
        op_cfgs (:obj:`dict`):
            Dictionary of quantization configure for each op.
        observer_type (:obj:`str`):
            Specify observer type.
    Returns:
        fx_op_cfgs (:obj:`dict`):
            Dictionary of quantization configure that meets the requirements of torch.fx.
    """
    fx_op_cfgs = dict()
    op_tuple_cfg_list = []
    for key, value in op_cfgs.items():
        if key == "default_qconfig":
            fx_op_cfgs[""] = value
            continue
        op_tuple = (key, value)
        op_tuple_cfg_list.append(op_tuple)
    fx_op_cfgs["module_name"] = op_tuple_cfg_list
    return fx_op_cfgs


def _get_quantizable_ops_recursively(
    self, model: torch.nn.Module, prefix: str, quantizable_ops: List[Tuple[str, str]]
) -> None:
    """Inc helper function for `query_fw_capability` which get all quantizable ops from model.
    Args:
        model (:obj:`torch.nn.Module`):
            Input model.
        prefix (:obj:`str`):
            Prefix of op name.
        quantizable_ops (:obj:`List[Tuple[str, str]]`):
            List of quantizable ops from model include op name and type.
    Returns:
        None
    """
    import torch

    from neural_compressor.adaptor.pytorch import unify_op_type_mapping

    for name, child in model.named_children():
        op_name = prefix + "." + name if prefix != "" else name
        if (
            type(child) in self.white_list
            and type(child) != torch.nn.Sequential
            and type(child) != torch.quantization.stubs.DeQuantStub
            and not isinstance(child, torch.nn.LayerNorm)
            and not isinstance(child, torch.nn.Embedding)
        ):

            quantizable_ops.append(
                (
                    op_name,
                    unify_op_type_mapping[str(child.__class__.__name__)]
                    if str(child.__class__.__name__) in unify_op_type_mapping
                    else str(child.__class__.__name__),
                )
            )
        else:
            self._get_quantizable_ops_recursively(child, op_name, quantizable_ops)


def remove_inputs_from_graph(gm_original: GraphModule, inputs_to_remove: List[str]) -> GraphModule:
    """
    Remove specified inputs from a GraphModule.

    Args:
        gm_original (:obj:`GraphModule`):
            Original GraphModule.
        inputs_to_remove (:obj:`List[str]`):
            List of string specifying the name of the inputs to remove from the GraphModule.
    Returns:
        gm (:obj:`GraphModule`):
            GraphModule with the removed inputs.
    """
    from transformers.utils.fx_transformations import deepcopy_graph

    try:
        gm = deepcopy_graph(gm_original)
    except Exception as e:
        gm = gm_original
        logger.warning(f"Deepcopy failed: {repr(e)}, model is modified inplace.")

    graph = gm.graph
    output_node = list(graph.nodes)[-1]

    def remove_users(node, output_node):
        output_args = output_node.args[0] if output_node is not None else None
        for user in list(node.users):
            remove_users(user, output_node)
        if output_args is not None and node in output_args.values():
            new_output_args = dict()
            for k, v in output_args.items():
                if node is v:
                    continue
                new_output_args[k] = v
            output_node.args = (new_output_args,)
        if node is not output_node:
            graph.erase_node(node)

    for node in graph.nodes:
        if node.op == "placeholder" and node.target in inputs_to_remove:
            remove_users(node, output_node)
            graph.erase_node(node)

    graph.lint()
    gm.recompile()

    return gm
