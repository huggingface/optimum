#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

from logging import getLogger
from typing import Optional, Union

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from .constants import BLOCK_PATTERNS, SEQLEN_KEYS_TRANFORMERS


logger = getLogger(__name__)


"""
Set of utilities to get specific attributes of a model
"""


def get_layers(module: nn.Module, layers=[Conv1D, nn.Conv2d, nn.Linear], prefix: Optional[str] = None, name: str = ""):
    """
    Get all the layers with a specific prefix in the module
    Args:
        module (`nn.Module`):
            The module that contains our layers
        layers (`list`, defaults to `[Conv1D, nn.Conv2d, nn.Linear]`):
            Type of the layers that we want to get
        prefix (`Optional[str]`, defaults to `None`):
            Prefix of layers
        name (`str`, defaults to `""`):
            Used for recursion. Don't modify

    Returns:
        `Dict[str,Union[Conv1D, nn.Conv2d, nn.Linear]]`: Mapping of the name of the layer and the actual layer
    """
    for layer in layers:
        if isinstance(module, layer):
            if prefix is not None:
                if name.startswith(prefix):
                    return {name: module}
            else:
                return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_layers(child, layers=layers, prefix=prefix, name=name + "." + name1 if name != "" else name1))
    return res


def get_block_name_with_pattern(model: nn.Module):
    """
    Get the name of the module that contains the transformers blocks by checking if any modules has a specific pattern

    Args:
        model (`nn.Module`):
        The input model
    Returns:
        `str`: The name of the module that contains the Transformer blocks.
    """
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any(name.startswith(pattern_candidate) for name in modules_names):
            return pattern_candidate
    raise ValueError("Block pattern could not be match. Pass `block_name_to_quantize` argument in `quantize_model`")


def get_preceding_modules(model: nn.Module, module_name: str):
    previous_module_name = []
    stop_adding = False

    def _get_preceding_modules(model: nn.Module, module_name: str, name: str = ""):
        nonlocal stop_adding
        for name_bis, child in model.named_children():
            new_name = name + "." + name_bis if name != "" else name_bis
            if new_name == module_name:
                stop_adding = True
                break
            _get_preceding_modules(child, module_name, name=new_name)
        if not stop_adding:
            previous_module_name.append(name)
        return previous_module_name

    return _get_preceding_modules(model, module_name)


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def get_seqlen(model: nn.Module):
    if hasattr(model, "config"):
        model_config = model.config.to_dict()
        if any(k in model_config for k in SEQLEN_KEYS_TRANFORMERS):
            for key in SEQLEN_KEYS_TRANFORMERS:
                if key in model_config:
                    return model_config[key]
    logger.info(
        "We couldn't get the model sequence length. Setting it to 2048. You can overwrite this value by passing `model_seqlen` in` GPTQQuantizer`"
    )
    return 2048


def move_to(obj: torch.Tensor, device: torch.device):
    if get_device(obj) != device:
        obj = obj.to(device)
    return obj


def nested_move_to(v, device):
    if isinstance(v, torch.Tensor):
        return move_to(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to(e, device) for e in v])
    else:
        return v
