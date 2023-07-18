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
from typing import Union

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from .constants import BLOCK_PATTERNS


logger = getLogger(__name__)


def get_module_by_name_prefix(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


def get_layers(module, layers=[Conv1D, nn.Conv2d, nn.Linear], name=""):
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_block_name(model):
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any([pattern_candidate in name for name in modules_names]):
            return pattern_candidate
    raise ValueError(
        "We are not able to infer the block name to quantize. Pass `block_name_to_quantize` argument in `quantize_model`"
    )


def get_preceding_modules(model, module_name):
    """
    We get the high-level modules preceding the one with `module_name`.
    """
    previous_module_name = []
    stop_adding = False

    def _get_preceding_modules(model, module_name, name=""):
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
