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

from torch import nn
from transformers.pytorch_utils import Conv1D
from logging import getLogger 
from typing import Union
import torch

logger = getLogger(__name__)

def get_module_by_name_prefix(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module
        
def get_layers(module, layers=None, name=''):
    if not layers:
        layers = [Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module,layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def get_sequential_layers_name(module):
    pass

def get_previous_module_name(module_name):
    pass

def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device