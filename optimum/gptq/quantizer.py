# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and BigScience workshop.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional
import logging

import torch

from auto_gptq.nn_modules.qlinear import QuantLinear
from auto_gptq.quantization import GPTQ

class GPTQQuantizer(object):
    r"""
    A simple API that converts a model to a quantized model.

    Args:
        n_bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        group_size (`int`):
            The groupe size to use for quantization, recommended value is 128.
    """
    def __init__(
        self, 
        n_bits: int, 
        group_size: int, 
        kernel_switch_threshold: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        symmetric_quantization: bool = True,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.kernel_switch_threshold = kernel_switch_threshold
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.symmetric_quantization = symmetric_quantization

    def _find_and_replace_linear(
        self,
        model: torch.nn.Module,
        names_to_not_convert: Optional[List[str]] = None
    ):
        r"""
        A simple function that finds all the linear layers in a model and replaces them with
        quantized linear layers.
        """
        if names_to_not_convert is None:
            names_to_not_convert = []

        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear) and name not in names_to_not_convert:
                current_device = module.weight.device

                quant_linear = QuantLinear(
                    bits=self.n_bits,
                    group_size=self.group_size,
                    infeatures=module.in_features,
                    outfeatures=module.out_features,
                    bias=module.bias is not None,
                )

                if current_device.type != "meta":
                    quant_linear = quant_linear.to(current_device)

                quant_linear.weight = module.weight
                model._modules[name] = quant_linear

                if module.weight.dtype != torch.float16:
                    logging.warning(f"GPTQ expects the model to be in float16, but {name} is in {module.weight.dtype}.")

            if len(list(module.children())) > 0:
                _ = self.convert_to_gptq_model(module)
        return model

    def convert_to_gptq_model(
        self, 
        model: torch.nn.Module,
        names_to_not_convert: Optional[List[str]] = None
    ):
        r"""
        A simple function that converts a model to a quantized model.
        """
        model = self._find_and_replace_linear(model, names_to_not_convert)

        # Users need to quantize the model before using it
        model._gptq_is_quantized = False

        return model


    def quantize_model(
        self,
        model: torch.nn.Module,
        examples: torch.Tensor,
        names_to_not_convert: Optional[List[str]] = None
    ):
        r"""
        Add docstring here
        """
        gptq_objects = {}
        forward_tmp_hooks = []

        model = model.to("cpu")

        for name, module in model.named_children():
            # if isinstance(module, QuantLinear):
            if isinstance(module, torch.nn.Linear):

                def add_batch(name):
                    def forward_hook(_, inp, output):
                        if isinstance(inp, tuple):
                            inp = inp[0].squeeze(0)

                        gptq_objects[name].add_batch(inp, output)
                    return forward_hook

                gptq_object = GPTQ(module)
                gptq_object.quantizer.configure(
                    self.n_bits,
                    perchannel=True,
                    sym=self.symmetric_quantization,
                    mse=False,
                )
                gptq_objects[name] = gptq_object
                forward_tmp_hooks.append(module.register_forward_hook(add_batch(name)))
        
        # Step 3 loop over all examples
        for example in examples:
            _ = model(**example)
        
        for hooks in forward_tmp_hooks:
            hooks.remove()

        model = self._find_and_replace_linear(model, names_to_not_convert)

        for name, module in model.named_children():
            if isinstance(module, QuantLinear):
                scale, zero, g_idx = gptq_objects[name].fasterquant(
                    percdamp=self.damp_percent,
                    group_size=self.group_size,
                    actorder=self.desc_act
                )

                module.pack(module, scale, zero, g_idx)

                gptq_objects[name].free()
                del module.weight
    
        return model