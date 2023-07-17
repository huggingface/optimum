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
import torch
import logging
from typing import List, Optional
from auto_gptq.nn_modules.qlinear import QuantLinear
from auto_gptq.quantization import GPTQ
from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
from .data import get_examples, prepare_examples
from .utils import get_module_by_name_prefix, get_layers, get_sequential_layers_name, get_previous_module_name, pack_model, get_device
from torch import nn
from logging import getLogger 
from transformers.pytorch_utils import Conv1D
logger = getLogger(__name__)

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
        damp_percent: float = 0.01,
        desc_act: bool = True,
        symmetric_quantization: bool = True,
        use_triton: bool = False,
        warmup_triton: bool = False,
        use_cuda_fp16: bool = True,
        true_sequential: bool = True
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.kernel_switch_threshold = kernel_switch_threshold
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.symmetric_quantization = symmetric_quantization
        self.use_triton = use_triton
        self.warmup_triton = warmup_triton
        self.use_cuda_fp16 = use_cuda_fp16
        self.true_sequential = true_sequential

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
        
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

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
                    sym=self.symmetric_quantization,
                    perchannel=True,
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
    
    def _replace_by_quant_linear(self,module,names,name=''):
        QuantLinear = dynamically_import_QuantLinear(use_triton=self.use_triton, desc_act=self.desc_act, group_size=self.group_size)
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            layer = getattr(module, attr)
            name1 = name + '.' + attr if name != '' else attr
            if name1 in names:
                device = get_device(layer)
                delattr(module, attr)
                if isinstance(layer,nn.Linear):
                    in_features = layer.in_features
                    out_features = layer.out_features
                elif isinstance(layer,nn.Conv2d):
                    in_features = layer.in_channels
                    out_features = layer.out_channels
                elif isinstance(layer,Conv1D):            
                    in_features = layer.weight.shape[0]
                    out_features = layer.weight.shape[1]
                if (not(self.desc_act) or self.group_size == -1) and not self.use_triton:
                    new_layer = QuantLinear(self.n_bits, self.group_size, in_features, out_features, True, use_cuda_fp16=self.use_cuda_fp16)
                else:
                    new_layer = QuantLinear(self.n_bits, self.group_size, in_features, out_features, True)
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_linear(child, names, name + '.' + name1 if name != '' else name1)


    def quantize_model_bis(self, model, examples, tokenizer=None):
        device = model.device
        if isinstance(examples,str):
            if tokenizer is None:
                raise ValueError(f"You need to provide a tokenizer in order to process the data from {examples} dataset")
            examples = get_examples(examples,tokenizer)
        # padding not necessary if batch size==1
        # specific to transformers
        examples = prepare_examples(examples, model.config.pad_token_id)
        
        # need to get the input for the related layers
        layer_inputs = []
        layer_outputs = []
        layer_input_kwargs = []
        
        # specific to transformers
        use_cache = model.config.use_cache
        model.config.use_cache = False
        
        class Catcher(nn.Module):
            """hijack layer's forward pass to cache data"""
            def __init__(self, m):
                super().__init__()
                self.module = m

            def forward(self, input=None, **kwargs):
                # specific to transformers
                 # some models use all key-value arguments in forward pass call
                if input is None:
                    if "hidden_states" in kwargs:
                        input = kwargs["hidden_states"]
                    else: 
                        raise ValueError("No input value found in the foward pass")
                layer_inputs.append(input)
                other_kwargs = dict()
                for k, v in kwargs.items():  # make sure other arguments also be captured
                    if k not in ["hidden_states"]:
                        other_kwargs[k] = v
                layer_input_kwargs.append(other_kwargs)
                raise ValueError

        # specific to transformers
        # block to quantize
        # only one type of block for now (encoder-decoder not supported yet)
        block_to_quantize = "model.decoder.layers"
        blocks = get_module_by_name_prefix(model, block_to_quantize)
        
        # get layers that are before that block 
        module_name_before_first_block = ["model.decoder.embed_tokens", "model.decoder.embed_positions","model.decoder.final_layer_norm"]
        if module_name_before_first_block is None:
            module_name_before_first_block = get_previous_module_name(blocks[0])
        
        # put modules from module_name_before_first_block on cuda
        for module_name in module_name_before_first_block:
            module = get_module_by_name_prefix(model, module_name)
            if module is None:
                raise ValueError(f"Module {module_name} was not found in model")
            module = module.to(0)
            
        layers_block_name = "model.decoder.layers"

        # get inputs for first block
        # run the block on gpu 
        blocks[0] = blocks[0].to(0)
        blocks[0] = Catcher(blocks[0])
        for example in examples:
            for k, v in example.items():
                # put on gpu, we won't put them back to cpu
                example[k] = v.to(0)
            try:
                model(**example)
            except ValueError:
                pass
        blocks[0] = blocks[0].module
        
        # move everything back to device
        blocks[0] = blocks[0].to(device)
        for module_name in module_name_before_first_block:
            module = get_module_by_name_prefix(model, module_name)
            if module is None:
                raise ValueError(f"Module {module_name} was not found in model")
            module = module.to(device)
        # clean cache
        torch.cuda.empty_cache()
        
        quantizers = {}
        num_batches = len(examples)
        # start quantizing the blocks
        for i, block in enumerate(blocks):
            logger.info(f"Start quantizing block {i + 1}/{len(blocks)}")
            # move block to cuda
            block = block.to(0)
            # get linear layers, should only be composed of linear layers ... 
            layers = get_layers(block)
            if self.true_sequential:
                # should be able to get those sequential layers from the model
                layers_name_list = get_sequential_layers_name(block)
                layers_name_list = [
                    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                    ["self_attn.out_proj"],
                    ["fc1"],
                    ["fc2"]
                    ]
            else:
                layers_name_list = [list(layers.keys())]
            logger.info(f'Module to quantize {layers_name_list}')
            for subset_name_list in layers_name_list:
                subset_layers = {name: layers[name] for name in subset_name_list}
                gptq = {}
                handles = []
                # add hook for each layer in subset_layers
                for name in subset_layers:
                    gptq[name] = GPTQ(subset_layers[name])
                    gptq[name].quantizer.configure(
                        bits=self.n_bits,
                        sym=self.symmetric_quantization,
                        perchannel=True
                    )
                    def add_batch(name):
                        def tmp(_, input, output):
                            gptq[name].add_batch(input[0].data, output.data)
                        return tmp
                    handles.append(subset_layers[name].register_forward_hook(add_batch(name)))
                # update Hessian for each layer in subset_layers thanks to the hook 
                for j in range(num_batches):
                    # the args are already on the gpu
                    # don't need to store the output
                    block(layer_inputs[j],**layer_input_kwargs[j])
                # remove hook 
                for h in handles:
                    h.remove()
                # quantize layers but we keep the layer as Linear layers
                # we can maybe pack them here
                for name in subset_name_list:
                    logger.info(f'Quantizing {name} in block {i + 1}/{len(blocks)}...')
                    
                    scale, zero, g_idx = gptq[name].fasterquant(
                        percdamp=self.damp_percent,
                        group_size=self.group_size,
                        actorder=self.desc_act
                    )
                    # if we pack the model at the end
                    # quantizers[f'{layers_block_name}.{i}.{name}'] = (
                    #     gptq[name].quantizer,
                    #     scale,
                    #     zero,
                    #     g_idx
                    # )
                    # gptq[name].free()
                    
                    # put on cpu because it is not possible to quantize on cuda for now
                    subset_layers[name], scale, zero, g_idx = subset_layers[name].to('cpu'), scale.to('cpu'), zero.to('cpu'), g_idx.to('cpu')
                    layer_name = f'{layers_block_name}.{i}.{name}'
                    self._replace_by_quant_linear(model,[layer_name])
                    quantized_layer = get_module_by_name_prefix(model,layer_name)
                    quantized_layer = quantized_layer.to("cpu")
                    quantized_layer.pack(subset_layers[name], scale, zero, g_idx)
                    quantized_layer=quantized_layer.to(0)
                del subset_layers
            # we get the new output from the partial quantized block 
            for j in range(num_batches):
                layer_output = block(layer_inputs[j],**layer_input_kwargs[j])[0]
                layer_outputs.append(layer_output)

            # put back to device 
            blocks[i] = block.to(device)
            del gptq
            del layers
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()
        
        # if we pack the model at the end
        # self.pack_model(
        #     model=model,
        #     quantizers=quantizers
        # )
        model.config.use_cache = use_cache
        self._quantized = True
        
        torch.cuda.empty_cache()
        
        return model
    
    def pack_model(
        self,
        model,
        quantizers,
    ):
        QuantLinear = dynamically_import_QuantLinear(use_triton=self.use_triton, desc_act=self.desc_act, group_size=self.group_size)
        logger.info('Packing model...')
        layers = get_layers(model)
        layers = {n: layers[n] for n in quantizers}
        self._replace_by_quant_linear(model, quantizers)
        qlayers = get_layers(model, [QuantLinear])
        for name in qlayers:
            logger.info(name)
            quantizers[name], scale, zero, g_idx = quantizers[name]
            # so far can only pack layer on CPU
            layer_device = qlayers[name].device
            qlayers[name].to('cpu')
            layers[name], scale, zero, g_idx = layers[name].to('cpu'), scale.to('cpu'), zero.to('cpu'), g_idx.to('cpu')
            qlayers[name].pack(layers[name], scale, zero, g_idx)
            qlayers[name].to(layer_device)
            
        logger.info('Model packed.')

        if self.use_triton and self.warmup_triton:
            logger.warning(
                "using autotune_warmup will move model to GPU, make sure you have enough VRAM to load the whole model."
            )
            QuantLinear.warmup(model.to(0), seqlen=model.seqlen)