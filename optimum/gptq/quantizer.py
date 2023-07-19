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
from logging import getLogger
import os
import json

from typing import Optional
import torch
from auto_gptq.quantization import GPTQ
from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
from torch import nn
from transformers.pytorch_utils import Conv1D

from .data import get_examples, prepare_examples
from .utils import get_block_name, get_device, get_layers, get_module_by_name_prefix, get_preceding_modules, get_seqlen
from .constants import WEIGHTS_NAME, GPTQ_CONFIG
from ..utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator, load_checkpoint_and_dispatch

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
        damp_percent: float = 0.01,
        desc_act: bool = True,
        symmetric_quantization: bool = True,
        true_sequential: bool = True,
        pack_sequentially: bool = True,
        use_triton: bool = False,
        warmup_triton: bool = False,
        use_cuda_fp16: bool = True,
        model_seqlen: Optional[int] = None, 
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[str] = None,
    ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.symmetric_quantization = symmetric_quantization
        self.true_sequential = true_sequential
        self.pack_sequentially = pack_sequentially
        self.use_triton = use_triton
        self.warmup_triton = warmup_triton
        self.use_cuda_fp16 = use_cuda_fp16
        self.block_name_to_quantize=block_name_to_quantize
        self.module_name_preceding_first_block=module_name_preceding_first_block
        self.model_seqlen = model_seqlen

    def to_dict(self):
        '''
        Only return args that are useful for loading a quantized model
        '''
        return {
            "n_bits": self.n_bits,
            "group_size": self.group_size,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "symmetric_quantization": self.symmetric_quantization,
            "use_triton":self.use_triton,
            "warmup_triton":self.warmup_triton,
            "true_sequential":self.true_sequential,
            "pack_sequentially": self.pack_sequentially,
            "use_cuda_fp16": self.use_cuda_fp16,
            "model_seqlen": self.model_seqlen,
            "block_name_to_quantize": self.block_name_to_quantize,
            "module_name_preceding_first_block":self.module_name_preceding_first_block
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def _replace_by_quant_layers(self, module, names, name=""):
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=self.use_triton, desc_act=self.desc_act, group_size=self.group_size
        )
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            layer = getattr(module, attr)
            name1 = name + "." + attr if name != "" else attr
            if name1 in names:
                device = get_device(layer)
                delattr(module, attr)
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    out_features = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    in_features = layer.in_channels
                    out_features = layer.out_channels
                elif isinstance(layer, Conv1D):
                    in_features = layer.weight.shape[0]
                    out_features = layer.weight.shape[1]
                if (not (self.desc_act) or self.group_size == -1) and not self.use_triton:
                    new_layer = QuantLinear(
                        self.n_bits, self.group_size, in_features, out_features, True, use_cuda_fp16=self.use_cuda_fp16
                    )
                else:
                    new_layer = QuantLinear(self.n_bits, self.group_size, in_features, out_features, True)
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_layers(child, names, name + "." + name1 if name != "" else name1)

    def quantize_model(
        self,
        model,
        examples,
        tokenizer=None,
        pad_token_id=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed to quantize model.")
        
        device = get_device(model)

        if self.model_seqlen is None:
            seqlen = get_seqlen(model)
            if seqlen is not None:
                self.model_seqlen = seqlen
            else:
                self.model_seqlen = 4096
                logger.warning("can't get model's sequence length from model config, will set to 4096.")
    
        # Step 1: Prepare the data
        if isinstance(examples, str):
            if tokenizer is None:
                raise ValueError(
                    f"You need to provide a tokenizer in order to process the data from {examples} dataset"
                )
            examples = get_examples(examples, tokenizer, seqlen=self.model_seqlen)
        examples = prepare_examples(examples, pad_token_id=pad_token_id)

        # Step 2: get the input of the 1st block
        layer_inputs = []
        layer_outputs = []
        layer_input_kwargs = []

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
                other_kwargs = {}
                for k, v in kwargs.items():  # make sure other arguments also be captured
                    if k not in ["hidden_states"]:
                        other_kwargs[k] = v
                layer_input_kwargs.append(other_kwargs)
                raise ValueError

        # get block_name
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name(model)

        # get modules_name that are preceding the first block
        if self.module_name_preceding_first_block is None:
            self.module_name_preceding_first_block = get_preceding_modules(model, self.block_name_to_quantize)

        # get block
        blocks = get_module_by_name_prefix(model, self.block_name_to_quantize)

        # put modules from module_name_preceding_first_block on cuda
        for module_name in self.module_name_preceding_first_block:
            module = get_module_by_name_prefix(model, module_name)
            if module is None:
                raise ValueError(f"Module {module_name} was not found in model")
            module = module.to(0)

        # get inputs by running self.module_name_preceding_first_block + first block on gpu
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
        for module_name in self.module_name_preceding_first_block:
            module = get_module_by_name_prefix(model, module_name)
            if module is None:
                raise ValueError(f"Module {module_name} was not found in model")
            module = module.to(device)
        torch.cuda.empty_cache()

        # Step 3: Quantize the blocks
        quantizers = {}
        # start quantizing the blocks
        for i, block in enumerate(blocks):
            logger.info(f"Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}")
            # move block to cuda
            block = block.to(0)
            layers = get_layers(block)
            if self.true_sequential:
                # lazy sequential but works well
                layers_name_list = [[key] for key in layers.keys()]
            else:
                layers_name_list = [list(layers.keys())]
            logger.info(f"Module to quantize {layers_name_list}")
            for subset_name_list in layers_name_list:
                subset_layers = {name: layers[name] for name in subset_name_list}
                gptq = {}
                handles = []
                # add hook for each layer in subset_layers
                for name in subset_layers:
                    gptq[name] = GPTQ(subset_layers[name])
                    gptq[name].quantizer.configure(bits=self.n_bits, sym=self.symmetric_quantization, perchannel=True)

                    def add_batch(name):
                        def tmp(_, input, output):
                            gptq[name].add_batch(input[0].data, output.data)

                        return tmp

                    handles.append(subset_layers[name].register_forward_hook(add_batch(name)))
                # update Hessian for each layer in subset_layers thanks to the hook
                for j in range(len(examples)):
                    # the args are already on the gpu
                    # don't need to store the output
                    block(layer_inputs[j], **layer_input_kwargs[j])
                # remove hook
                for h in handles:
                    h.remove()
                for name in subset_name_list:
                    logger.info(f"Quantizing {name} in block {i + 1}/{len(blocks)}...")
                    scale, zero, g_idx = gptq[name].fasterquant(
                        percdamp=self.damp_percent, group_size=self.group_size, actorder=self.desc_act
                    )
                    # if we pack the model at the end
                    if not self.pack_sequentially:
                        quantizers[f"{self.block_name_to_quantize}.{i}.{name}"] = (gptq[name].quantizer, scale, zero, g_idx)
                    else:
                        # put on cpu because it is not possible to quantize on cuda for now
                        subset_layers[name], scale, zero, g_idx = (
                            subset_layers[name].to("cpu"),
                            scale.to("cpu"),
                            zero.to("cpu"),
                            g_idx.to("cpu"),
                        )
                        layer_name = f"{self.block_name_to_quantize}.{i}.{name}"
                        self._replace_by_quant_layers(model, [layer_name])
                        quantized_layer = get_module_by_name_prefix(model, layer_name)
                        quantized_layer = quantized_layer.to("cpu")
                        quantized_layer.pack(subset_layers[name], scale, zero, g_idx)
                        quantized_layer = quantized_layer.to(0)
                    gptq[name].free()
                del subset_layers
            # we get the new output from the partial quantized block
            for j in range(len(examples)):
                layer_output = block(layer_inputs[j], **layer_input_kwargs[j])[0]
                layer_outputs.append(layer_output)

            # put back to device
            blocks[i] = block.to(device)
            del layers
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        # Step 4 (Optional) : Pack the model at the end (Replacing the layers)
        # if we pack the model at the end
        if not self.pack_sequentially:
            self.pack_model(model=model, quantizers=quantizers)

        # Step 5 (Optional) :Warmup
        if self.use_triton and self.warmup_triton:
            QuantLinear = dynamically_import_QuantLinear(
                use_triton=self.use_triton, desc_act=self.desc_act, group_size=self.group_size
            )
            logger.warning(
                "using autotune_warmup will move model to GPU, make sure you have enough VRAM to load the whole model."
            )
            QuantLinear.warmup(model.to(0), seqlen=self.model_seqlen)

        model._gptq_is_quantized = True
        torch.cuda.empty_cache()
        return model

    def pack_model(
        self,
        model,
        quantizers,
    ):
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=self.use_triton, desc_act=self.desc_act, group_size=self.group_size
        )
        logger.info("Packing model...")
        layers = get_layers(model)
        layers = {n: layers[n] for n in quantizers}
        self._replace_by_quant_layers(model, quantizers)
        qlayers = get_layers(model, [QuantLinear])
        for name in qlayers:
            logger.info(name)
            quantizers[name], scale, zero, g_idx = quantizers[name]
            # so far can only pack layer on CPU
            layer_device = qlayers[name].device
            qlayers[name].to("cpu")
            layers[name], scale, zero, g_idx = layers[name].to("cpu"), scale.to("cpu"), zero.to("cpu"), g_idx.to("cpu")
            qlayers[name].pack(layers[name], scale, zero, g_idx)
            qlayers[name].to(layer_device)

        logger.info("Model packed.")

    def save(self, model, save_dir, max_shard_size="10GB", safe_serialization = False):
        """Save quantized model and configs to local disk"""
        # TODO: Modify save_pretrained from transformers to save gptq quantized model
        os.makedirs(save_dir, exist_ok=True)
        if not model._gptq_is_quantized:
            raise EnvironmentError("can only save quantized model, please execute .quantize first.")
        model = model.to('cpu')
        # save model and config
        accelerator = Accelerator()
        accelerator.save_model(model, save_dir, max_shard_size=max_shard_size, safe_serialization=max_shard_size)
        with open(os.path.join(save_dir, GPTQ_CONFIG), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        # Specific to transformers but I will keep it here for now
        if hasattr(model,"config"):
            model.config.save_pretrained(save_dir)

def load_quantized_model(model, 
                        save_folder=None,
                        device_map=None,
                        max_memory=None,
                        no_split_module_classes=None,
                        offload_folder=None,
                        offload_buffers=None,
                        offload_state_dict=None
                        ):
    
    if device_map is None:
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        else:
            raise RuntimeError("No GPU found. A GPU is needed to run quantized model.")
        logger.info("The device_map was not initialized." "Setting device_map to `{'':torch.cuda.current_device()}`.")
    
    with open(os.path.join(save_folder,GPTQ_CONFIG), "r", encoding="utf-8") as f:
        quantize_config_dict = json.load(f)
    quantizer = GPTQQuantizer.from_dict(quantize_config_dict)
    
    block_name = quantizer.block_name_to_quantize
    layers_to_be_replaced = get_layers(model,prefix=block_name)
    quantizer._replace_by_quant_layers(model,layers_to_be_replaced)
    if no_split_module_classes is None:
        block_class_name = get_module_by_name_prefix(model,block_name)[0].__class__.__name__
        no_split_module_classes = [block_class_name]

    # specific to transformers
    # make sure that the weights are tied
    model.tie_weights()
    
    model = load_checkpoint_and_dispatch(model,
                                        checkpoint=save_folder,
                                        device_map=device_map,
                                        max_memory=max_memory,
                                        no_split_module_classes=no_split_module_classes,
                                        offload_folder=offload_folder,
                                        offload_buffers=offload_buffers,
                                        offload_state_dict=offload_state_dict
                                        )
    # put on eval mode
    model.eval()
    if quantizer.warmup_triton:
        QuantLinear = dynamically_import_QuantLinear(
                use_triton=quantizer.use_triton, desc_act=quantizer.desc_act, group_size=quantizer.group_size
            )
        QuantLinear.warmup(model, seqlen=quantizer.model_seqlen)
    return model