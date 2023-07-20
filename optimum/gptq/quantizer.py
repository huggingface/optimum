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
import copy
import json
import os
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from ..utils import is_accelerate_available, is_autogptq_available
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import get_block_name, get_device, get_layers, get_module_by_name_prefix, get_preceding_modules, get_seqlen


if is_accelerate_available():
    from accelerate import Accelerator, load_checkpoint_and_dispatch

if is_autogptq_available():
    from auto_gptq.quantization import GPTQ
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

logger = getLogger(__name__)


class GPTQQuantizer(object):
    r"""
    A simple API for GTPQ Quantization
    """

    def __init__(
        self,
        bits: int,
        group_size: int = 128,
        damp_percent: float = 0.01,
        desc_act: bool = True,
        sym: bool = True,
        true_sequential: bool = True,
        pack_sequentially: bool = False,
        use_cuda_fp16: bool = True,
        model_seqlen: Optional[int] = None,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            bits (`int`):
                The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
            group_size (int, *optional*, defaults to -1):
                The group size to use for quantization. Recommended value is 128 and -1 uses full row.
            damp_percent (`float`, *optional*, defaults to `0.01`):
                The percent of the average Hessian diagonal to use for dampening, recommended value is 0.01.
            desc_act (`bool`, *optional*, defaults to `True`):
                Whether to quantize columns in order of decreasing activation size.
                Setting it to False can significantly speed up inference but the perplexity may become slightly worse.
                Also known as act-order.
            sym (`bool`, *optional*, defaults to `True`):
                Whether to use symetric quantization.
            true_sequential (`bool`, *optional*, defaults to `True`):
                Whether to performing sequential quantization even within a single Transformer block.
            pack_sequentially (`bool`, *optional*, defaults to `True`):
                Whether to pack the layer just after it is quantized. If False, we will pack the model at the end.
            use_cuda_fp16 (`bool`, *optional*, defaults to `True`):
                Whether or not to use optmized cuda kernel for fp16 model. Need to have model in fp16.
            model_seqlen (`int`, *optional*, defaults to `None`):
                The model sequence length
            block_name_to_quantize (`str`, *optional*, defaults to `None`):
                The transformers block name to quantize.
            module_name_preceding_first_block (`str`, *optional*, defaults to `None`):
                The layers that are preceding the first Transformer block.
        """

        self.bits = bits
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.pack_sequentially = pack_sequentially
        self.use_cuda_fp16 = use_cuda_fp16
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block

        if self.bits not in [2, 4, 6, 8]:
            raise ValueError("only support quantize to [2,4,6,8] bits.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")
        for boolean in [
            "desc_act",
            "sym",
            "true_sequential",
            "pack_sequentially",
            "use_cuda_fp16",
        ]:
            if not isinstance(getattr(self, boolean), bool):
                raise ValueError(f"{boolean} must be a float")
        if self.model_seqlen is not None and not isinstance(self.model_seqlen, int):
            raise ValueError("model_seqlen must be an int")
        if self.block_name_to_quantize is not None and not isinstance(self.block_name_to_quantize, str):
            raise ValueError("block_name_to_quantize must be a string")
        if self.module_name_preceding_first_block is not None and not isinstance(
            self.module_name_preceding_first_block, list
        ):
            raise ValueError("block_name_to_quantize must be a list of string")

    def to_dict(self):
        """
        Return the args in dict format
        """
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `GPTQQuantizer` using config_dict as kwargs

        Args:
            config_dict (`Dict[str,Any]`):
                quantization config

        Returns:
            `GPTQQuantizer`:  The quantizer object instantiated from those parameters.
        """
        return cls(**config_dict)

    def _replace_by_quant_layers(self, module: nn.Module, names: List[str], name: str = ""):
        """
        Replace linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, *optional*, defaults to `""`):
                To keep track of the name of the current module
        """
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False, desc_act=self.desc_act, group_size=self.group_size
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
                if not (self.desc_act) or self.group_size == -1:
                    new_layer = QuantLinear(
                        self.bits, self.group_size, in_features, out_features, True, use_cuda_fp16=self.use_cuda_fp16
                    )
                else:
                    new_layer = QuantLinear(self.bits, self.group_size, in_features, out_features, True)
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_layers(child, names, name + "." + name1 if name != "" else name1)

    @torch.no_grad()
    def quantize_model(
        self,
        model: nn.Module,
        tokenizer: Any,
        dataset: Union[List[str], str],
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
    ):
        """
        Quantize the model using the dataset

        Args:
            model (`nn.Module`):
                The model to quantize
            dataset (`Union[List[str],str]`):
                The dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used
                in the paper ['wikitext2','c4'].
            tokenizer (`Any`, defaults to `None`):
                The tokenizer to use in order to prepare the dataset
            batch_size (`Optional[int]`, *optional*, defaults to `1`):
                The batch size of the dataset
            pad_token_id (`Optional[int]`, *optional*, defaults to `None`):
                The pad token id. Needed to prepare the dataset when `batch_size` > 1.

        Returns:
            `nn.Module`: The quantized model
        """

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed to quantize model.")

        model.eval()

        # For Transformer model
        has_config = False
        if hasattr(model, "config"):
            has_config = True
        if hasattr(model,"hf_device_map"):
            # If the model has a device_map, we don't move to model. We have already dispatch the hook that will do the work
            has_device_map = True
        device = get_device(model)
        
        if has_config:
            use_cache = model.config.use_cache
            model.config.use_cache = False

        if self.model_seqlen is None:
            self.model_seqlen = get_seqlen(model)

        # Step 1: Prepare the data
        if isinstance(dataset, str):
            dataset, _ = get_dataset(dataset, tokenizer, seqlen=self.model_seqlen)
        elif isinstance(dataset, list):
            dataset = [tokenizer(data, return_tensors="pt") for data in dataset]
        dataset = prepare_dataset(dataset, pad_token_id=pad_token_id, batch_size=batch_size)

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

        if not has_device_map:
            # put modules from module_name_preceding_first_block on cuda
            for module_name in self.module_name_preceding_first_block:
                module = get_module_by_name_prefix(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} was not found in model")
                module = module.to(0)
            # get inputs by running self.module_name_preceding_first_block + first block on gpu
            blocks[0] = blocks[0].to(0)
    
        blocks[0] = Catcher(blocks[0])
        for data in dataset:
            for k, v in data.items():
                # put the data on gpu, we won't put them back to cpu
                data[k] = v.to(0)
            try:
                model(**data)
            except ValueError:
                pass
        blocks[0] = blocks[0].module
        if not has_device_map:
            blocks[0].to(device)
            for module_name in self.module_name_preceding_first_block:
                module = get_module_by_name_prefix(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} was not found in model")

        torch.cuda.empty_cache()

        # Step 3: Quantize the blocks
        quantizers = {}
        # start quantizing the blocks
        for i, block in enumerate(blocks):
            logger.info(f"Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}")
            # move block to cuda if needed
            if not has_device_map:
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
                    gptq[name].quantizer.configure(bits=self.bits, sym=self.sym, perchannel=True)

                    def add_batch(name):
                        def tmp(_, input, output):
                            gptq[name].add_batch(input[0].data, output.data)

                        return tmp
                    # TODO : need to rework on these hooks if we use a device_map 
                    # because it adding a hook will replace the old one. 
                    handles.append(subset_layers[name].register_forward_hook(add_batch(name)))
                # update Hessian for each layer in subset_layers thanks to the hook
                for j in range(len(dataset)):
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
                        pass
                        quantizers[f"{self.block_name_to_quantize}.{i}.{name}"] = (
                            gptq[name].quantizer,
                            scale,
                            zero,
                            g_idx,
                        )
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
                        device_layer = get_device(quantized_layer)
                        quantized_layer = quantized_layer.to("cpu")
                        quantized_layer.pack(subset_layers[name], scale, zero, g_idx)
                        quantized_layer = quantized_layer.to(device_layer)
                    gptq[name].free()
                del subset_layers
            # we get the new output from the partial quantized block
            for j in range(len(dataset)):
                layer_output = block(layer_inputs[j], **layer_input_kwargs[j])[0]
                layer_outputs.append(layer_output)

            # put back to device
            if not has_device_map:
                blocks[i] = block.to(device)
            del layers
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        # Step 4 (Optional) : Pack the model at the end (Replacing the layers)
        # if we pack the model at the end
        if not self.pack_sequentially:
            self.pack_model(model=model, quantizers=quantizers)

        model._is_quantized_gptq = True
        if has_config:
            model.config.use_cache = use_cache

        torch.cuda.empty_cache()
        return model

    def pack_model(
        self,
        model: nn.Module,
        quantizers: Dict[str, Tuple],
    ):
        """
        Pack the model by replacing the layers by quantized layers

        Args:
            model (`nn.Module`):
                _description_
            quantizers (`Dict[str,Tuple]`):
                _description_
        """
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False, desc_act=self.desc_act, group_size=self.group_size
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

    def save(self, model: nn.Module, save_dir: str, max_shard_size: str = "10GB", safe_serialization: bool = False):
        """
        Save model state dict and configs

        Args:
            model (`nn.Module`):
                Model to be saved. The model can be wrapped or unwraped.
            save_dir (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`str`, *optional*, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        """

        if not is_accelerate_available():
            raise RuntimeError(
                "You need to install accelerate in order to save a quantized model. You can do it with `pip install accelerate`"
            )

        os.makedirs(save_dir, exist_ok=True)
        if not model._is_quantized_gptq:
            raise EnvironmentError("can only save quantized model, please execute .quantize first.")
        model = model.to("cpu")
        # save model and config
        accelerator = Accelerator()
        accelerator.save_model(model, save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
        with open(os.path.join(save_dir, GPTQ_CONFIG), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        # TODO: remove that when the integration with transformers is done
        if hasattr(model, "config"):
            model.config.save_pretrained(save_dir)


def load_quantized_model(
    model: nn.Module,
    save_folder: str,
    quant_config_name: Optional[str] = GPTQ_CONFIG,
    state_dict_name: Optional[str] = None,
    device_map: Optional[str] = None,
    max_memory: Optional[Dict] = None,
    no_split_module_classes: Optional[Dict] = None,
    offload_folder: Optional[str] = None,
    offload_buffers: Optional[str] = None,
    offload_state_dict: bool = False,
):
    """
    Load quantized weights from the save_folder into the converted model and dispatch the weights according to the device_map.

    Args:
        model (`nn.Module`):
            The model can be enpty or not.
        save_folder (`Optional[str]`, *optional*, defaults to `None`):
            Directory to which to load the weights.
        quant_config_name (`Optional[str]`, *optional*, defaults to `GPTQ_CONFIG`):
            Name of the quantization config file
        state_dict_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the state dict file
        device_map (`Optional[str]`, *optional*, defaults to `None`):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`.
        max_memory (`Optional[Dict]`, *optional*, defaults to `None`):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`Optional[Dict]`, *optional*, defaults to `None`):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`Optional[str]`, *optional*, defaults to `None`):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`Optional[str]`, *optional*, defaults to `None`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.

    Returns:
        `nn.Module`: The quantized model
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. A GPU is needed to run quantized model.")
    if not is_accelerate_available():
        raise RuntimeError(
            "You need to install accelerate in order to load and dispatch weights to"
            "a quantized model. You can do it with `pip install accelerate`"
        )
    if device_map is None:
        device_map = {"": torch.cuda.current_device()}
        logger.info("The device_map was not initialized." "Setting device_map to `{'':torch.cuda.current_device()}`.")

    with open(os.path.join(save_folder, quant_config_name), "r", encoding="utf-8") as f:
        quantize_config_dict = json.load(f)
    quantizer = GPTQQuantizer.from_dict(quantize_config_dict)

    if quantizer.block_name_to_quantize is None:
        quantizer.block_name_to_quantize = get_block_name(model)
    block_name = quantizer.block_name_to_quantize

    layers_to_be_replaced = get_layers(model, prefix=block_name)
    quantizer._replace_by_quant_layers(model, layers_to_be_replaced)
    if no_split_module_classes is None:
        block_class_name = get_module_by_name_prefix(model, block_name)[0].__class__.__name__
        no_split_module_classes = [block_class_name]

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(save_folder, state_dict_name) if state_dict_name is not None else save_folder,
        device_map=device_map,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
        offload_folder=offload_folder,
        offload_buffers=offload_buffers,
        offload_state_dict=offload_state_dict,
    )
    # put on eval mode
    model.eval()
    return model
