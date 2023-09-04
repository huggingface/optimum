# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and GPTQ and AutoGPTQ authors.
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
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod

from ..utils import is_accelerate_available, is_auto_gptq_available
from ..utils.modeling_utils import recurse_getattr
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import get_block_name_with_pattern, get_device, get_layers, get_preceding_modules, get_seqlen


if is_accelerate_available():
    from accelerate import (
        Accelerator,
        cpu_offload_with_hook,
        load_checkpoint_and_dispatch,
    )
    from accelerate.hooks import remove_hook_from_module

if is_auto_gptq_available():
    from auto_gptq import exllama_set_max_input_length
    from auto_gptq.modeling._utils import autogptq_post_init
    from auto_gptq.quantization import GPTQ
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

logger = getLogger(__name__)


class GPTQQuantizer(object):
    r"""
    A simple API for GPTQ Quantization
    """

    def __init__(
        self,
        bits: int,
        dataset: Optional[Union[List[str], str]] = None,
        group_size: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        sym: bool = True,
        true_sequential: bool = True,
        use_cuda_fp16: bool = False,
        model_seqlen: Optional[int] = None,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
        disable_exllama: bool = False,
        max_input_length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            bits (`int`):
                The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
            dataset (`Union[List[str],str]`, defaults to None):
                The dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used
                in GPTQ paper ['wikitext2','c4','c4-new','ptb','ptb-new'].
            group_size (int, defaults to 128):
                The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
            damp_percent (`float`, defaults to `0.1`):
                The percent of the average Hessian diagonal to use for dampening, recommended value is 0.1.
            desc_act (`bool`, defaults to `False`):
                Whether to quantize columns in order of decreasing activation size.
                Setting it to False can significantly speed up inference but the perplexity may become slightly worse.
                Also known as act-order.
            sym (`bool`, defaults to `True`):
                Whether to use symetric quantization.
            true_sequential (`bool`, defaults to `True`):
                Whether to perform sequential quantization even within a single Transformer block.
                Instead of quantizing the entire block at once, we perform layer-wise quantization.
                As a result, each layer undergoes quantization using inputs that have passed through the previously quantized layers.
            use_cuda_fp16 (`bool`, defaults to `False`):
                Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16.
            model_seqlen (`Optional[int]`, defaults to `None`):
                The maximum sequence length that the model can take.
            block_name_to_quantize (`Optional[str]`, defaults to `None`):
                The transformers block name to quantize.
            module_name_preceding_first_block (`Optional[List[str]]`, defaults to `None`):
                The layers that are preceding the first Transformer block.
            batch_size (`int`, defaults to `1`):
                The batch size of the dataset
            pad_token_id (`Optional[int]`, defaults to `None`):
                The pad token id. Needed to prepare the dataset when `batch_size` > 1.
            disable_exllama (`bool`, defaults to `False`):
                Whether to use exllama backend. Only works with `bits` = 4.
            max_input_length (`Optional[int]`, defaults to `None`):
                The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
                It is specific to the exllama backend with act-order.
        """

        self.bits = bits
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.use_cuda_fp16 = use_cuda_fp16
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.disable_exllama = disable_exllama
        self.max_input_length = max_input_length
        self.quant_method = QuantizationMethod.GPTQ

        if self.bits not in [2, 3, 4, 8]:
            raise ValueError("only support quantize to [2,3,4,8] bits.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def to_dict(self):
        """
        Returns the args in dict format.
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

    def convert_model(self, model: nn.Module):
        """
        Convert the model to a GPTQ model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted

        """
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = get_layers(model, prefix=block_name)
        self._replace_by_quant_layers(model, layers_to_be_replaced)

        return model

    def get_no_split_module_classes(self, model):
        """
        Get the modules that should not be split across multiple devices.
        Args:
            model (`nn.Module`):
                The input model
        """

        block_class_name = recurse_getattr(model, self.block_name_to_quantize)[0].__class__.__name__
        no_split_module_classes = [block_class_name]
        return no_split_module_classes

    def _replace_by_quant_layers(self, module: nn.Module, names: List[str], name: str = ""):
        """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=self.desc_act,
            group_size=self.group_size,
            bits=self.bits,
            disable_exllama=self.disable_exllama,
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
    def quantize_model(self, model: nn.Module, tokenizer: Any):
        """
        Quantizes the model using the dataset

        Args:
            model (`nn.Module`):
                The model to quantize
            tokenizer (`Any`):
                The tokenizer to use in order to prepare the dataset. You can pass either:
                    - A custom tokenizer object.
                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        Returns:
            `nn.Module`: The quantized model
        """

        if not is_auto_gptq_available():
            raise RuntimeError("auto-gptq is required in order to perform quantzation : `pip install auto-gptq`")
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed to quantize model.")

        model.eval()

        # For Transformer model
        has_config = False
        has_device_map = False
        if hasattr(model, "config"):
            has_config = True
            use_cache = model.config.use_cache
            model.config.use_cache = False

        if hasattr(model, "hf_device_map"):
            devices = list(model.hf_device_map.values())
            if "disk" in devices:
                raise ValueError("disk offload is not supported with GPTQ quantization")
            if "cpu" in devices and len(model.hf_device_map) > 1:
                logger.info("Cpu offload is not recommended. There might be some issues with the memory")
                hook = None
                for name, device in model.hf_device_map.items():
                    if device == "cpu":
                        module = recurse_getattr(model, name)
                        remove_hook_from_module(module, recurse=True)
                        module, hook = cpu_offload_with_hook(module, prev_module_hook=hook)
            # If the model has a device_map, we don't move to model. We have already dispatched the hook that will do the work
            has_device_map = True

        if hasattr(model, "dtype"):
            self.use_cuda_fp16 = model.dtype == torch.float16

        if self.model_seqlen is None:
            self.model_seqlen = get_seqlen(model)

        device = get_device(model)

        # Step 1: Prepare the data
        if isinstance(tokenizer, str):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    f"""We were not able to get the tokenizer using `AutoTokenizer.from_pretrained`
                    with the string that you have passed {tokenizer}. If you have a custom tokenizer, you can pass it as input.
                    For now, we only support quantization for text model. Support for vision, speech and multimodel will come later."""
                )
        if self.dataset is None:
            raise ValueError("You need to pass `dataset` in order to quantize your model")
        elif isinstance(self.dataset, str):
            dataset = get_dataset(self.dataset, tokenizer, seqlen=self.model_seqlen, split="train")
        elif isinstance(self.dataset, list):
            dataset = [tokenizer(data, return_tensors="pt") for data in self.dataset]
        else:
            raise ValueError("You need to pass a list of string or a string for `dataset`")

        dataset = prepare_dataset(dataset, pad_token_id=self.pad_token_id, batch_size=self.batch_size)

        # Step 2: get the input of the 1st block
        # To do that, we need to put the modules preceding the first block on the same device as the first bloc.
        # Then we run the model and it will stop at the first bloc as we added a prehook that raise an Exception after storing the inputs.

        layer_inputs = []
        layer_outputs = []
        layer_input_kwargs = []

        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)

        if self.module_name_preceding_first_block is None:
            self.module_name_preceding_first_block = get_preceding_modules(model, self.block_name_to_quantize)

        blocks = recurse_getattr(model, self.block_name_to_quantize)

        if not has_device_map:
            # put modules from module_name_preceding_first_block on cuda
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} was not found in model")
                module = module.to(0)
            blocks[0] = blocks[0].to(0)

        def store_input_hook(_, input, *args):
            kwargs = args[0]
            input = input[0]
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

        handle = blocks[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for data in dataset:
            for k, v in data.items():
                # put the data on gpu, we won't put them back to cpu
                data[k] = v.to(0)
            try:
                model(**data)
            except ValueError:
                pass

        handle.remove()
        if not has_device_map:
            blocks[0].to(device)
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} was not found in model")

        torch.cuda.empty_cache()

        # Step 3: Quantize the blocks
        quantizers = {}
        for i, block in enumerate(tqdm(blocks, desc=f"Quantizing {self.block_name_to_quantize} blocks ")):
            logger.info(f"Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}")
            # move block to cuda if needed
            # in case we have offload modules, we need to put them on cuda because of GPTQ object
            if not has_device_map or get_device(block) == torch.device("cpu"):
                block = block.to(0)
            layers = get_layers(block)
            if self.true_sequential:
                # lazy sequential but works well
                layers_name_list = [[key] for key in layers.keys()]
            else:
                layers_name_list = [list(layers.keys())]
            logger.info(f"Module to quantize {layers_name_list}")
            for subset_name_list in tqdm(layers_name_list, leave=False, desc="Quantizing layers inside the block"):
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
                    quantizers[f"{self.block_name_to_quantize}.{i}.{name}"] = (
                        gptq[name].quantizer,
                        scale,
                        zero,
                        g_idx,
                    )
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

        if self.bits == 4 and not self.disable_exllama:
            if device == torch.device("cpu") or (has_device_map and any(d in devices for d in ["cpu", "disk"])):
                logger.warning(
                    "Found modules on cpu/disk. Using Exllama backend requires all the modules to be on GPU. Setting `disable_exllama=True`"
                )
                self.disable_exllama = True
            elif self.desc_act:
                logger.warning(
                    "Using Exllama backend with act_order will reorder the weights offline, thus you will not be able to save the model with the right weights."
                    "Setting `disable_exllama=True`. You should only use Exllama backend with act_order for inference. "
                )
                self.disable_exllama = True
        # Step 4: Pack the model at the end (Replacing the layers)
        self.pack_model(model=model, quantizers=quantizers)

        model.is_quantized = True
        model.quantization_method = QuantizationMethod.GPTQ
        if has_config:
            model.config.use_cache = use_cache
            model.config.quantization_config = self.to_dict()

        # Step 5: Any post-initialization that require device information, for example buffers initialization on device.
        model = self.post_init_model(model)

        torch.cuda.empty_cache()
        return model

    def post_init_model(self, model):
        """
        Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """
        if self.bits == 4 and not self.disable_exllama:
            if get_device(model) == torch.device("cpu") or (
                hasattr(model, "hf_device_map") and any(d in model.hf_device_map for d in ["cpu", "disk"])
            ):
                raise ValueError(
                    "Found modules on cpu/disk. Using Exllama backend requires all the modules to be on GPU."
                    "You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object"
                )

        class StoreAttr(object):
            pass

        model.quantize_config = StoreAttr()
        model.quantize_config.desc_act = self.desc_act
        model = autogptq_post_init(model, use_act_order=self.desc_act)
        if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
            model = exllama_set_max_input_length(model, self.max_input_length)
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
                The model to pack
            quantizers (`Dict[str,Tuple]`):
                A mapping of the layer name and the data needed to pack the layer
        """
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=self.desc_act,
            group_size=self.group_size,
            bits=self.bits,
            disable_exllama=self.disable_exllama,
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
            max_shard_size (`str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>
            safe_serialization (`bool`, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        """

        if not is_accelerate_available():
            raise RuntimeError(
                "You need to install accelerate in order to save a quantized model. You can do it with `pip install accelerate`"
            )

        os.makedirs(save_dir, exist_ok=True)
        # save model and config
        accelerator = Accelerator()
        accelerator.save_model(model, save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
        with open(os.path.join(save_dir, GPTQ_CONFIG), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_quantized_model(
    model: nn.Module,
    save_folder: str,
    quant_config_name: str = GPTQ_CONFIG,
    state_dict_name: Optional[str] = None,
    device_map: Optional[str] = None,
    max_memory: Optional[Dict] = None,
    no_split_module_classes: Optional[Dict] = None,
    offload_folder: Optional[str] = None,
    offload_buffers: Optional[str] = None,
    offload_state_dict: bool = False,
    disable_exllama: bool = False,
    max_input_length: Optional[int] = None,
):
    """
    Load quantized weights from the save_folder into the converted model and dispatch the weights according to the device_map.

    Args:
        model (`nn.Module`):
            The model can be enpty or not.
        save_folder (`str`):
            Directory to which to load the weights.
        quant_config_name (`str`, defaults to `GPTQ_CONFIG`):
            Name of the quantization config file
        state_dict_name (`Optional[str]`, defaults to `None`):
            Name of the state dict file
        device_map (`Optional[str]`, defaults to `None`):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`.
        max_memory (`Optional[Dict]`, defaults to `None`):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`Optional[Dict]`, defaults to `None`):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`Optional[str]`, defaults to `None`):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`Optional[str]`, defaults to `None`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        offload_state_dict (`bool`, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        disable_exllama (`bool`, defaults to `False`):
            Whether to use exllama backend. Only works with `bits` = 4.
        max_input_length (`Optional[int]`, defaults to `None`):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
            It is specific to the exllama backend with act-order.

    Returns:
        `nn.Module`: The quantized model
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. A GPU is needed to run quantized model.")
    if not is_auto_gptq_available():
        raise RuntimeError("auto-gptq is required in order to load quantized weights : `pip install auto-gptq`")
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
    quantizer.disable_exllama = disable_exllama
    quantizer.max_input_length = max_input_length

    model = quantizer.convert_model(model)

    if no_split_module_classes is None:
        no_split_module_classes = quantizer.get_no_split_module_classes(model)

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

    model = quantizer.post_init_model(model)
    model.is_quantized = True
    model.quantization_method = QuantizationMethod.GPTQ
    model.eval()
    return model
