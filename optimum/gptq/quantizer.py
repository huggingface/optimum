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
import importlib
import json
import os
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod

from ..utils import is_accelerate_available, is_auto_gptq_available, is_gptqmodel_available
from ..utils.modeling_utils import recurse_getattr
from ..version import __version__ as optimum_version
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import (
    get_block_name_with_pattern,
    get_device,
    get_layers,
    get_preceding_modules,
    get_seqlen,
    nested_move_to,
)


if is_accelerate_available():
    from accelerate import (
        cpu_offload_with_hook,
        load_checkpoint_and_dispatch,
    )
    from accelerate.hooks import remove_hook_from_module

if is_auto_gptq_available():
    from auto_gptq import __version__ as autogptq_version
    from auto_gptq import exllama_set_max_input_length
    from auto_gptq.modeling._utils import autogptq_post_init as gptq_post_init
    from auto_gptq.quantization import GPTQ
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear as hf_select_quant_linear

if is_gptqmodel_available():
    from gptqmodel import exllama_set_max_input_length
    from gptqmodel.quantization import GPTQ
    from gptqmodel.utils.importer import hf_select_quant_linear
    from gptqmodel.utils.model import hf_convert_gptq_v1_to_v2_format, hf_convert_gptq_v2_to_v1_format
    from gptqmodel.utils.model import hf_gptqmodel_post_init as gptq_post_init
    from gptqmodel.version import __version__ as gptqmodel_version

logger = getLogger(__name__)


def has_device_more_than_cpu():
    return torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())


class ExllamaVersion(int, Enum):
    ONE = 1
    TWO = 2


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
        exllama_config: Optional[Dict[str, Any]] = None,
        max_input_length: Optional[int] = None,
        cache_block_outputs: Optional[bool] = True,
        modules_in_block_to_quantize: Optional[List[List[str]]] = None,
        checkpoint_format: str = "gptq",
        meta: Optional[Dict[str, any]] = None,
        backend: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            bits (`int`):
                The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
            dataset (`Union[List[str], str, Any]`, defaults to `None`):
                The dataset used for quantization. You can provide your own dataset in a list of string or in a list of tokenized data
                (e.g. [{ "input_ids": [ 1, 100, 15, ... ],"attention_mask": [ 1, 1, 1, ... ]},...])
                or just use the original datasets used in GPTQ paper ['wikitext2','c4','c4-new'].
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
                The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
            module_name_preceding_first_block (`Optional[List[str]]`, defaults to `None`):
                The layers that are preceding the first Transformer block.
            batch_size (`int`, defaults to `1`):
                The batch size of the dataset
            pad_token_id (`Optional[int]`, defaults to `None`):
                The pad token id. Needed to prepare the dataset when `batch_size` > 1.
            disable_exllama (`bool`, defaults to `False`):
                Whether to use exllama backend. Only works with `bits` = 4.
            exllama_config (`Dict[str, Any]`, *optional*):
                The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults to `{"version": 2}` if unset.
            max_input_length (`Optional[int]`, defaults to `None`):
                The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
                It is specific to the exllama backend with act-order.
            cache_block_outputs (`bool`, defaults to `True`):
                Whether to cache block outputs to reuse as inputs for the succeeding block. It allows optimization of non-standard models
                (e.g. ChatGLM) but can require more time.
            modules_in_block_to_quantize (`Optional[List[List[str]]]`, defaults to `None`):
                List list of module names to quantize in the block specified. This argument is useful to exclude certain linear modules from being quantized.
                The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially.
                If not set, we will quantize all linear layers. Example: `inside_layer_modules=[["self_attention.query_key_value"], ["mlp.dense_h_to_4h"]]`
            checkpoint_format (`str`, *optional*, defaults to `gptq`):
                GPTQ weight format. `gptq`(v1) is supported by both gptqmodel and auto-gptq. `gptq_v2` is gptqmodel only.
            meta (`Dict[str, any]`, *optional*):
                Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta.
                i.e. `meta.quantizer`: ["optimum:_version_", "gptqmodel:_version_"]
            backend (`str`, *optional*):
                Controls which gptq kernel to be used. Valid values for gptqmodel are `auto`, `auto_trainable` and more. For auto-gptq, only valid value is None and `auto_trainable`. Ref gptqmodel backends: https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py
        """

        self.bits = bits
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.checkpoint_format = checkpoint_format.lower()
        self.meta = meta
        self.backend = backend.lower() if backend is not None else None
        self.use_cuda_fp16 = use_cuda_fp16
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.disable_exllama = disable_exllama
        self.exllama_config = exllama_config
        self.max_input_length = max_input_length
        self.quant_method = QuantizationMethod.GPTQ
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize

        self.serialization_keys = [
            "bits",
            "dataset",
            "group_size",
            "damp_percent",
            "desc_act",
            "sym",
            "true_sequential",
            "quant_method",
            "modules_in_block_to_quantize",
            "checkpoint_format",
            "meta",
        ]

        if self.bits not in [2, 3, 4, 8]:
            raise ValueError("only support quantize to [2,3,4,8] bits.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

        if self.exllama_config is None:
            self.exllama_config = {"version": ExllamaVersion.TWO}
        else:
            if "version" not in self.exllama_config:
                raise ValueError("`exllama_config` needs to have a `version` key")
            elif self.exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
                version = self.exllama_config["version"]
                raise ValueError(
                    f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {version}"
                )
        self.exllama_version = self.exllama_config["version"]

    def select_quant_linear(self, device_map: Union[str, dict], pack: bool = False):
        if is_gptqmodel_available():
            self.quant_linear = hf_select_quant_linear(
                bits=self.bits,
                group_size=self.group_size,
                desc_act=self.desc_act,
                sym=self.sym,
                checkpoint_format=self.checkpoint_format,
                meta=self.meta,
                device_map=device_map,
                backend=self.backend,
                pack=pack,
            )
        else:
            self.quant_linear = hf_select_quant_linear(
                use_triton=False,
                desc_act=self.desc_act,
                group_size=self.group_size,
                bits=self.bits,
                disable_exllama=self.disable_exllama or self.exllama_version != ExllamaVersion.ONE,
                disable_exllamav2=self.disable_exllama or self.exllama_version != ExllamaVersion.TWO,
            )

    def to_dict(self):
        """
        Returns the args in dict format.
        """
        gptq_dict = {}
        for key in self.serialization_keys:
            gptq_dict[key] = getattr(self, key)

        if gptq_dict.get("meta") is None:
            gptq_dict["meta"] = {}

        meta = gptq_dict["meta"]
        # store both optimum:version and gptq_lib:version into quantize_config.meta.quantizer
        if meta.get("quantizer") is None:
            meta["quantizer"] = [f"optimum:{optimum_version}"]

            if is_gptqmodel_available():
                meta["quantizer"].append(f"gptqmodel:{gptqmodel_version}")
            elif is_auto_gptq_available():
                meta["quantizer"].append(f"auto_gptq:{autogptq_version}")

        return gptq_dict

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

    def convert_model(self, model: nn.Module, **kwargs):
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
        if self.modules_in_block_to_quantize is not None:
            layers_to_keep = sum(self.modules_in_block_to_quantize, [])
            for name in list(layers_to_be_replaced.keys()):
                if not any(name.endswith(layer) for layer in layers_to_keep):
                    logger.info(
                        f"Quantization disabled for {name} (only modules_in_block_to_quantize={self.modules_in_block_to_quantize} are quantized)"
                    )
                    del layers_to_be_replaced[name]

        self.select_quant_linear(device_map=kwargs.get("device_map", None), pack=False)

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
        if isinstance(module, self.quant_linear):
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
                bias = layer.bias is not None
                if is_gptqmodel_available():
                    new_layer = self.quant_linear(
                        self.bits,
                        self.group_size,
                        self.desc_act,
                        self.sym,
                        in_features,
                        out_features,
                        bias,
                        weight_dtype=layer.weight.dtype,
                    )
                else:
                    if not (self.desc_act) or self.group_size == -1:
                        new_layer = self.quant_linear(
                            self.bits,
                            self.group_size,
                            in_features,
                            out_features,
                            bias,
                            use_cuda_fp16=self.use_cuda_fp16,
                            weight_dtype=layer.weight.dtype,
                        )
                    else:
                        new_layer = self.quant_linear(
                            self.bits,
                            self.group_size,
                            in_features,
                            out_features,
                            bias,
                            weight_dtype=layer.weight.dtype,
                        )
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_layers(child, names, name + "." + name1 if name != "" else name1)

    @torch.no_grad()
    def quantize_model(self, model: nn.Module, tokenizer: Optional[Any] = None):
        """
        Quantizes the model using the dataset

        Args:
            model (`nn.Module`):
                The model to quantize
            tokenizer (Optional[`Any`], defaults to `None`):
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

        if not is_auto_gptq_available() and not is_gptqmodel_available():
            raise RuntimeError(
                "gptqmodel or auto-gptq is required in order to perform gptq quantzation: `pip install gptqmodel` or `pip install auto-gptq`. Please notice that auto-gptq will be deprecated in the future."
            )
        elif is_gptqmodel_available() and is_auto_gptq_available():
            logger.warning(
                "Detected gptqmodel and auto-gptq, will use gptqmodel. The auto_gptq will be deprecated in the future."
            )

        gptq_supports_cpu = (
            is_auto_gptq_available()
            and version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
        ) or is_gptqmodel_available()

        if not gptq_supports_cpu and not torch.cuda.is_available():
            raise RuntimeError(
                "No cuda gpu or cpu support using Intel/IPEX found. A gpu or cpu with Intel/IPEX is required for quantization."
            )

        if not self.sym and not is_gptqmodel_available():
            raise ValueError(
                "Asymmetric sym=False quantization is not supported with auto-gptq. Please use gptqmodel: `pip install gptqmodel`"
            )

        if self.checkpoint_format == "gptq_v2" and not is_gptqmodel_available():
            raise ValueError(
                "gptq_v2 format only supported with gptqmodel. Please install gptqmodel: `pip install gptqmodel`"
            )

        model.eval()

        # gptqmodel internal is gptq_v2 for asym support, gptq(v1) can only support sym=True
        if is_gptqmodel_available() and self.checkpoint_format != "gptq_v2":
            self.checkpoint_format = "gptq_v2"

        # For Transformer model
        has_config = False
        has_device_map = False
        if hasattr(model, "config"):
            has_config = True
            use_cache = model.config.use_cache
            model.config.use_cache = False

        # If the model has a device_map, we don't move to model. We have already dispatched the hook that will do the work
        if hasattr(model, "hf_device_map"):
            devices = list(model.hf_device_map.values())
            has_device_map = True
            if "disk" in devices:
                raise ValueError("disk offload is not supported with GPTQ quantization")
            if "cpu" in devices or torch.device("cpu") in devices:
                if len(model.hf_device_map) > 1:
                    logger.info("Cpu offload is not recommended. There might be some issues with the memory")
                    hook = None
                    for name, device in model.hf_device_map.items():
                        if device == "cpu":
                            module = recurse_getattr(model, name)
                            remove_hook_from_module(module, recurse=True)
                            module, hook = cpu_offload_with_hook(module, prev_module_hook=hook)
                else:
                    has_device_map = False

        if hasattr(model, "dtype"):
            self.use_cuda_fp16 = model.dtype == torch.float16

        if self.model_seqlen is None:
            # We allow a max value of 4028 to avoid passing data with huge length to the model during the calibration step
            self.model_seqlen = min(4028, get_seqlen(model))

        device = get_device(model)

        # Step 1: Prepare the data
        if isinstance(self.dataset, list) and not isinstance(self.dataset[0], str):
            dataset = self.dataset
            logger.info("GPTQQuantizer dataset appears to be already tokenized. Skipping tokenization.")
        else:
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
                raise ValueError(
                    f"You need to pass a list of string, a list of tokenized data or a string for `dataset`. Found: {type(self.dataset)}."
                )

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

        cur_layer_device = get_device(blocks[0])
        if not is_gptqmodel_available() and cur_layer_device.type == "cpu":
            cur_layer_device = 0

        if not has_device_map:
            # put modules from module_name_preceding_first_block on cuda or xpu or cpu
            to_device = cur_layer_device
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} was not found in model")
                module = module.to(to_device)
            blocks[0] = blocks[0].to(to_device)

        def store_input_hook(_, input, *args):
            kwargs = args[0]
            if input is None:
                if "hidden_states" in kwargs:
                    input = (nested_move_to(kwargs["hidden_states"], cur_layer_device),)
                else:
                    raise ValueError("No input value found in the foward pass")
            layer_inputs.append(input)
            other_kwargs = {}
            for k, v in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states"]:
                    other_kwargs[k] = nested_move_to(v, cur_layer_device)
            layer_input_kwargs.append(other_kwargs)
            raise ValueError

        if self.cache_block_outputs:
            handle = blocks[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
            for data in dataset:
                for k, v in data.items():
                    data[k] = nested_move_to(v, cur_layer_device)
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
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        # Step 3: Quantize the blocks
        quantizers = {}
        for i, block in enumerate(tqdm(blocks, desc=f"Quantizing {self.block_name_to_quantize} blocks ")):
            logger.info(f"Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}")

            if not self.cache_block_outputs:
                handle = block.register_forward_pre_hook(store_input_hook, with_kwargs=True)
                for data in dataset:
                    for k, v in data.items():
                        data[k] = nested_move_to(v, cur_layer_device)
                    try:
                        model(**data)
                    except ValueError:
                        pass
                handle.remove()

            # move block to cuda if needed
            # in case we have offload modules, we need to put them on cuda because of GPTQ object
            if (not has_device_map or get_device(block) == torch.device("cpu")) and has_device_more_than_cpu():
                block = block.to(0)
            layers = get_layers(block)
            block_device = get_device(block)
            if not is_gptqmodel_available() and block_device.type == "cpu":
                block_device = 0
            if isinstance(self.modules_in_block_to_quantize, list) and len(self.modules_in_block_to_quantize) > 0:
                if self.true_sequential:
                    layers_name_list = self.modules_in_block_to_quantize
                else:
                    layers_name_list = [sum(self.modules_in_block_to_quantize, [])]
            else:
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
                    layer_inputs[j] = nested_move_to(layer_inputs[j], block_device)
                    for k, v in layer_input_kwargs[j].items():
                        layer_input_kwargs[j][k] = nested_move_to(v, block_device)

                    block(*layer_inputs[j], **layer_input_kwargs[j])
                # remove hook
                for h in handles:
                    h.remove()
                for name in subset_name_list:
                    logger.info(f"Quantizing {name} in block {i + 1}/{len(blocks)}...")
                    quant_outputs = gptq[name].fasterquant(
                        percdamp=self.damp_percent, group_size=self.group_size, actorder=self.desc_act
                    )
                    scale, zero, g_idx = quant_outputs[0], quant_outputs[1], quant_outputs[2]
                    quantizers[f"{self.block_name_to_quantize}.{i}.{name}"] = (
                        gptq[name].quantizer,
                        scale,
                        zero,
                        g_idx,
                    )
                    gptq[name].free()
                del subset_layers
            # we get the new output from the partial quantized block
            if self.cache_block_outputs:
                for j in range(len(dataset)):
                    layer_output = block(*layer_inputs[j], **layer_input_kwargs[j])
                    layer_outputs.append(layer_output)

                # put back to device
                if not has_device_map:
                    blocks[i] = block.to(device)
                del layers
                del layer_inputs
                layer_inputs, layer_outputs = layer_outputs, []
            else:
                del layers
                del layer_inputs
                layer_inputs = []
            torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

        if self.bits == 4:
            # device not on gpu
            if device.type != "cuda" or (has_device_map and any(d in devices for d in ["cpu", "disk", "hpu"])):
                if not self.disable_exllama and not is_gptqmodel_available():
                    logger.warning(
                        "Found modules on cpu/disk. Using Exllama/Exllamav2 backend requires all the modules to be on GPU. Setting `disable_exllama=True`"
                    )
                    self.disable_exllama = True
            # act order and exllama
            elif self.desc_act and not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE:
                logger.warning(
                    "Using Exllama backend with act_order will reorder the weights offline, thus you will not be able to save the model with the right weights."
                    "Setting `disable_exllama=True`. You should only use Exllama backend with act_order for inference. "
                )
                self.disable_exllama = True
            elif not self.disable_exllama and self.exllama_version == ExllamaVersion.TWO:
                logger.warning(
                    "Using Exllamav2 backend will reorder the weights offline, thus you will not be able to save the model with the right weights."
                    "Setting `disable_exllama=True`. You should only use Exllamav2 backend for inference. "
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
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        return model

    def post_init_model(self, model):
        """
        Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """
        if self.bits == 4 and not self.disable_exllama:
            if get_device(model).type != "cuda" or (
                hasattr(model, "hf_device_map") and any(d in model.hf_device_map for d in ["cpu", "disk", "hpu"])
            ):
                if not self.disable_exllama:
                    logger.warning(
                        "Found modules on cpu/disk. Using Exllama/Exllamav2 backend requires all the modules to be on GPU. Setting `disable_exllama=True`"
                    )
                    self.disable_exllama = True

        class StoreAttr(object):
            pass

        if is_gptqmodel_available():
            model, _ = hf_convert_gptq_v1_to_v2_format(
                model, self.bits, self.quant_linear, self.checkpoint_format, self.meta
            )

        model.quantize_config = StoreAttr()
        model.quantize_config.desc_act = self.desc_act
        model = gptq_post_init(model, use_act_order=self.desc_act)
        if (
            self.desc_act
            and (not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE)
            and self.max_input_length is not None
        ):
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
        logger.info("Packing model...")
        layers = get_layers(model)
        layers = {n: layers[n] for n in quantizers}

        self.select_quant_linear(device_map=model.hf_device_map, pack=True)

        self._replace_by_quant_layers(model, quantizers)
        qlayers = get_layers(model, [self.quant_linear])
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

    def save(self, model: nn.Module, save_dir: str, max_shard_size: str = "10GB", safe_serialization: bool = True):
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
            safe_serialization (`bool`, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        """

        # convert gptqmodel internal gptq_v2 format to v1 for max compatibility
        if is_gptqmodel_available():
            model, converted = hf_convert_gptq_v2_to_v1_format(
                model, self.sym, self.bits, self.quant_linear, self.checkpoint_format, self.meta
            )
            if converted:
                self.checkpoint_format = "gptq"

        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
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
    exllama_config: Optional[Dict[str, Any]] = None,
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
        disable_exllama (`Optional[bool]`, defaults to `None`):
            Whether to use exllama backend. Only works with `bits` = 4.
        exllama_config (`Optional[Dict[str, Any]]`, defaults to `None`):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults to `{"version": 2}` if unset.
        max_input_length (`Optional[int]`, defaults to `None`):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
            It is specific to the exllama backend with act-order.

    Returns:
        `nn.Module`: The quantized model
    """
    if not torch.cuda.is_available() and not is_gptqmodel_available():
        raise RuntimeError("No GPU found. A GPU is needed to run quantized model by auto_gptq.")
    if not is_auto_gptq_available() and not is_gptqmodel_available():
        raise RuntimeError(
            "gptqmodel (`pip install gptqmodel`) or auto-gptq (`pip install auto-gptq`) is required in order to load quantized weights. Please notice that auto-gptq will be deprecated in the future."
        )
    if not is_accelerate_available():
        raise RuntimeError(
            "You need to install accelerate in order to load and dispatch weights to"
            "a quantized model. You can do it with `pip install accelerate`"
        )
    if device_map is None:
        device_map = {"": torch.cuda.current_device()}
        logger.info("The device_map was not initialized." "Setting device_map to `{'':torch.cuda.current_device()}`.")

    if exllama_config is None:
        exllama_config = {"version": ExllamaVersion.TWO}
    else:
        if "version" not in exllama_config:
            raise ValueError("`exllama_config` needs to have a `version` key")
        elif exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
            version = exllama_config["version"]
            raise ValueError(
                f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {version}"
            )

    # this branch will check if model is from huggingface
    try:
        if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
            quantize_config_dict = model.config.quantization_config.to_dict()
        else:
            with open(os.path.join(save_folder, quant_config_name), "r", encoding="utf-8") as f:
                quantize_config_dict = json.load(f)
    except Exception as err:
        raise ValueError(
            f"Failed to load quantization config from {save_folder} (lookup for traceback): {err}\nTip: If the save directory is saved from a transformers.PreTrainedModel, make sure that `config.json` contains a 'quantization_config' key."
        ) from err
    quantizer = GPTQQuantizer.from_dict(quantize_config_dict)
    quantizer.disable_exllama = disable_exllama
    quantizer.exllama_config = exllama_config
    quantizer.exllama_version = quantizer.exllama_config["version"]
    quantizer.max_input_length = max_input_length

    model = quantizer.convert_model(model, device_map=device_map)

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
