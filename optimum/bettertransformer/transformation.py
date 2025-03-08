# Copyright 2022 The HuggingFace and Meta Team.  All rights reserved.
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
import logging
import os
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
from packaging.version import parse

from ..utils import (
    check_if_pytorch_greater,
    check_if_torch_greater,
    is_accelerate_available,
    recurse_getattr,
    recurse_setattr,
)
from .models import BetterTransformerManager


if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_module

ERROR_MESSAGE = r"The Better Transformers implementation for the model {model_name} has not been implemented yet. Please open an issue requesting the addition of this model with its `BetterTransformer` implementation."


def raise_save_or_push_incompatible(*_, **__):
    r"""
    Simply raise an error if the user tries to save or push a model that is not compatible with
    `BetterTransformer` and needs to be reverted to the original model before calling these
    functions.
    """
    raise ValueError(
        "You are trying to save or push a model that has been converted with `BetterTransformer`.",
        " Please revert the model to its original state before calling `save_pretrained` or `push_to_hub`.",
        " By calling model = BetterTransformer.reverse(model) before saving or pushing.",
    )


def replace_to_bettertransformer(model, config):
    r"""
    Replaces the current model to its `BetterTransformer` implementation. Loops recursively into the model and replaces the
    `Layer` modules with its `BetterTransformer` correspondant model

    - Step 1: Recurse over the modules of the model
    - Step 2: Verify if the module `BetterTransformer` is present for that model
    - Step 3: If yes, replace the `...Layer` module with the `...LayerBetterTransformer` modules
    - Step 4: If not, yield an error.
    - Step 5: Post process the potentially converted model by setting the `is_last_layer` attribute to `True` for the last `BetterTransformer` layer.
    (done in `set_last_layer` function)

    Args:
        `model` (`torch.nn.Module`):
            The input model to convert
        `config` (`transformers.PreTrainedConfig`):
            The configuration dictionary of the model
    Returns:
        The converted model
    """
    for name, module in model.named_children():
        if hasattr(module, "SCB"):
            # 8-bit modules are not supported
            raise ValueError(
                "`load_in_8bit` and `BetterTransformers` are mutually exclusive",
                " please pass a model that is not loaded in 8-bit.",
            )

        # replace the module if it is a transformer layer compatible with bettertransformer
        target_classes = list(BetterTransformerManager.MODEL_MAPPING[config.model_type].keys())

        # We may want to override methods without having to override whole modules.
        # For example, some methods handle the mask generation, which we do not need when using PyTorch SDPA.
        if config.model_type in BetterTransformerManager.OVERWRITE_METHODS:
            for class_name, method_name_and_replacement in BetterTransformerManager.OVERWRITE_METHODS[
                config.model_type
            ].items():
                if module.__class__.__name__ == class_name:
                    method_name = method_name_and_replacement[0]
                    new_method = method_name_and_replacement[1]
                    setattr(module, method_name, types.MethodType(new_method, module))

        should_replace_module = False
        for target_class in target_classes:
            should_replace_module = module.__class__.__name__ == target_class
            if should_replace_module:
                bettertransformer_module = BetterTransformerManager.MODEL_MAPPING[config.model_type][target_class](
                    module, config
                )
                model._modules[name] = bettertransformer_module
                break

        if len(list(module.children())) > 0 and should_replace_module is False:
            # we may explicitly exclude part of the model to use BetterTransformer
            if config.model_type not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM or (
                config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM
                and name not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[config.model_type]
            ):
                replace_to_bettertransformer(module, config)

    return model


def set_last_layer(model: torch.nn.Module):
    r"""
    Iterates over the module list containing the `LayerBetterTransformer` modules. Sets the last layer's `is_last_layer`
    attribute to `True`

    Args:
        `model` (`torch.nn.Module`):
            The input converted model
    Raises:
        `NotImplementedError`: Raised if this method fails, in which case the model is not supported.
    """
    dict_named_module = dict(model.named_modules())
    sort_fn = lambda list_modules: [module.__class__.__name__ for module in list_modules]  # noqa: E731

    modulelist_lengths = []

    for key in dict_named_module.keys():
        if (
            isinstance(dict_named_module[key], torch.nn.ModuleList)
            and "encoder" in key
            and (
                model.config.model_type not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM
                or (
                    model.config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM
                    and all(
                        name not in key
                        for name in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[model.config.model_type]
                    )
                )
            )
        ):
            modulelist_lengths.append((len(dict_named_module[key]), key))

    # For Albert, each transformer layer is wrapped
    # inside a ModuleList
    if len(modulelist_lengths) > 1:
        _, key = max(modulelist_lengths, key=lambda item: item[0])
        largest_module_list = dict_named_module[key]

        for module in largest_module_list[-1].modules():
            if "LayerBetterTransformer" in module.__class__.__name__:
                setattr(module, "is_last_layer", True)
                return
    else:
        for key in dict_named_module.keys():
            if isinstance(dict_named_module[key], torch.nn.ModuleList) and all(
                "LayerBetterTransformer" in module_name for module_name in sort_fn(dict_named_module[key])
            ):
                setattr(dict_named_module[key][-1], "is_last_layer", True)
                return

    raise Exception(
        f"The transformation of the model {model.__class__.__name__} to BetterTransformer failed while it should not. Please fill"
        " a bug report or open a PR to support this model at https://github.com/huggingface/optimum/"
    )


class BetterTransformer(object):
    r"""
    A conversion wrapper that takes as an input the `transformers` model to be converted
    and returns the converted `BetterTransformer` model. The `BetterTransformer` model is based on the `BetterTransformer`
    recently released by PyTorch from its 1.12 version:
    https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/

    # Original PR from: https://github.com/huggingface/transformers/pull/19553 adapted and wrapped in this script.
    """

    @check_if_pytorch_greater(
        "1.13.99",
        "Please upgrade PyTorch following https://pytorch.org/get-started/locally/ in order to use BetterTransformer.",
    )
    def transform(
        model: torch.nn.Module,
        keep_original_model: bool = False,
        max_memory: Optional[Dict] = None,
        offload_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> torch.nn.Module:
        r"""
        Conversion script from `transformers` model to its BetterTransformers version

        Args:
            model (`torch.nn.Module`):
                Original `transformers` model
            keep_original_model (`bool`, defaults to `False`):
                whether to keep or override the original model - essentially
                for memory efficiency reasons
            max_memory (`Optional[Dict]`, defaults to `None`):
                Same argument as `max_memory` argument from `.from_pretrained` function
                in `transformers`.
        Returns:
            The converted model if the conversion has been successful.
        """

        logger.warning(
            "The class `optimum.bettertransformers.transformation.BetterTransformer` is deprecated and will be removed in optimum v2.0."
        )

        hf_config = model.config
        if hf_config.model_type in ["falcon", "gpt_bigcode", "llama", "whisper"]:
            raise ValueError(
                f"Transformers now supports natively BetterTransformer optimizations (torch.nn.functional.scaled_dot_product_attention) for the model type {hf_config.model_type}. "
                "As such, there is no need to use `model.to_bettertransformers()` or `BetterTransformer.transform(model)` from the Optimum library. "
                "Please upgrade to transformers>=4.36 and torch>=2.1.1 to use it. "
                "Details: https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention."
            )

        if hasattr(hf_config, "_attn_implementation") and hf_config._attn_implementation == "sdpa":
            raise ValueError(
                "This model already uses BetterTransformer optimizations from Transformers (torch.nn.functional.scaled_dot_product_attention). "
                "As such, there is no need to use `model.to_bettertransformers()` or `BetterTransformer.transform(model)` from the Optimum library. "
                "Details: https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention."
            )

        if hasattr(model, "use_bettertransformer") and model.use_bettertransformer is True:
            raise Exception(
                "`BetterTransform.transform()` was called on a model already using Better Transformer modeling."
            )

        if BetterTransformerManager.cannot_support(model.config.model_type):
            raise ValueError(
                f"The model type {model.config.model_type} can not be supported to be used with BetterTransformer. The identified reason is:"
                f" {BetterTransformerManager.CAN_NOT_BE_SUPPORTED[model.config.model_type]}. Currently supported models are:"
                f" {BetterTransformerManager.MODEL_MAPPING.keys()}."
            )
        if not BetterTransformerManager.supports(model.config.model_type):
            raise NotImplementedError(
                f"The model type {model.config.model_type} is not yet supported to be used with BetterTransformer. Feel free"
                f" to open an issue at https://github.com/huggingface/optimum/issues if you would like this model type to be supported."
                f" Currently supported models are: {BetterTransformerManager.MODEL_MAPPING.keys()}."
            )

        if not check_if_torch_greater("2.0"):
            raise ValueError(
                f"BetterTransformer requires torch>=2.0 but {torch.__version__} is installed. Please upgrade PyTorch."
            )

        hf_config = model.config

        # Check if we have to load the model using `accelerate`
        if hasattr(model, "hf_device_map"):
            load_accelerate = True
            hf_device_map = model.hf_device_map
        else:
            load_accelerate = False

        if load_accelerate:
            # Remove the hooks from the original model to avoid weights being on `meta` device.
            remove_hook_from_module(model, recurse=True)

        training_mode = model.training

        if keep_original_model:
            try:
                if not check_if_pytorch_greater(2.0, "Please upgrade PyTorch to >=2.0 to use training mode"):
                    model = model.requires_grad_(False)
                model_fast = deepcopy(model)
            except RuntimeError:
                raise ValueError(
                    f"The model {model.__class__.__name__} does not support `deepcopy` operation that is"
                    " internally used to create a copy of the original model when using"
                    " `keep_original_model=True`. Please run the conversion with"
                    " `keep_original_model=False` and create a new copy of the original"
                    " model somewhere else."
                )
            model_fast = replace_to_bettertransformer(model_fast, hf_config)
        else:
            model_fast = replace_to_bettertransformer(model, hf_config)
            model = None

        if BetterTransformerManager.requires_nested_tensor(model_fast.config.model_type):
            set_last_layer(model_fast)

        # Add a class arguments, we might need to identify whether the model
        # has been correctly converted to its `BetterTransformer` version.
        setattr(model_fast, "use_bettertransformer", True)

        if load_accelerate:
            all_model_tensors = [name for name, _ in model_fast.state_dict().items()]
            for module_name in hf_device_map.keys():
                all_model_tensors = [name for name in all_model_tensors if not name.startswith(module_name)]

            if len(all_model_tensors) > 0:
                # This is the case where a transformed submodule is broken into several devices:
                # as the submodules map may differ, we need to reinfer the device map
                bt_device_map = infer_auto_device_map(model_fast, max_memory=max_memory)
            else:
                bt_device_map = hf_device_map

            model_fast = dispatch_model(model_fast, bt_device_map, offload_dir=offload_dir)

            # It is not recommended to have `keep_original_model=True` with a model
            # that is loaded with accelerate but just in case
            if keep_original_model:
                model = dispatch_model(model, hf_device_map, offload_dir=offload_dir)

        # See: https://github.com/pytorch/pytorch/issues/96099
        logger.warning(
            "The BetterTransformer implementation"
            " does not support padding during training, as the fused kernels do not support"
            " attention masks. Beware that passing padded batched data during training may result in unexpected outputs. Please refer to https://huggingface.co/docs/optimum/bettertransformer/overview for more details."
        )

        # Overwrite the `save_pretrained` method
        # by raising an error if the user tries to save the model
        # or push it to the hub.
        model_fast._old_save_pretrained = model_fast.save_pretrained
        model_fast._old_push_to_hub = model_fast.push_to_hub

        model_fast.save_pretrained = raise_save_or_push_incompatible
        model_fast.push_to_hub = raise_save_or_push_incompatible

        if training_mode:
            model_fast = model_fast.train()
        else:
            model_fast = model_fast.eval()

        return model_fast

    def reverse(bt_model: "PreTrainedModel") -> "PreTrainedModel":
        """
        Converts back a model using BetterTransformer to its canonical transformers modeling implementation, in order to save
        and share it.

        Args:
            bt_model (`PreTrainedModel`):
                Model using BetterTransform to convert back to use transformers modeling.

        Returns:
            PreTrainedModel: _description_
        """
        if getattr(bt_model, "use_bettertransformer", False) is False:
            raise ValueError(
                "The method BetterTransformer.reverse() should be used on a model already transformed to the BetterTransformer"
                " format, which appears to not be the case."
            )

        if parse(torch.__version__) <= parse("1.14"):
            raise ValueError(
                f"BetterTransformer reverse transform requires torch>=2.0 but {torch.__version__} is installed. Please upgrade PyTorch."
            )
        config = bt_model.config

        if config.model_type not in ["wav2vec2", "hubert", "bark"]:
            with torch.device("meta"):
                reversed_model = bt_model.__class__(config)
        else:
            # TODO: fix once this is fixed in pytorch
            # reference: https://github.com/pytorch/pytorch/issues/96409
            logger.warning(
                "The reverse transform for the architectures wav2vec2, hubert, bark is memory-heavy due to a bug in PyTorch."
            )
            reversed_model = bt_model.__class__(config)

        if bt_model.training is False:
            reversed_model = reversed_model.eval()

        reversed_modules_paths = []
        for path, module in reversed_model.named_modules():
            if path.startswith(tuple(reversed_modules_paths)):
                continue

            if config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM and any(
                subname in path for subname in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[config.model_type]
            ):
                continue

            target_classes = list(BetterTransformerManager.MODEL_MAPPING[config.model_type].keys())
            has_been_replaced = False
            for target_class in target_classes:
                if module.__class__.__name__ == target_class:
                    has_been_replaced = True
                    break

            # replace parameters, buffers (or possibly full modules) that were modified by the bettertransformer transform
            if has_been_replaced:
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path)._revert(module))
                reversed_modules_paths.append(path + ".")  # add a . to avoid issues with startswith

        # replace back parameters and buffers that were untouched by the bettertransformer transform
        for path, param in reversed_model.state_dict().items():
            if param.device == torch.device("meta") or not path.startswith(tuple(reversed_modules_paths)):
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path))

        # some buffers may be non-persistent, hence not in the state_dict (as token_type_ids for some models)
        for path, param in reversed_model.named_buffers():
            if param.device == torch.device("meta") or not path.startswith(tuple(reversed_modules_paths)):
                recurse_setattr(reversed_model, path, recurse_getattr(bt_model, path))

        return reversed_model
