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
from copy import deepcopy
from typing import Optional

import torch

from optimum.utils import check_if_pytorch_greater, is_accelerate_available

from .models import BETTER_TRANFORMER_LAYERS_MAPPING_DICT, warn_uncompatible_save


if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_module


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
        if len(list(module.children())) > 0:
            replace_to_bettertransformer(module, config)

        if hasattr(module, "is_decoder"):
            # Decoders are not supported yet on Better Transformers
            if module.is_decoder:
                continue

        if hasattr(module, "SCB"):
            # 8-bit modules are not supported
            raise ValueError(
                "`load_in_8bit` and `BetterTransformers` are mutually exclusive",
                " please pass a model that is not loaded in 8-bit.",
            )

        class_name = module.__class__.__name__
        is_bt_compatible = class_name in BETTER_TRANFORMER_LAYERS_MAPPING_DICT

        if is_bt_compatible:
            fast_module = BETTER_TRANFORMER_LAYERS_MAPPING_DICT[class_name](module, config)
            model._modules[name] = fast_module
    return model


def set_last_layer(model):
    r"""
    Iterates over the module list containing the `LayerBetterTransformer` modules. Sets the last layer's `is_last_layer`
    attribute to `True`

    Args:
        `model` (`torch.nn.Module`, **required**):
            The input converted model
    Returns:
        Returns `True` if it has succesfully set the attribute to `True`, otherwise return `False`.
    """
    dict_named_module = dict(model.named_modules())
    sort_fn = lambda list_modules: [module.__class__.__name__ for module in list_modules]  # noqa: E731

    modulelist_lengths = []

    for key in dict_named_module.keys():
        if isinstance(dict_named_module[key], torch.nn.ModuleList) and "encoder" in key:
            modulelist_lengths.append((len(dict_named_module[key]), key))

    # For Albert, each transformer layer is wrapped
    # inside a ModuleList
    if len(modulelist_lengths) > 1:
        _, key = max(modulelist_lengths, key=lambda item: item[0])
        largest_module_list = dict_named_module[key]

        for module in largest_module_list[-1].modules():
            if "LayerBetterTransformer" in module.__class__.__name__:
                setattr(module, "is_last_layer", True)
                return True
        return False
    else:
        for key in dict_named_module.keys():
            if isinstance(dict_named_module[key], torch.nn.ModuleList) and all(
                "LayerBetterTransformer" in module_name for module_name in sort_fn(dict_named_module[key])
            ):
                setattr(dict_named_module[key][-1], "is_last_layer", True)
                return True
        return False


class BetterTransformer(object):
    r"""
    A conversion wrapper that takes as an input the `transformers` model to be converted
    and returns the converted `BetterTransformer` model. The `BetterTransformer` model is based on the `BetterTransformer`
    recently released by PyTorch from its 1.12 version:
    https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/

    # Original PR from: https://github.com/huggingface/transformers/pull/19553 adapted and wrapped in this script.
    """

    @check_if_pytorch_greater(
        "1.13.0",
        "Please upgrade PyTorch following https://pytorch.org/get-started/locally/ in order to use BetterTransformer.",
    )
    def transform(
        model: torch.nn.Module, keep_original_model: bool = False, max_memory: Optional[dict] = None, **kwargs
    ) -> torch.nn.Module:
        r"""
        Conversion script from `transformers` model to its BetterTransformers version

        Args:
            model, (`torch.nn.Module`):
                Original `transformers` model
            keep_original_model (`bool`, *optional*):
                whether to keep or override the original model - essentially
                for memory efficiency reasons
            max_memory (`dict`, *optional*):
                Same argument as `max_memory` argument from `.from_pretrained` function
                in `transformers`.
        Returns:
            The converted model if the conversion has been successful.
        """

        # Check if we have to load the model using `accelerate`
        if hasattr(model, "hf_device_map"):
            load_accelerate = True
        else:
            load_accelerate = False

        hf_config = model.config

        if load_accelerate:
            # remove the hooks from the original model to
            # avoid weights being on `meta` device.
            remove_hook_from_module(model, recurse=True)

        if keep_original_model:
            model = model.requires_grad_(False)
            try:
                model_fast = deepcopy(model)
            except RuntimeError:
                raise ValueError(
                    f"The model {model.__class__.__name__} does not support `deepcopy` operation that is "
                    " internally used to create a copy of the original model when using"
                    " `keep_original_model=True`. Please run the conversion with "
                    " `keep_original_model=False` and create a new copy of the original"
                    " model somewhere else."
                )
            model_fast = replace_to_bettertransformer(model_fast, hf_config).eval()
        else:
            model_fast = replace_to_bettertransformer(model, hf_config).eval()
            model = None

        successfully_converted_model = set_last_layer(model_fast)
        if not successfully_converted_model:
            raise NotImplementedError(
                f"The Better Transformers implementation for the model {model_fast.__class__.__name__} has not been"
                f"implemented yet. Please open an issue requesting the addition of this model with its `BetterTransformer`"
                f"implementation."
            )

        # Step 6: Add a class arguments, we might need to identify whether the model
        # has been correctly converted to its `BetterTransformer` version.
        setattr(model_fast, "use_bettertransformer", True)

        # Step 7: dispatch model if `accelerate` is enabled
        if load_accelerate:
            device_map_bt = infer_auto_device_map(model_fast, max_memory=max_memory)

            remove_hook_from_module(model_fast, recurse=True)

            model_fast = dispatch_model(model_fast, device_map_bt)

            if keep_original_model:
                # It is not recommended to have `keep_original_model=True` with a model
                # that is loaded with accelerate but just in case ..
                model = dispatch_model(model, model.hf_device_map)

        # Step 8: overwrite the `save_pretrained` method
        # by adding a context manager
        setattr(model_fast, "save_pretrained", warn_uncompatible_save(model_fast.save_pretrained))

        return model_fast
