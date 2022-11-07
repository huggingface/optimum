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

from optimum.utils import check_if_pytorch_greater_112, is_accelerate_available

from .models import is_module_fast


if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import named_module_tensors, set_module_tensor_to_device


def init_accelerate_hook(module):
    r"""
    This function initializes `accelerate` hooks by setting the
    parameters of the module on the `cpu` if there is any offloading involved

    Args:
        `module` (`torch.nn.Module`, **required**):
            The input module to initialize
    Returns:
        The initialized module
    """
    for name, child in module.named_children():
        if hasattr(child, "_hf_hook"):
            if child._hf_hook.weights_map is not None:
                hook = child._hf_hook
                if child._hf_hook.offload:
                    for name, _ in named_module_tensors(
                        child, include_buffers=hook.offload_buffers, recurse=hook.place_submodules
                    ):
                        set_module_tensor_to_device(child, name, "cpu", value=hook.weights_map[name])
        init_accelerate_hook(child)
    return module


def replace_to_fast(model):
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
        `model` (`torch.nn.Module`, **required**):
            The input model to convert
    Returns:
        The converted model
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_to_fast(module)

        if hasattr(module, "is_decoder"):
            # Decoders are not supported yet on Better Transformers
            if module.is_decoder:
                continue

        class_name = module.__class__.__name__
        maybe_fast_module = is_module_fast(class_name)

        if not isinstance(maybe_fast_module, bool):
            fast_module = maybe_fast_module(module)
            model._modules[name] = fast_module
        elif hasattr(module, "_hf_hook"):
            # If the module has `accelerate` hooks, manually
            # set the parameters on the `CPU` if there is
            # any offload involved.
            module = init_accelerate_hook(module)
            model._modules[name] = module
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

    for key in dict_named_module.keys():
        if isinstance(dict_named_module[key], torch.nn.ModuleList) and all(
            "LayerBetterTransformer" in module_name for module_name in sort_fn(dict_named_module[key])
        ):
            setattr(dict_named_module[key][-1], "is_last_layer", True)
            return True
    return False


@check_if_pytorch_greater_112()
class BetterTransformer(object):
    r"""
    A conversion wrapper that takes as an input the `transformers` model to be converted
    and returns the converted `BetterTransformer` model. The `BetterTransformer` model is based on the `BetterTransformer`
    recently released by PyTorch from its 1.12 version:
    https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/

    Original PR from: https://github.com/huggingface/transformers/pull/19553 adapted and wrapped in
    this script.
    """

    def transform(
        model: torch.nn.Module, keep_original_model: bool = False, max_memory: Optional[dict] = None, **kwargs
    ) -> torch.nn.Module:
        r"""
        Conversion script from `transformers` model to its BetterTransformers version

        Args:
            model, (`torch.nn.Module`, **required**):
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

        if keep_original_model:
            model_fast = deepcopy(model)
            model_fast = replace_to_fast(model_fast).eval()
        else:
            model_fast = replace_to_fast(model).eval()

        successfully_converted_model = set_last_layer(model_fast)
        if not successfully_converted_model:
            raise NotImplementedError(
                f"The Better Transformers implementation for the model {model.__class__.__name__} has not been"
                f"implemented yet. Please open an issue requesting the addition of this model with its `BetterTransformer`"
                f"implementation."
            )

        # Step 6: Add a class arguments, we might need to identify whether the model
        # has been correctly converted to its `BetterTransformer` version.
        setattr(model_fast, "is_fast", True)

        # Step 7: dispatch model if `accelerate` is enabled
        if load_accelerate:
            device_map_bt = infer_auto_device_map(model_fast, max_memory=max_memory)

            remove_hook_from_module(model_fast, recurse=True)

            model_fast = dispatch_model(model_fast, device_map_bt)

        return model_fast
