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

import torch

from .models import is_module_fast
from .utils import check_if_pytorch_greater_112, is_accelerate_available


if is_accelerate_available():
    from accelerate.hooks import attach_align_device_hook_on_blocks, remove_hook_from_submodules


def init_accelerate_hooks(slow_module, fast_module):
    r"""
    This will initilialize `accelerate` hooks into the `Fast` module.
    """
    if (not is_accelerate_available()) or (not hasattr(slow_module, "_hf_hook")):
        return fast_module

    # Delete the previous hook
    remove_hook_from_submodules(fast_module)

    exec_hook_dict = {fast_module.__class__.__name__: str(slow_module._hf_hook.execution_device)}

    attach_align_device_hook_on_blocks(fast_module, exec_hook_dict)


# Step 1: Recurse over the modules of the model
# Step 2: Verify if the module `Fast` is present for that model
# Step 3: If yes, replace the `...Layer` module with the `...LayerFast` modules
# Step 4: If not, yield an error.
def replace_to_fast(model):
    r"""
    Replaces the current model to its `Fast` implementation. Loops recursively into the model and replaces the
    `Layer` modules with its `Fast` correspondant model

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
            if module.is_decoder:
                continue

        class_name = module.__class__.__name__
        maybe_fast_module = is_module_fast(class_name)
        if not isinstance(maybe_fast_module, bool):
            fast_module = maybe_fast_module(module)

            # if is_accelerate_available():
            #     init_accelerate_hooks(module, fast_module)

            model._modules[name] = fast_module
    return model


# Step 5: Post process the potentially converted model by setting the `is_last_layer` attribute to `True`
# For the last `Fast` layer.
def set_last_layer(model):
    r"""
    Args:
    Iterates over the module list containing the `LayerFast` modules. Sets the last layer's `is_last_layer`
    attribute to `True`
        `model` (`torch.nn.Module`, **required**):
            The input converted model
    Returns:
        Returns `True` if it has succesfully set the attribute to `True`, otherwise return `False`.
    """
    dict_named_module = dict(model.named_modules())
    sort_fn = lambda list_modules: [module.__class__.__name__ for module in list_modules]  # noqa: E731

    for key in dict_named_module.keys():
        if isinstance(dict_named_module[key], torch.nn.ModuleList) and all(
            "LayerFast" in module_name for module_name in sort_fn(dict_named_module[key])
        ):
            setattr(dict_named_module[key][-1], "is_last_layer", True)
            return True
    return False


@check_if_pytorch_greater_112()
class BetterTransformer(object):
    r"""
    A conversion wrapper that takes as an input the `transformers` model to be converted
    and returns the converted `Fast` model. The `Fast` model is based on the `BetterTransformer`
    recently released by PyTorch from its 1.12 version:
    https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/

    Original PR from: https://github.com/huggingface/transformers/pull/19553 adapted and wrapped in
    this script.
    """

    def transform(model, keep_original_model=False):
        r"""
        Conversion script from `transformers` model to its BetterTransformers version

        Args:
            model, (`torch.nn.Module`, **required**):
                Original `transformers` model
            keep_original_model (`bool`, *optional):
                whether to keep or override the original model - essentially
                for memory efficiency reasons
        """
        if keep_original_model:
            model_fast = deepcopy(model)
            model_fast = replace_to_fast(model_fast).eval()
        else:
            model_fast = replace_to_fast(model).eval()

        successfully_converted_model = set_last_layer(model_fast)
        if not successfully_converted_model:
            raise NotImplementedError(
                f"The Better Transformers implementation for the model {model.__class__.__name__} has not been"
                f"implemented yet. Please open an issue requesting the addition of this model with its `Fast`"
                f"implementation."
            )

        # Step 6: Add a class arguments, we might need to identify whether the model
        # has been correctly converted to its `Fast` version.
        setattr(model_fast, "is_fast", True)

        return model_fast
