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
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers import PretrainedConfig

import torch

from ...utils import logging, recurse_getattr, recurse_setattr


KNOWN_ACTIVATION_ATTRIBUTES = ["hidden_act", "activation", "act_fn", "activation_function"]
KNOWN_POS_EMB_ATTRIBUTES = ["position_embedding_type"]
KNOWN_NUM_LAYERS = ["num_hidden_layers", "num_layers", "encoder_layers", "n_layers"]

SUPPORTED_ACTIVATION_FUNCTIONS = ["gelu", "relu", "gelu_new"]
USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS = ["quick_gelu"]


logger = logging.get_logger(__name__)


class BetterTransformerBaseLayer:
    def __init__(
        self,
        config: "PretrainedConfig",
    ):
        r"""
        Base layer for `BetterTransformer` integration. This class is used to wrap all the necessary
        components for the `BetterTransformer` integration.

        Args:
            config (`transformers.PretrainedConfig`):
                The config of the model.
        """
        self.norm_first = False
        self.use_gelu = False
        self.act_fn = None
        self.pos_emb_type = None
        self.num_heads = None
        self.embed_dim = None
        self.num_layers = None
        self.original_layers_mapping = {}
        self.module_mapping = None
        # Some models does not have some attributes thus needs to be ignored
        # e.g. whisper does not have self_attn.k_proj.bias but has self_attn.v_proj.bias & self_attn.q_proj.bias
        self.keys_to_ignore = []

        # Get activation function
        for attr in KNOWN_ACTIVATION_ATTRIBUTES:
            if hasattr(config, attr):
                self.act_fn = getattr(config, attr)
                break

        # if act_fn not found in the config, fall back to the private `_get_activation_function` if available
        if self.act_fn is None and hasattr(self, "_get_activation_function"):
            self.act_fn = self._get_activation_function(config)

        # Get pos emb type
        for attr in KNOWN_POS_EMB_ATTRIBUTES:
            if hasattr(config, attr):
                self.pos_emb_type = getattr(config, attr)
                break

        # Get num_layers
        for attr in KNOWN_NUM_LAYERS:
            if hasattr(config, attr):
                self.num_layers = getattr(config, attr)
                break

    def validate_bettertransformer(self):
        r"""
        A wrapper function to validate the `BetterTransformer` implementation. Implements most relevant checks
        that are present in: https://github.com/pytorch/pytorch/blob/0fc7de398636f4b53e6c3fde38b4e48a5ff5b37d/torch/nn/modules/transformer.py#L457-L475
        """
        # Sanity checks
        if self.num_heads is None:
            raise ValueError("Number of heads not set for `BetterTransformer` integration.")

        if self.embed_dim is None:
            raise ValueError("Embedding dimension not set for `BetterTransformer` integration.")

        if self.norm2_eps is None or self.norm1_eps is None:
            raise ValueError("`norm2_eps` and `norm1_eps` not set for `BetterTransformer` integration.")

        # Check positional embedding
        if self.pos_emb_type is not None and self.pos_emb_type != "absolute":
            raise ValueError(
                f"Positional embedding type {self.pos_emb_type} not " "supported for `BetterTransformer` integration"
            )

        # Check norm1 epsilon and norm2 epsilon equality
        if self.norm1_eps != self.norm2_eps:
            raise ValueError("norm1_eps and norm2_eps must be equal for `BetterTransformer` integration.")

        # Check activation function
        if self.act_fn in USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS:
            logger.warning(
                f"Overridding {self.act_fn} activation with gelu. Use the transformed model at your own risk, the output logits could be significantly different."
            )
            self.act_fn = "gelu"
        elif self.act_fn not in SUPPORTED_ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Activation function {self.act_fn} not supported" " for `BetterTransformer` integration."
            )
        self.use_gelu = (self.act_fn == "gelu") or (self.act_fn == "gelu_new")

        # Check num_head is even
        if self.num_heads % 2 == 1:
            raise ValueError(
                f"Number of heads {self.num_heads} is not supported"
                " for `BetterTransformer` integration."
                f" Number of heads must be even."
            )

    def _revert(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.module_mapping is not None:
            if "" in self.module_mapping.values():
                for bt_module_attr_name, value in self.module_mapping.items():
                    if value == "":
                        module = getattr(self, bt_module_attr_name)
                        return module
            else:
                raise NotImplementedError("replacing a submodule in revert is not supported")

        for modified_layer_key_names, original_layer_key_names in self.original_layers_mapping.items():
            if isinstance(original_layer_key_names, list):
                current_weight = getattr(self, modified_layer_key_names)

                # Split the current weight n chunks - this is useful to split
                # the qkv layers into q, k, v layers for example.
                split_index = current_weight.shape[0] // len(original_layer_key_names)
                for i, subparam_name in enumerate(original_layer_key_names):
                    if recurse_getattr(module, subparam_name) is None:
                        # this is for example the case if bias=False is set for a nn.Linear layer
                        continue

                    if module not in self.keys_to_ignore:
                        # TODO: remove the clone once https://github.com/huggingface/transformers/pull/27314 & https://github.com/huggingface/safetensors/pull/379 are released.
                        # Safetensors is bugged when using views of tensors.
                        parameter = current_weight[i * split_index : (i + 1) * split_index].clone()
                        if isinstance(recurse_getattr(module, subparam_name), torch.nn.Parameter):
                            parameter = torch.nn.Parameter(parameter)
                        recurse_setattr(module, subparam_name, parameter)
            elif isinstance(original_layer_key_names, str):
                if recurse_getattr(module, original_layer_key_names) is None:
                    # this is for example the case if bias=False is set for a nn.Linear layer
                    continue

                parameter = getattr(self, modified_layer_key_names)
                if isinstance(recurse_getattr(module, original_layer_key_names), torch.nn.Parameter):
                    parameter = torch.nn.Parameter(parameter)
                recurse_setattr(module, original_layer_key_names, parameter)
            else:
                raise ValueError(
                    f"Invalid type {type(modified_layer_key_names)} for `original_layers_mapping`",
                    " please use either `str` or `list`.",
                )
        return module
