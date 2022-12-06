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
import torch
import torch.nn as nn

from ...utils import logging


KNOWN_ACTIVATION_ATTRIBUTES = ["hidden_act", "activation", "act_fn", "activation_function"]
KNOWN_POS_EMB_ATTRIBUTES = ["position_embedding_type"]
KNOWN_NUM_LAYERS = ["num_hidden_layers", "num_layers", "encoder_layers", "n_layers"]

SUPPORTED_ACTIVATION_FUNCTIONS = ["gelu", "relu", "gelu_new"]
USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS = ["quick_gelu"]


logger = logging.get_logger(__name__)


class BetterTransformerBaseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_first = False
        self.use_gelu = False
        self.act_fn = None
        self.pos_emb_type = None
        self.num_heads = None
        self.embed_dim = None
        self.num_layers = None

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

    def forward_checker(self, *args, **kwargs):
        if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
            raise ValueError("Autocast is not supported for `BetterTransformer` integration.")

        if self.training:
            raise ValueError(
                "Training is not supported for `BetterTransformer` integration.",
                " Please use `model.eval()` before running the model.",
            )
