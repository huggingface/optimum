# Copyright 2023 The HuggingFace and Meta Team.  All rights reserved.
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
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, SamMaskDecoderConfig, SamVisionConfig
from transformers.models.sam.modeling_sam import SamAttention, SamVisionAttention

from .attention import sam_attention_forward, sam_vision_attention_forward
from .base import BetterTransformerBaseLayer


class SamAttentionLayerBetterTransformer(BetterTransformerBaseLayer, SamAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig", parent_attrs: Dict):
        super().__init__(config)

        if isinstance(config, PretrainedConfig) and not isinstance(config, SamMaskDecoderConfig):
            config = config.mask_decoder_config

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config, downsample_rate=config.hidden_size // layer.internal_dim
            )

        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.supports_training = False

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return sam_attention_forward(self, *args, **kwargs)


class SamVisionAttentionLayerBetterTransformer(BetterTransformerBaseLayer, SamVisionAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig", parent_attrs: Dict):
        """
        `parent_attrs` should contain "window_size".
        """
        super().__init__(config)

        if isinstance(config, PretrainedConfig) and not isinstance(config, SamVisionConfig):
            config = config.vision_config

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, window_size=parent_attrs["window_size"])

        submodules = [
            "qkv",
            "proj",
        ]
        if layer.use_rel_pos:
            submodules.append("rel_pos_h")
            submodules.append("rel_pos_w")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.supports_training = False

    # Adapted from transformers.models.sam.modeling_sam.SamVisionAttention.add_decomposed_rel_pos
    def get_mask(
        self,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        mask = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        return mask.reshape(batch_size, query_height * query_width, key_height * key_width)

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return sam_vision_attention_forward(self, *args, **kwargs)
