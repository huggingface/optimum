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
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention
from transformers.models.codegen.modeling_codegen import CodeGenAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention
from transformers.models.marian.modeling_marian import MarianAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.pegasus.modeling_pegasus import PegasusAttention
from transformers.models.t5.modeling_t5 import T5Attention

from .attention import (
    bart_forward,
    codegen_wrapped_scaled_dot_product,
    gpt2_wrapped_scaled_dot_product,
    gpt_neo_wrapped_scaled_dot_product,
    opt_forward,
    t5_forward,
)
from .base import BetterTransformerBaseLayer


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class GPT2AttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPT2Attention):
    _attn = gpt2_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        submodules = ["c_proj", "c_attn", "attn_dropout", "resid_dropout"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.is_cross_attention:
            setattr(self, "q_attn", getattr(layer, "q_attn"))
            self.original_layers_mapping["q_attn"] = "q_attn"

        self.downcast_qk = False
        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return super().forward(*args, **kwargs)


# TODO: validate
class GPTJAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTJAttention, nn.Module):
    _attn = gpt2_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        submodules = ["k_proj", "v_proj", "q_proj", "out_proj", "attn_dropout", "resid_dropout"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True
        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return super().forward(*args, **kwargs)


# TODO: validate
class GPTNeoXAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoXAttention, nn.Module):
    _attn = gpt2_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        self.module_mapping = None
        submodules = ["rotary_emb", "query_key_value", "dense"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True
        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return super().forward(*args, **kwargs)


class GPTNeoAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoSelfAttention, nn.Module):
    _attn = gpt_neo_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.bias[0][0][-1][0] == 1:
            self.attention_type = "global"
        else:
            self.attention_type = "local"

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return super().forward(*args, **kwargs)


class CodegenAttentionLayerBetterTransformer(BetterTransformerBaseLayer, CodeGenAttention, nn.Module):
    _attn = codegen_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "qkv_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return super()(*args, **kwargs)


class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer, OPTAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.is_cross_attention, layer.layer_idx)

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return opt_forward(self, *args, **kwargs)


class T5AttentionLayerBetterTransformer(BetterTransformerBaseLayer, T5Attention, torch.nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.has_relative_attention_bias)

        submodules = ["q", "k", "v", "o"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        head_dim = layer.d_model // layer.n_heads  # hidden size / num attention heads
        self.scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}
        self.module_mapping = None

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return t5_forward(self, *args, **kwargs)


class BartAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BartAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.has_relative_attention_bias)

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return bart_forward(self, *args, **kwargs)


class BlenderbotAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BlenderbotAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config,
                layer.embed_dim,
                layer.num_heads,
                layer.dropout,
                layer.is_decoder,
                layer.k_proj.bias is not None,
            )

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return bart_forward(self, *args, **kwargs)


class M2M100AttentionLayerBetterTransformer(BetterTransformerBaseLayer, M2M100Attention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config,
                layer.embed_dim,
                layer.num_heads,
                layer.dropout,
                layer.is_decoder,
                layer.k_proj.bias is not None,
            )

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return bart_forward(self, *args, **kwargs)


class MarianAttentionLayerBetterTransformer(BetterTransformerBaseLayer, MarianAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config,
                layer.embed_dim,
                layer.num_heads,
                layer.dropout,
                layer.is_decoder,
                layer.k_proj.bias is not None,
            )

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return bart_forward(self, *args, **kwargs)


class PegasusAttentionLayerBetterTransformer(BetterTransformerBaseLayer, PegasusAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config,
                layer.embed_dim,
                layer.num_heads,
                layer.dropout,
                layer.is_decoder,
                layer.k_proj.bias is not None,
            )

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.is_decoder = True

    def forward(self, *args, **kwargs):
        super().forward_checker()
        return bart_forward(self, *args, **kwargs)
