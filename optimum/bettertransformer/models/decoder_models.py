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
from transformers.models.bloom.modeling_bloom import BloomAttention
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

from ...utils.import_utils import check_if_transformers_greater


if check_if_transformers_greater("4.31"):
    from transformers.models.bark.modeling_bark import BarkSelfAttention
else:
    from ...utils.dummy_bettertransformer_objects import BarkSelfAttention

from .attention import (
    bark_wrapped_scaled_dot_product,
    bart_forward,
    bloom_forward,
    codegen_wrapped_scaled_dot_product,
    gpt2_wrapped_scaled_dot_product,
    gpt_neo_wrapped_scaled_dot_product,
    gptj_wrapped_scaled_dot_product,
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

        submodules = ["c_proj", "c_attn", "attn_dropout", "resid_dropout", "bias", "masked_bias"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.is_cross_attention:
            setattr(self, "q_attn", getattr(layer, "q_attn"))
            self.original_layers_mapping["q_attn"] = "q_attn"

        self.downcast_qk = False
        self.dropout_prob_attn = config.attn_pdrop

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GPTJAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTJAttention, nn.Module):
    _attn = gptj_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        submodules = [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "attn_dropout",
            "resid_dropout",
            "scale_attn",
        ]
        # Attribute only for transformers>=4.28
        if hasattr(layer, "embed_positions"):
            submodules.append("embed_positions")

        # Attribute only for transformers<4.45
        if hasattr(layer, "bias"):
            submodules.append("bias")
        if hasattr(layer, "masked_bias"):
            submodules.append("masked_bias")

        # Attribute only for transformers>=4.45
        if hasattr(layer, "layer_idx"):
            submodules.append("layer_idx")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.module_mapping = None
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True
        self.dropout_prob_attn = config.attn_pdrop

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GPTNeoXAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoXAttention, nn.Module):
    _attn = gpt2_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.module_mapping = None
        submodules = ["rotary_emb", "query_key_value", "dense", "bias", "masked_bias", "norm_factor"]

        # Attribute only for transformers>=4.45
        if hasattr(layer, "layer_idx"):
            submodules.append("layer_idx")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.downcast_qk = True
        self.dropout_prob_attn = 0.0  # no dropout for gpt-neox

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GPTNeoAttentionLayerBetterTransformer(BetterTransformerBaseLayer, GPTNeoSelfAttention, nn.Module):
    _attn = gpt_neo_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        if layer.bias[0][0][-1][0] == 1:
            self.attention_type = "global"
        else:
            self.attention_type = "local"

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, self.attention_type)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "k_proj", "v_proj", "q_proj", "out_proj", "bias", "masked_bias"]

        # Attribute only for transformers>=4.45
        if hasattr(layer, "layer_id"):
            submodules.append("layer_id")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.dropout_prob_attn = float(config.attention_dropout)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class BarkAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BarkSelfAttention, nn.Module):
    _attn = bark_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig", is_causal: bool = False):
        super().__init__(config)

        is_causal = layer.is_causal

        config.dropout = layer.dropout

        config.hidden_size = layer.embed_dim
        config.num_heads = layer.num_heads
        config.bias = layer.out_proj.bias is not None

        if is_causal:
            config.block_size = layer.bias.shape[-1]

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, is_causal)

        self.module_mapping = None
        submodules = ["dropout", "attn_dropout", "resid_dropout", "att_proj", "out_proj"]

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if is_causal:
            setattr(self, "bias", getattr(layer, "bias"))
            self.original_layers_mapping["bias"] = "bias"

        self.supports_training = False
        self.dropout_prob_attn = float(config.dropout)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class BloomAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BloomAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.dropout_prob_attn = config.attention_dropout

        self.module_mapping = None
        self.layer_idx = getattr(layer, "layer_idx", None)

        submodules = ["query_key_value", "dense", "attention_dropout"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    def forward(self, *args, **kwargs):
        return bloom_forward(self, *args, **kwargs)


class CodegenAttentionLayerBetterTransformer(BetterTransformerBaseLayer, CodeGenAttention, nn.Module):
    _attn = codegen_wrapped_scaled_dot_product

    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config)

        self.module_mapping = None
        submodules = ["attn_dropout", "resid_dropout", "qkv_proj", "out_proj", "scale_attn"]

        # Attribute only for transformers>=4.28
        if hasattr(layer, "embed_positions"):
            submodules.append("embed_positions")

        # Attribute only for transformers<4.45
        if hasattr(layer, "causal_mask"):
            submodules.append("causal_mask")

        # Attribute only for transformers>=4.45
        if hasattr(layer, "layer_idx"):
            submodules.append("layer_idx")

        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        self.dropout_prob_attn = config.attn_pdrop

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer, OPTAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(
                config,
                layer.is_decoder,
            )

        self.scale = torch.sqrt(torch.tensor(layer.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.module_mapping = None
        submodules = ["k_proj", "v_proj", "q_proj", "out_proj"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

    def forward(self, *args, **kwargs):
        return opt_forward(self, *args, **kwargs)


class T5AttentionLayerBetterTransformer(BetterTransformerBaseLayer, T5Attention, torch.nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        if hasattr(config, "text_config"):
            config = config.text_config
        super().__init__(config)

        with torch.device("meta"):
            super(BetterTransformerBaseLayer, self).__init__(config, layer.has_relative_attention_bias)

        submodules = ["q", "k", "v", "o"]
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))

        head_dim = layer.d_model // layer.n_heads  # hidden size / num attention heads
        self.scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.original_layers_mapping = {submodule: submodule for submodule in submodules}

        if layer.has_relative_attention_bias:
            setattr(self, "relative_attention_bias", layer.relative_attention_bias)
            self.original_layers_mapping["relative_attention_bias"] = "relative_attention_bias"

        self.layer_idx = getattr(layer, "layer_idx", None)
        self.is_decoder = layer.is_decoder
        self.module_mapping = None

    def forward(self, *args, **kwargs):
        return t5_forward(self, *args, **kwargs)


def bart_bettertransformer_init(self, layer: "nn.Module", config: "PretrainedConfig"):
    with torch.device("meta"):
        super(BetterTransformerBaseLayer, self).__init__(
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

    self.is_decoder = layer.is_decoder


class BartAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BartAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class BlenderbotAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BlenderbotAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class M2M100AttentionLayerBetterTransformer(BetterTransformerBaseLayer, M2M100Attention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class MarianAttentionLayerBetterTransformer(BetterTransformerBaseLayer, MarianAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)


class PegasusAttentionLayerBetterTransformer(BetterTransformerBaseLayer, PegasusAttention, nn.Module):
    def __init__(self, layer: "nn.Module", config: "PretrainedConfig"):
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)
