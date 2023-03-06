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
from typing import Optional, Tuple

import torch

from .base import BetterTransformerBaseLayer


def raise_on_head_mask(head_mask: Optional[torch.Tensor]):
    if head_mask is not None:
        raise ValueError(
            "layer_head_mask different than None is unsupported for now with BetterTransformer, please"
            "open a PR or an issue at https://github.com/huggingface/optimum."
        )


class GPT2AttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer, config):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

    def wrapped_scaled_dot_product(self, query, key, value, attention_mask=None, head_mask=None):
        batch_size = query.shape[0]

        if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

        if batch_size == 1:
            if query.shape[2] > 1:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            query_length, key_length = query.size(-2), key.size(-2)

            # causal_mask is always [True, ..., True] otherwise, so executing this
            # is unnecessary
            if query_length > 1:
                causal_mask = self.gpt_layer.bias[:, :, key_length - query_length : key_length, :key_length].to(
                    torch.bool
                )

                causal_mask = torch.where(causal_mask, 0, self._mask_value)

                # torch.Tensor.expand does no memory copy
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
                attention_mask = causal_mask + attention_mask

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        raise_on_head_mask(kwargs["head_mask"])
        return self.gpt_layer(*args, **kwargs)


class GPTNeoAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer, config):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

        if self.gpt_layer.bias[0][0][-1][0] == 1:
            self.attention_type = "global"
        else:
            self.attention_type = "local"

        self.scale = torch.sqrt(torch.tensor(self.gpt_layer.head_dim, dtype=torch.float32)).to(
            torch.get_default_dtype()
        )

    def wrapped_scaled_dot_product(self, query, key, value, attention_mask=None, head_mask=None):
        query = query * self.scale
        batch_size = query.shape[0]
        if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

        if batch_size == 1 and self.attention_type == "global":
            if query.shape[2] > 1:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            query_length, key_length = query.size(-2), key.size(-2)

            causal_mask = self.gpt_layer.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, self._mask_value)
            if batch_size > 1:
                # torch.Tensor.expand does no memory copy
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            attention_mask = causal_mask + attention_mask

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        raise_on_head_mask(kwargs["head_mask"])
        return self.gpt_layer(*args, **kwargs)


class CodegenAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer, config):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

    def wrapped_scaled_dot_product(self, query, key, value, attention_mask=None, head_mask=None):
        batch_size = query.shape[0]
        if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

        if batch_size == 1:
            if query.shape[2] > 1:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            query_length, key_length = query.size(-2), key.size(-2)

            # causal_mask is always [True, ..., True] otherwise, so executing this
            # is unnecessary
            if query_length > 1:
                causal_mask = self.gpt_layer.causal_mask[:, :, key_length - query_length : key_length, :key_length].to(
                    torch.bool
                )

                causal_mask = torch.where(causal_mask, 0, self._mask_value)

                # torch.Tensor.expand does no memory copy
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

                # we use torch.min to avoid having tensor(-inf)
                attention_mask = torch.min(causal_mask, attention_mask)

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        raise_on_head_mask(kwargs["head_mask"])
        return self.gpt_layer(*args, **kwargs)


class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, opt_layer, config):
        super().__init__(config, opt_layer)

        self.opt_layer = opt_layer

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

        self.scale = torch.sqrt(torch.tensor(self.opt_layer.head_dim, dtype=torch.float32)).to(
            torch.get_default_dtype()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        if output_attentions is True:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        raise_on_head_mask(layer_head_mask)

        # TODO: raise on batch_size = 1 + padding

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        batch_size, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.opt_layer.q_proj(hidden_states) * self.opt_layer.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.opt_layer._shape(self.opt_layer.k_proj(key_value_states), -1, batch_size)
            value_states = self.opt_layer._shape(self.opt_layer.v_proj(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self.opt_layer._shape(self.opt_layer.k_proj(hidden_states), -1, batch_size)
            value_states = self.opt_layer._shape(self.opt_layer.v_proj(hidden_states), -1, batch_size)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self.opt_layer._shape(self.opt_layer.k_proj(hidden_states), -1, batch_size)
            value_states = self.opt_layer._shape(self.opt_layer.v_proj(hidden_states), -1, batch_size)

        if self.opt_layer.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_states = self.opt_layer._shape(query_states, tgt_len, batch_size)

        query_states = query_states * self.scale
        if batch_size == 1:
            if query_states.shape[2] > 1:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        if attn_output.size() != (batch_size, self.opt_layer.num_heads, tgt_len, self.opt_layer.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.opt_layer.num_heads, tgt_len, self.opt_layer.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # attn_output = attn_output.view(bsz, self.opt_layer.num_heads, tgt_len, self.opt_layer.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(batch_size, tgt_len, self.opt_layer.embed_dim)

        attn_output = self.opt_layer.out_proj(attn_output)

        return attn_output, None, past_key_value
