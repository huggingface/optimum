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
import torch

from .base import BetterTransformerBaseLayer


class GPT2AttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer, config):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

    def wrapped_scaled_dot_product(self, query, key, value, attention_mask=None, head_mask=None):
        if query.shape[0] == 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.gpt_layer.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, self._mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(2, -1, -1, -1)

            # we use torch.min to avoid having tensor(-inf)
            attention_mask = torch.min(causal_mask, attention_mask)

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        return self.gpt_layer(*args, **kwargs)


class CodegenAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer, config):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = self.wrapped_scaled_dot_product

        mask_value = torch.finfo(torch.float32).min
        self._mask_value = torch.full([], mask_value, dtype=torch.float32)

    def wrapped_scaled_dot_product(self, query, key, value, attention_mask=None, head_mask=None):
        if query.shape[0] == 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.gpt_layer.causal_mask[:, :, key_length - query_length : key_length, :key_length].to(
                torch.bool
            )

            causal_mask = torch.where(causal_mask, 0, self._mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(2, -1, -1, -1)

            # we use torch.min to avoid having tensor(-inf)
            attention_mask = torch.min(causal_mask, attention_mask)

            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        return sdpa_result, None

    def forward(self, *args, **kwargs):
        return self.gpt_layer(*args, **kwargs)
