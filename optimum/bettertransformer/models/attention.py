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

from ...utils.import_utils import check_if_transformers_greater


# TODO: remove once we are much higher than 4.31
if check_if_transformers_greater("4.31"):
    from transformers.models.llama.modeling_llama import _expand_mask as _llama_expand_mask
    from transformers.models.llama.modeling_llama import _make_causal_mask as _llama_make_causal_mask
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
else:
    from ...utils.dummy_bettertransformer_objects import _expand_mask as _llama_expand_mask
    from ...utils.dummy_bettertransformer_objects import _make_causal_mask as _llama_make_causal_mask
    from ...utils.dummy_bettertransformer_objects import apply_rotary_pos_emb, repeat_kv

# TODO (CRITICAL): Layer-wise attention scaling is broken for several archs (see a fix in gpt_bigcode_wrapped_scaled_dot_product).


def raise_on_head_mask(head_mask: Optional[torch.Tensor]):
    if head_mask is not None:
        raise ValueError(
            "layer_head_mask different than None is unsupported for now with BetterTransformer, please"
            "open a PR or an issue at https://github.com/huggingface/optimum."
        )


# Adapted from transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn
def gpt2_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]

    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    if self.downcast_qk:
        query = query.to(value.dtype)
        key = key.to(value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        # causal_mask is always [True, ..., True] otherwise, so executing this
        # is unnecessary
        if query_length > 1:
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            if attention_mask is not None:
                attention_mask = causal_mask + attention_mask

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    if self.downcast_qk:
        sdpa_result = sdpa_result.to(value.dtype)

    return sdpa_result, None


# Adapted from transformers.models.bark.modeling_bark.BarkSelfAttention._attn
def bark_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    raise_on_head_mask(head_mask)

    # When `past_kv` is provided, we're doing incremental decoding and `q.shape[2] == 1`: q only contains
    # the query for the last token. scaled_dot_product_attention interprets this as the first token in the
    # sequence, so if is_causal=True it will mask out all attention from it. This is not what we want, so
    # to work around this we set is_causal=False.
    is_causal = self.is_causal and query.shape[2] != 1

    sdpa_result = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
    )

    return sdpa_result, None


# Adapted from transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention._attn
def gpt_neo_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    raise_on_head_mask(head_mask)
    query = query * self.scale
    batch_size = query.shape[0]

    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if (batch_size == 1 or self.training) and self.attention_type == "global":
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        causal_mask = torch.where(causal_mask, 0, mask_value)
        if batch_size > 1:
            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

        if attention_mask is not None:
            attention_mask = causal_mask + attention_mask

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    return sdpa_result, None


# Adapted from transformers.models.codegen.modeling_codegen.CodeGenAttention._attn
def codegen_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, 0, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    # in codegen the query and key are always in fp32 regardless of the dtype of the model
    # https://github.com/huggingface/transformers/blob/5b28b7833297adf65c5160a685425ddb1eee5ce2/src/transformers/models/codegen/modeling_codegen.py#L226
    query = query.to(value.dtype)
    key = key.to(value.dtype)

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            # first step of the decoding
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # in this case, which is the later decoding steps, the `causal_mask`` in
            # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/gpt2/modeling_gpt2.py#L195
            # is [True, ..., True] so actually not causal
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        # causal_mask is always [True, ..., True] otherwise, so executing this
        # is unnecessary
        if query_length > 1:
            causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

            # we use torch.min to avoid having tensor(-inf)
            attention_mask = torch.min(causal_mask, attention_mask)

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    return sdpa_result, None


# Adapted from transformers.models.opt.modeling_opt.OPTAttention.forward
def opt_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    raise_on_head_mask(layer_head_mask)

    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    # TODO: raise on batch_size = 1 + padding

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    batch_size, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
        value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    query_states = self._shape(query_states, tgt_len, batch_size)

    query_states = query_states * self.scale

    dropout_p = self.dropout if self.training else 0.0
    if batch_size == 1 or self.training:
        if query_states.shape[2] > 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    if attn_output.size() != (batch_size, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value


# Adapted from transformers.models.t5.modeling_t5.T5Attention.forward
def t5_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    **kwargs,
):
    raise_on_head_mask(layer_head_mask)

    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")
    if len(self.pruned_heads) > 0:
        raise ValueError(f"Setting `pruned_heads` is unsupported with BetterTransformer, found {self.pruned_heads}.")
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            elif past_key_value.shape[2] != key_value_states.shape[1]:
                # checking that the `sequence_length` of the `past_key_value` is the same as
                # the provided `key_value_states` to support prefix tuning
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    dropout_p = self.dropout if self.training else 0.0
    query_states = self.scale * query_states
    if position_bias is None and not self.has_relative_attention_bias:
        if mask is None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
        elif mask is not None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=mask, dropout_p=dropout_p, is_causal=False
            )

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length),
                device=value_states.device,
                dtype=value_states.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=value_states.device)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.has_relative_attention_bias:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=position_bias, dropout_p=dropout_p, is_causal=False
            )
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=position_bias, dropout_p=dropout_p, is_causal=False
        )

    attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    return outputs


# Adapted from transformers.models.bart.modeling_bart.BartAttention.forward
def bart_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    raise_on_head_mask(layer_head_mask)
    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states)
    # get key, value proj
    # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `key_value_states` to support prefix tuning
    if is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    query_states = self._shape(query_states, tgt_len, bsz)
    key_states = key_states
    value_states = value_states

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )

    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value


# Adapted from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _llama_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    # We do not care about the attention mask in the batch size = 1 case
    if attention_mask.size(0) > 1:
        if input_shape[-1] > 1:
            combined_attention_mask = _llama_make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _llama_expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
    else:
        if input_shape[-1] > 1 and attention_mask is not None and attention_mask[0][0] == 0:
            raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    return combined_attention_mask


def llama_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            torch.nn.functional.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            torch.nn.functional.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            torch.nn.functional.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if bsz == 1 or self.training:
        # BEWARE: at this stage, attention_mask is not the same as in transformers llama
        if query_states.shape[2] > 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
            )
    else:
        # At this stage, **attention_mask is the same** as in transformers llama
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(
            [torch.nn.functional.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def gpt_bigcode_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    raise_on_head_mask(head_mask)

    # TODO: remove once PyTorch 2.1 is released with the scale argument to SDPA
    if not self.scale_attn_weights:
        query = query / self.head_dim**0.5

    # MQA models: (batch_size, query_length, num_heads * head_dim)
    # MHA models: (batch_size, num_heads, query_length, head_dim)
    query_shape = query.shape
    batch_size = query_shape[0]
    kv_seq_len = key.shape[-2]

    if self.multi_query:
        query_length = query_shape[1]

        # NOTE: Maybe there is better than this?
        query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
    else:
        raise NotImplementedError(
            "BetterTransformer integration with GPT BigCode without Multi-Query Attention (MQA) has not been implemented. Please open an issue or PR at https://github.com/huggingface/optimum."
        )

    dropout_p = self.dropout_prob_attn if self.training else 0.0

    # I did not find how to avoid these unsqueeze, SDPA complains otherwise as the query and key/value have a different number of dimensions.
    key = key.unsqueeze(1)
    value = value.unsqueeze(1)

    # Although these expand are not numerically useful, PyTorch 2.0.1 and 2.1.0.dev20230805+cu118 can not dispatch to mem-efficient attention
    # and flash attention from the shapes
    # query = [batch_size, num_heads, query_length, head_dim]
    # key = [batch_size, 1, past_length, head_dim]
    # value = [batch_size, 1, past_length, head_dim]
    # which is unfortunate. Hopefully can be changed in the future. These expand should not be too expansive as they do not do memory copy.
    key = key.expand(-1, self.num_heads, -1, -1)
    value = value.expand(-1, self.num_heads, -1, -1)

    # We treat self.training and (batch_size == 1 and query_length == 1) cases separately to still allow the dispatch to Flash Attention.
    if self.training:
        is_causal = True
        attn_mask = None
    elif batch_size == 1 and query_length == 1:
        is_causal = False
        attn_mask = None
    elif batch_size == 1 and kv_seq_len == query_length:
        is_causal = True
        attn_mask = None
    elif attention_mask is not None:
        mask_value = self._get_mask_value(query.device, query.dtype)

        # gpt_bigcode has the bad taste to use a causal mask a
        # [batch_size, target_length, 1, source_length] which is different from
        # **all** other architectures and not compatible with SDPA.
        # We could avoid this transpose by overriding the forward from GPTBigCodeModel,
        # but it is probably not worth it.
        attention_mask = attention_mask.transpose(1, 2)
        attn_mask = torch.where(attention_mask, 0.0, mask_value)
        is_causal = False
    else:
        attn_mask = None
        is_causal = True

    sdpa_result = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )

    if self.multi_query:
        # (batch_size, num_heads, seq_len, head_dim) --> (batch_size, seq_len, num_heads, head_dim)
        sdpa_result = sdpa_result.transpose(1, 2)

        # Reshape is kind of expensive here (as here it does a memory copy)
        # but I did not manage to make away without it.
        # (batch_size, seq_len, num_heads, head_dim) --> (batch_size, seq_len, num_heads * head_dim)
        sdpa_result = sdpa_result.reshape(query_shape)

    return sdpa_result


def gpt_bigcode_forward(
    self,
    hidden_states: torch.Tensor,
    layer_past: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn") or not self.is_cross_attention:
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key_value = self.c_attn(encoder_hidden_states)
        attention_mask = encoder_attention_mask
    elif self.multi_query:
        query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
    else:
        # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
        # i.e., the memory layout is not the same as GPT2.
        # This makes the concatenation with past_key_value more efficient.
        query, key_value = (
            self.c_attn(hidden_states)
            .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
            .transpose(1, 2)
            .split((self.head_dim, 2 * self.head_dim), dim=3)
        )

    if layer_past is not None:
        key_value = torch.cat((layer_past, key_value), dim=-2)
    present = key_value if use_cache else None

    key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

    # Difference with the transformers implementation: there is no need to transpose the key here,
    # as SDPA expects seq_length to be at index -2
    attn_output = self._attn(query, key, value, attention_mask, head_mask)

    if not self.multi_query:
        attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)

    return outputs


# Adapted from transformers.models.bloom.modeling_bloom.BloomAttention.forward
def bloom_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    **kwargs,
):
    raise_on_head_mask(head_mask)

    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape

    # Permute to [batch_size, num_heads, seq_length, head_dim]
    query_layer = query_layer.transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        past_key = past_key.transpose(1, 2)

        key_layer = key_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

        # concatenate along seq_length dimension
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)

        # untangle batch_size from self.num_heads
        key_layer = key_layer.reshape(batch_size, self.num_heads, *key_layer.shape[1:])
        value_layer = value_layer.reshape(batch_size, self.num_heads, *value_layer.shape[1:])
    else:
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

    alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
    alibi = torch.masked_fill(alibi, attention_mask, torch.finfo(alibi.dtype).min)

    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer,
        key_layer,
        value_layer,
        attn_mask=alibi,
        dropout_p=self.dropout_prob_attn if self.training else 0.0,
    )

    # Transform [batch_size, num_heads, seq_length, head_dim] to [batch_size, seq_length, num_heads * head_dim]
    context_layer = context_layer.transpose(1, 2)
    context_layer = context_layer.reshape(*context_layer.shape[:2], -1)

    # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + torch.nn.functional.linear(
                context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
            )
    else:
        output_tensor = self.dense(context_layer)

    output_tensor = torch.nn.functional.dropout(output_tensor, p=self.hidden_dropout, training=self.training)
    output_tensor = residual + output_tensor

    if use_cache is True:
        present = (
            key_layer.reshape(-1, *key_layer.shape[2:]).transpose(1, 2),
            value_layer.reshape(-1, *value_layer.shape[2:]),
        )
    else:
        present = None

    return (output_tensor, present)


def falcon_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    **kwargs,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
    num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, query_length, _, _ = query_layer.shape

    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")
    elif head_mask is not None:
        raise ValueError("Non-None head_mask is not supported in BetterTransformer.")

    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(
        batch_size * num_kv_heads,
        query_length,
        self.head_dim,
    )
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

    past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
    query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length, position_ids)

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, kv_length, head_dim]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    _, kv_length, _ = key_layer.shape
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
    key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
    value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

    # We treat self.training and (batch_size == 1 and query_length == 1) cases separately to still allow the dispatch to Flash Attention.
    if self.training:
        attention_mask = None
    elif batch_size == 1 and query_length == 1:
        attention_mask = None
    elif batch_size == 1 and kv_length == query_length:
        attention_mask = None

    if alibi is None:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer_,
            key_layer_,
            value_layer_,
            attention_mask,
            0.0,
            is_causal=self.is_causal and attention_mask is None and query_length > 1,
        )
        attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

    else:
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer_,
            key_layer_,
            value_layer_,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None,
        )
        context_layer = context_layer.transpose(1, 2)
        context_layer = context_layer.reshape(batch_size, query_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(context_layer)

    return output_tensor, present
