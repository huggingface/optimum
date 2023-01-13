#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.file_utils import add_start_docstrings_to_model_forward


DECODER_WITH_LM_HEAD_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing attention on padding token indices of `input_ids`.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""


# Currently inherits from PreTrainedModel for export constraint coming from transformers.onnx.export
class _DecoderWithLMhead(PreTrainedModel):
    """
    Decoder model with a language modeling head on top.
    Arguments:
        model (`transformers.PreTrainedModel`):
            The model from which to extract the decoder and the language modeling head.
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__(model.config)
        self.config = model.config
        self.decoder = model.get_decoder()
        self.lm_head = model.get_output_embeddings()
        self.final_logits_bias = getattr(model, "final_logits_bias", None)

    @add_start_docstrings_to_model_forward(DECODER_WITH_LM_HEAD_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True,
        )
        last_hidden_state = decoder_outputs.last_hidden_state

        if self.config.model_type == "t5" and self.config.tie_word_embeddings:
            # T5 needs its output to be rescaled before projecting on vocab
            last_hidden_state = last_hidden_state * (self.config.d_model**-0.5)

        lm_logits = self.lm_head(last_hidden_state)

        # Add the final bias if present in the model
        if self.final_logits_bias is not None:
            lm_logits += self.final_logits_bias

        if labels is None:
            return lm_logits, decoder_outputs.past_key_values
        else:
            # Calculate loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return loss, lm_logits, decoder_outputs.past_key_values
