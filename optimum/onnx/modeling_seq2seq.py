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
from transformers import PreTrainedModel


# Currently inherits from PreTrainedModel for export constraint coming from transformers.onnx.export
class _DecoderWithLMhead(PreTrainedModel):
    # Decoder with language modeling head
    def __init__(self, model):
        super().__init__(model.config)
        self.config = model.config
        self.decoder = model.get_decoder()
        self.lm_head = model.get_output_embeddings()
        self.final_logits_bias = getattr(model, "final_logits_bias", None)

    def forward(
        self, input_ids, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None, past_key_values=None
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

        return lm_logits, decoder_outputs.past_key_values
