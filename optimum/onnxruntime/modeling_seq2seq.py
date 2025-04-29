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
"""
ORTModelForXXX classes related to seq2seq, allowing to run ONNX Models with ONNX Runtime using the same API as Transformers.
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    GenerationConfig,
    GenerationMixin,
    Pix2StructForConditionalGeneration,
    WhisperForConditionalGeneration,
)
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
from transformers.utils import cached_file

from onnxruntime import InferenceSession, SessionOptions

from ..exporters.onnx import main_export
from ..onnx.utils import _get_external_data_paths
from ..utils import NormalizedConfigManager, is_transformers_version
from ..utils.file_utils import validate_file_exists
from ..utils.logging import get_logger, warn_once
from ..utils.save_utils import maybe_save_preprocessors
from .base import ORTSessionMixin
from .constants import (
    DECODER_MERGED_ONNX_FILE_PATTERN,
    DECODER_ONNX_FILE_PATTERN,
    DECODER_WITH_PAST_ONNX_FILE_PATTERN,
    ENCODER_ONNX_FILE_PATTERN,
)
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    DummyWhisperModel,
    parse_device,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = get_logger(__name__)


SEQ2SEQ_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
"""

SPEECH_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor`):
            Mel / fbank features extracted from the raw speech waveform. `(batch_size, feature_size, encoder_sequence_length)`.
"""

VISION_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor`):
            Features extracted from an Image. This tensor should be of shape `(batch_size, num_channels, height, width)`.
"""

PIX2STRUCT_INPUTS_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_channels x patch_height x patch_width)`):
            Flattened and padded pixel values.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding pixel values.
"""

DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor`):
            Mel features extracted from the raw speech waveform.
            `(batch_size, feature_size, encoder_sequence_length)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

VISION_ENCODER_DECODER_SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor`):
            Features extracted from an Image. This tensor should be of shape
            `(batch_size, num_channels, height, width)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

PIX2STRUCT_ONNX_MODEL_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, seq_length, hidden_size)`):
            Flattened pixel patches. the `hidden_size` is obtained by the following formula: `hidden_size` =
            `num_channels` * `patch_size` * `patch_size`
            The process of flattening the pixel patches is done by `Pix2StructProcessor`.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.
            Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

_TOKENIZER_FOR_DOC = "AutoTokenizer"
_PROCESSOR_FOR_DOC = "AutoProcessor"
_IMAGE_PROCESSER_FOR_DOC = "AutoImageProcessor"

TRANSLATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Eustache and I like to", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_translation = pipeline("translation_en_to_de", model=model, tokenizer=tokenizer)

    >>> text = "My name is Eustache."
    >>> pred = onnx_translation(text)
    ```
"""


AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor.feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> gen_tokens = model.generate(inputs=inputs.input_features)
    >>> outputs = processor.tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> speech_recognition = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> pred = speech_recognition(ds[0]["audio"]["array"])
    ```
"""


IMAGE_TO_TEXT_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}, {tokenizer_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from PIL import Image
    >>> import requests


    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> tokenizer = {tokenizer_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> inputs = processor(image, return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, {tokenizer_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}
    >>> from PIL import Image
    >>> import requests


    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> tokenizer = {tokenizer_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_to_text = pipeline("image-to-text", model=model, tokenizer=tokenizer, feature_extractor=processor, image_processor=processor)
    >>> pred = image_to_text(image)
    ```
"""

PIX2STRUCT_EXAMPLE = r"""
    Example of pix2struct:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from PIL import Image
    >>> import requests

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True, use_io_binding=True)

    >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
    >>> inputs = processor(images=image, text=question, return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = processor.batch_decode(gen_tokens, skip_special_tokens=True)
    ```
"""


class ORTEncoder(ORTSessionMixin):
    """
    Encoder of an encoder-decoder model for ONNX Runtime inference.
    """

    main_input_name = "input_ids"

    def __init__(self, session: InferenceSession, parent_model: ORTModel):
        self.initialize_ort_attributes(session, parent_model.use_io_binding)

        self.parent_model = parent_model
        config = (
            self.parent_model.config.encoder
            if hasattr(self.parent_model.config, "encoder")
            else self.parent_model.config
        )
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> BaseModelOutput:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.parent_model.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoderForSeq2Seq(ORTSessionMixin):
    """
    Decoder of an encoder-decoder model for ONNX Runtime inference.
    """

    main_input_name = "input_ids"

    def __init__(self, session: InferenceSession, parent_model: ORTModel):
        self.initialize_ort_attributes(session, parent_model.use_io_binding)

        self.parent_model = parent_model
        config = (
            self.parent_model.config.decoder
            if hasattr(self.parent_model.config, "decoder")
            else self.parent_model.config
        )

        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)

        # TODO: make this less hacky.
        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]

        # To handle the old case when past_key_values were following the format: past_key_values_{idx}
        if len(self.key_value_input_names) == 0:
            self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        if len(self.key_value_output_names) == 0:
            self.key_value_output_names = [key for key in self.output_names if "key_values" in key]

        if self.parent_model.use_cache is True and len(self.key_value_output_names) == 0:
            raise RuntimeError("Could not find the past key values in the provided model.")

        self.use_past_in_outputs = len(self.key_value_output_names) > 0
        self.use_past_in_inputs = len(self.key_value_input_names) > 0
        self.use_fp16 = self.dtype == torch.float16

        # We may use ORTDecoderForSeq2Seq for vision-encoder-decoder models, where models as gpt2
        # can be used but do not support KV caching for the cross-attention key/values, see:
        # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/gpt2/modeling_gpt2.py#L302-L311
        # This attribute is used to avoid returning cross-attention KV-cache in this case.
        self.no_cross_attention_cache = getattr(self.parent_model, "no_cross_attention_cache", False)

        if (not self.parent_model.use_merged and self.use_past_in_inputs) or self.no_cross_attention_cache:
            self.num_pkv = 2
        else:
            # When using a merged model, we always have the same number of output whether we use past key values or not,
            # and in the case past key values are used, empty tensors are given as cross-attention past key values as they
            # are constants
            self.num_pkv = 4

        self.past_key_values_cross_attention_output_names = set()
        for output_name in self.output_names:
            if output_name.startswith("present") and "encoder" in output_name:
                self.past_key_values_cross_attention_output_names.add(output_name)

        self.use_legacy_outputs = (
            self.parent_model.use_merged is False and len(self.past_key_values_cross_attention_output_names) > 0
        )

    def compute_past_key_values_output_shapes(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        use_cache_branch: Optional[bool],
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Dict[str, int]:
        batch_size = input_ids.size(0)

        num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads

        sequence_length = input_ids.size(1)
        encoder_sequence_length = encoder_hidden_states.size(1)
        if past_key_values is not None and use_cache_branch is not False:
            # Here, use_cache_branch may be None in the case of separate decoder without/with past, or True if the with past branch
            # of a merged decoder is used
            sequence_length += past_key_values[0].size(2)

        self_attn_shape = (batch_size, num_attention_heads, sequence_length, embed_size_per_head)

        if past_key_values is not None and use_cache_branch is True:
            cross_attn_shape = (0, num_attention_heads, 1, embed_size_per_head)
        else:
            cross_attn_shape = (batch_size, num_attention_heads, encoder_sequence_length, embed_size_per_head)

        past_key_values_shapes = {}
        for idx, name in enumerate(self.key_value_output_names):
            is_self_attn = idx % 4 < 2
            # decoder with past does not ouput cross attention key/values as they are constants
            past_key_values_shapes[name] = self_attn_shape if (is_self_attn or self.num_pkv == 2) else cross_attn_shape
        return past_key_values_shapes

    def get_outputs_not_to_bind(self, use_merged_cache: bool) -> Set[str]:
        result = {
            output_name
            for output_name in self.output_names
            if (not output_name.startswith("present") and output_name not in {"loss", "logits"})
        }
        if use_merged_cache is True:
            # When using a merged decoder and the use cache branch, we output 0-dim tensors that IO Binding do not supp
            # Therefore, we do not bind them.
            result = result.union(self.past_key_values_cross_attention_output_names)
        return result

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Seq2SeqLMOutput:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )

        # no-ops if merged decoder is not used
        use_merged_no_cache = past_key_values is None and self.parent_model.use_merged
        use_merged_cache = past_key_values is not None and self.parent_model.use_merged
        use_cache_branch_tensor, past_key_values, cache_position = self.prepare_inputs_for_merged(
            input_ids, past_key_values, cache_position, use_torch=use_torch
        )

        model_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_attention_mask": encoder_attention_mask,
            "use_cache_branch": use_cache_branch_tensor,
            "cache_position": cache_position,
        }
        if past_key_values is not None:
            model_inputs.update(zip(self.key_value_input_names, past_key_values))

        if self.parent_model.use_io_binding:
            known_output_shapes = self.compute_past_key_values_output_shapes(
                input_ids,
                encoder_hidden_states,
                use_cache_branch=use_cache_branch_tensor.item() if use_cache_branch_tensor is not None else None,
                past_key_values=past_key_values,
            )
            outputs_to_not_bind = self.get_outputs_not_to_bind(use_merged_cache)

            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs,
                outputs_to_not_bind=outputs_to_not_bind,
                known_output_shapes=known_output_shapes,
            )

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            # Set -1 for sequence_length as it could be larger than the real sequence_length
            for name, shape in output_shapes.items():
                if name in self.key_value_output_names:
                    output_shapes[name] = shape[:2] + (-1,) + shape[3:]

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            out_past_key_values = ()
            for name in self.key_value_output_names:
                # TODO: this should be improved
                if name in self.past_key_values_cross_attention_output_names and use_merged_cache:
                    continue
                out_past_key_values += (output_buffers[name].view(output_shapes[name]),)

            logits = output_buffers["logits"].view(output_shapes["logits"])

            loss = None
            if "loss" in self.output_names:
                loss = output_buffers["loss"].view(output_shapes["loss"])

            if not self.use_past_in_outputs:
                out_past_key_values = None
            elif not self.use_past_in_inputs or use_merged_no_cache or self.no_cross_attention_cache:
                out_past_key_values = tuple(
                    out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
                )
            else:
                if self.use_legacy_outputs is True:
                    msg = (
                        "For the decoder with past, using ONNX models outputting cross attention past key values"
                        " is deprecated and the support will be removed in optimum 2.0. We recommend exporting again the model"
                        " with optimum>=1.7.3."
                    )
                    warn_once(logger, msg=msg)
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                # grab the cross attention key/values from the inputs
                elif self.num_pkv == 2:
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                elif self.num_pkv == 4:
                    # despite num_pkv being 4, we did not bind the cross-attention output
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + 2] + past_key_values[2 * i + 2 : 2 * i + 4]
                        for i in range(0, len(out_past_key_values), 2)
                    )
                else:
                    raise ValueError("Unsupported num_pkv")
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            # TODO: using a new variable out_past_key_values is memory inefficient,
            # past_key_values is not used anymore at this point
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            out_past_key_values = tuple(model_outputs[output_name] for output_name in self.key_value_output_names)

            loss = model_outputs.get("loss", None)
            logits = model_outputs["logits"]

            # TODO: this is extremely ugly and unreadable. What if cross-attention k/v change?
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to:
            # * 4 for the decoder without cache (k/v of self-attention + k/v of cross-attention)
            # * 2 for the decoder with cache (k/v of self-attention as cross-attention cache is constant)
            if not self.use_past_in_outputs:
                out_past_key_values = None
            elif not self.use_past_in_inputs or use_merged_no_cache or self.no_cross_attention_cache:
                out_past_key_values = tuple(
                    out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
                )
            else:
                if self.use_legacy_outputs is True:
                    msg = (
                        "For the decoder with past, using ONNX models outputting cross attention past key values"
                        " is deprecated and the support will be removed in optimum 2.0. We recommend exporting again the model"
                        " with optimum>=1.7.3."
                    )
                    warn_once(logger, msg=msg)
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                # grab the cross attention key/values from the inputs
                elif self.num_pkv == 2:
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + self.num_pkv]
                        + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                elif self.num_pkv == 4:
                    out_past_key_values = tuple(
                        out_past_key_values[i : i + 2] + past_key_values[i + 2 : i + 4]
                        for i in range(0, len(out_past_key_values), self.num_pkv)
                    )
                else:
                    raise ValueError("Unsupported num_pkv")

        return Seq2SeqLMOutput(loss=loss, logits=logits, past_key_values=out_past_key_values)

    def prepare_inputs_for_merged(
        self,
        input_ids: Optional[Union[torch.LongTensor, np.ndarray]],
        past_key_values: Optional[Tuple[Union[torch.FloatTensor, np.ndarray]]],
        cache_position: Optional[Union[torch.LongTensor, np.ndarray]],
        use_torch: bool,
    ):
        constructor = torch if use_torch is True else np

        if self.parent_model.use_merged:
            # Uses without/with branch of a merged decoder depending on whether real past key values are passed
            use_cache_branch_tensor = constructor.full((1,), past_key_values is not None)
            if use_torch and use_cache_branch_tensor is not None:
                use_cache_branch_tensor = use_cache_branch_tensor.to(self.device)
        else:
            use_cache_branch_tensor = None

        # Generate dummy past for the first forward if uses a merged decoder
        if self.parent_model.use_merged and past_key_values is None:
            batch_size = input_ids.shape[0]
            num_attention_heads = self.normalized_config.num_attention_heads
            embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads
            dtype = constructor.float16 if self.use_fp16 else constructor.float32
            shape = (batch_size, num_attention_heads, 1, embed_size_per_head)
            key_or_value = constructor.zeros(shape, dtype=dtype)

            if use_torch is True:
                key_or_value = key_or_value.to(self.device)

            past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))

        # Generate dummy position cache for the first forward if uses a merged decoder
        if self.parent_model.use_merged and cache_position is None:
            cache_position = constructor.zeros((1,), dtype=constructor.int64)
            if use_torch is True:
                cache_position = cache_position.to(self.device)

        return use_cache_branch_tensor, past_key_values, cache_position


class ORTEncoderForSpeech(ORTEncoder):
    """
    Encoder model for ONNX Runtime inference for Whisper model.

    Args:
        session (`InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    main_input_name = "input_features"

    @add_start_docstrings_to_model_forward(SPEECH_ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:
        use_torch = isinstance(input_features, torch.Tensor)
        self.parent_model.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }

        if self.parent_model.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTEncoderForVisionEncoderDecoder(ORTEncoder):
    """
    Encoder model for ONNX Runtime inference for VisionEncoderDecoder models.

    Args:
        session (`InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    main_input_name = "pixel_values"

    @add_start_docstrings_to_model_forward(VISION_ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> BaseModelOutput:
        use_torch = isinstance(pixel_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "pixel_values": pixel_values,
        }

        if self.parent_model.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTEncoderForPix2Struct(ORTEncoder):
    """
    Encoder model for ONNX Runtime inference for Pix2Struct.

    Args:
        session (`InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    main_input_name = "flattened_patches"

    @add_start_docstrings_to_model_forward(PIX2STRUCT_INPUTS_DOCSTRING)
    def forward(
        self,
        flattened_patches: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:
        use_torch = isinstance(flattened_patches, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "flattened_patches": flattened_patches,
            "attention_mask": attention_mask,
        }

        if self.parent_model.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


# TODO: should use ORTSessionsWrapper as it contains multiple sessions
# TODO: should not inherit ORTModel as it rewrites all of its save/load/push logic
class ORTModelForConditionalGeneration(ORTModel):
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.

    Important attributes:
        config ([`PretrainedConfig`]):
            Instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
        use_io_binding (`Optional[bool]`, defaults to `None`):
            Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to `True`
            if the device is CUDA, otherwise defaults to `False`.
        use_cache (`bool`):
            Whether or not past key/values cache should be used. It is determined by whether an InferenceSession for
            that was provided or not.
        providers (`List[str`]):
            The list of execution providers the model is running on.
        encoder (`ORTEncoder`):
            The encoder model.
        decoder (`ORTDecoderForSeq2Seq`):
            The decoder model.
        decoder_with_past (`Optional[ORTDecoderForSeq2Seq]`):
            The decoder model handling the past key/values if `use_cache=True`, else `None`.

    Other attributes:
        encoder_file_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_ENCODER_NAME`):
            The name of the ONNX file containing the encoder part of the model.
        decoder_file_name (`str`,  defaults to `optimum.onnxruntime.utils.ONNX_DECODER_NAME`):
            The name of the ONNX file containing the decoder part of the model.
        decoder_file_with_past_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_WITH_PAST_NAME`):
            The name of the ONNX file containing the decoder with past key/values part of the model.
        model_save_dir (`str`, defaults to `""`):
            The directory under which the model exported to ONNX was saved.

    """

    _supports_cache_class = False

    def __init__(
        self,
        encoder_session: InferenceSession,
        decoder_session: InferenceSession,
        config: "PretrainedConfig",
        onnx_paths: List[str],
        decoder_with_past_session: Optional[InferenceSession] = None,
        use_cache: bool = True,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        """
        Args:
            encoder_session (`InferenceSession`):
                The ONNX Runtime inference session associated to the encoder.
            decoder_session (`InferenceSession`):
                The ONNX Runtime inference session associated to the decoder.
            config ([`PretrainedConfig`]):
                `config` is an instance of the configuration associated to the model. Initializing with a config file
                does not load the weights associated with the model, only the configuration.
            onnx_paths (`List[str]`):
                Path to ONNX files associated with the model.
            decoder_with_past_session (`Optional[InferenceSession]`, *optional*):
                The ONNX Runtime inference session associated to the decoder with past key values.
            use_io_binding (`bool`, *optional*, defaults to `None`):
                Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            model_save_dir (`str`, *optional*, defaults to `""`):
                The directory under which the model exported to ONNX was saved.
            preprocessors (`Optional[List]`, defaults to `None`):
                The list of the preprocessors (tokenizer, processor, feature_extractor) to save alongside the ORTModel.
            generation_config (`Optional[GenerationConfig]`, defaults to `None`):
                The generation configuration used by default when calling `generate()`.
                Refer to https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate.
        """
        super().__init__(
            model=encoder_session,
            config=config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
        )

        self.name_or_path = config.name_or_path
        self.onnx_paths = onnx_paths
        self.use_cache = use_cache

        if use_cache is True:
            # Auto-detect whether the provided session is a merged non-past / with-past or not
            # TODO: make __init__ private and pass `use_merged` as an argument
            use_merged = "use_cache_branch" in [inp.name for inp in decoder_session.get_inputs()]

            if use_merged is True and decoder_with_past_session is not None:
                raise ValueError(
                    "Detected a merged decoder, but decoder_with_past_session was provided."
                    "Please only set decoder_session, or provide a non-merged decoder_session."
                )
            if use_cache is True and use_merged is False and decoder_with_past_session is None:
                raise ValueError(
                    "The parameter use_cache was set as True, but neither decoder_with_past_session was passed"
                    " nor a use_cache branch can be found in the decoder_session."
                    " Please pass a decoder_with_past_session or set use_cache=False."
                )
        else:
            use_merged = False

            if decoder_with_past_session is not None:
                raise ValueError(
                    "The parameter decoder_with_past_session was passed, although use_cache is False."
                    "Please pass use_cache=True for decoder_with_past_session to be used."
                )

        if use_cache is False and use_io_binding is True:
            raise ValueError(
                "When using CUDAExecutionProvider, the parameters combination use_cache=False, use_io_binding=True"
                " is not supported. Please either pass use_cache=True, use_io_binding=True (default),"
                " or use_cache=False, use_io_binding=False."
            )

        self.use_merged = use_merged

        self.encoder = self._initialize_encoder(encoder_session)
        self.encoder_model_path = Path(encoder_session._model_path)
        self.encoder_model_name = self.encoder_model_path.name

        self.decoder = ORTDecoderForSeq2Seq(decoder_session, self)
        self.decoder_model_path = Path(decoder_session._model_path)
        self.decoder_model_name = self.decoder_model_path.name

        # If a decoder_with_past_path is provided, an inference session for the decoder with past key/values as inputs
        # will be enabled
        self.decoder_with_past = None
        self.decoder_with_past_model_path = None
        self.decoder_with_past_model_name = None
        if self.use_cache is True and self.use_merged is False:
            self.decoder_with_past = ORTDecoderForSeq2Seq(decoder_with_past_session, self)
            self.decoder_with_past_model_path = Path(decoder_with_past_session._model_path)
            self.decoder_with_past_model_name = self.decoder_with_past_model_path.name

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config

        if is_transformers_version(">=", "4.44.99"):
            misplaced_generation_parameters = self.config._get_non_default_generation_parameters()
            if len(misplaced_generation_parameters) > 0:
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(self.generation_config, param_name, param_value)
                    setattr(self.config, param_name, None)

    def _initialize_encoder(self, session: InferenceSession) -> ORTEncoder:
        return ORTEncoder(session, self)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model encoder, decoder and decoder with past key values as well as its configuration file to a
        directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_seq2seq.ORTModelForSeq2SeqLM.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path`]):
                The directory where to save the model files.
        """
        save_directory = Path(save_directory)
        src_paths = [Path(path) for path in self.onnx_paths]
        dst_paths = [save_directory / path.name for path in src_paths]

        # add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copyfile(src_path, dst_path)

        self.generation_config.save_pretrained(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        # hub options
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        encoder_file_name: str = ONNX_ENCODER_NAME,
        decoder_file_name: str = ONNX_DECODER_NAME,
        decoder_with_past_file_name: str = ONNX_DECODER_WITH_PAST_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        # session options
        provider: str = "CPUExecutionProvider",
        providers: Optional[Sequence[str]] = None,
        provider_options: Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
        session_options: Optional[SessionOptions] = None,
        # inference options
        use_cache: bool = True,
        use_merged: Optional[bool] = None,
        use_io_binding: Optional[bool] = None,
        # other arguments
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        model_path = Path(model_id)

        # We do not implement the logic for use_cache=False, use_merged=True
        if use_cache is False:
            if use_merged is True:
                raise ValueError(
                    "The parameters combination use_cache=False, use_merged=True is not supported."
                    " To use a merged decoder, past key values must be used."
                )
            use_merged = False

        decoder_merged_path = None
        # We use `is not False` here to include two cases: use_merged = None (in which case we auto-detect it),
        # and use_merged = True (explicitely specified by the user)
        if use_merged is not False:
            try:
                decoder_merged_path = ORTModelForConditionalGeneration.infer_onnx_filename(
                    model_id,
                    [DECODER_MERGED_ONNX_FILE_PATTERN],
                    argument_name=None,
                    subfolder=subfolder,
                    token=token,
                    revision=revision,
                )
                use_merged = True
                decoder_path = decoder_merged_path
            except FileNotFoundError as e:
                if use_merged is True:
                    raise FileNotFoundError(
                        "The parameter `use_merged=True` was passed to ORTModelForCausalLM.from_pretrained()"
                        " but no ONNX file for a merged decoder could be found in"
                        f" {str(Path(model_id, subfolder))}, with the error: {e}"
                    )
                use_merged = False

        decoder_without_past_path = None
        decoder_with_past_path = None
        if use_merged is False:
            if not validate_file_exists(
                model_id, decoder_file_name, subfolder=subfolder, revision=revision, token=token
            ):
                decoder_without_past_path = ORTModelForConditionalGeneration.infer_onnx_filename(
                    model_id,
                    [DECODER_ONNX_FILE_PATTERN],
                    "decoder_file_name",
                    subfolder=subfolder,
                    token=token,
                    revision=revision,
                )
            else:
                decoder_without_past_path = model_path / subfolder / decoder_file_name

            decoder_path = decoder_without_past_path

            decoder_regular_onnx_filenames = ORTModelForConditionalGeneration._generate_regular_names_for_filename(
                ONNX_DECODER_NAME
            )
            if decoder_path.name not in decoder_regular_onnx_filenames:
                logger.warning(
                    f"The ONNX file {decoder_path.name} is not a regular name used in optimum.onnxruntime that are {decoder_regular_onnx_filenames}, the "
                    f"{cls.__name__} might not behave as expected."
                )

            # If the decoder without / with past has been merged, we do not need to look for any additional file
            if use_cache is True and use_merged is False:
                if not validate_file_exists(
                    model_id, decoder_with_past_file_name, subfolder=subfolder, revision=revision, token=token
                ):
                    try:
                        decoder_with_past_path = ORTModelForConditionalGeneration.infer_onnx_filename(
                            model_id,
                            [DECODER_WITH_PAST_ONNX_FILE_PATTERN],
                            "decoder_with_past_file_name",
                            subfolder=subfolder,
                            token=token,
                            revision=revision,
                        )
                    except FileNotFoundError as e:
                        raise FileNotFoundError(
                            "The parameter `use_cache=True` was passed to ORTModelForCausalLM.from_pretrained()"
                            " but no ONNX file using past key values could be found in"
                            f" {str(Path(model_id, subfolder))}, with the error: {e}"
                        )
                else:
                    decoder_with_past_path = model_path / subfolder / decoder_with_past_file_name

                decoder_path = decoder_without_past_path

                decoder_with_past_regular_onnx_filenames = (
                    ORTModelForConditionalGeneration._generate_regular_names_for_filename(ONNX_DECODER_WITH_PAST_NAME)
                )

                if decoder_with_past_path.name not in decoder_with_past_regular_onnx_filenames:
                    logger.warning(
                        f"The ONNX file {decoder_with_past_path.name} is not a regular name used in optimum.onnxruntime that are {decoder_with_past_regular_onnx_filenames}, "
                        f"the {cls.__name__} might not behave as expected."
                    )

        if not validate_file_exists(model_id, encoder_file_name, subfolder=subfolder, revision=revision, token=token):
            encoder_path = ORTModelForConditionalGeneration.infer_onnx_filename(
                model_id,
                [ENCODER_ONNX_FILE_PATTERN],
                "encoder_file_name",
                subfolder=subfolder,
                token=token,
                revision=revision,
            )
        else:
            encoder_path = model_path / subfolder / encoder_file_name

        encoder_regular_onnx_filenames = ORTModelForConditionalGeneration._generate_regular_names_for_filename(
            ONNX_ENCODER_NAME
        )
        if encoder_path.name not in encoder_regular_onnx_filenames:
            logger.warning(
                f"The ONNX file {encoder_path.name} is not a regular name used in optimum.onnxruntime, the "
                "ORTModelForConditionalGeneration might not behave as expected."
            )

        if model_path.is_dir():
            new_model_save_dir = model_path
        else:
            attribute_name_to_filename = {
                "last_encoder_model_name": encoder_path.name,
                "last_decoder_model_name": decoder_path.name if use_merged is False else None,
                "last_decoder_with_past_model_name": (
                    decoder_with_past_path.name if (use_merged is False and use_cache is True) else None
                ),
                "last_decoder_merged_name": decoder_merged_path.name if use_merged is True else None,
            }
            paths = {}
            for attr_name, filename in attribute_name_to_filename.items():
                if filename is None:
                    continue
                model_cache_path = cached_file(
                    model_id,
                    filename=filename,
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                # try download external data
                try:
                    cached_file(
                        model_id,
                        subfolder=subfolder,
                        filename=filename + "_data",
                        token=token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                except EnvironmentError:
                    # If the external data file is not found, we assume that the model is not using external data.
                    pass

                paths[attr_name] = Path(model_cache_path).name
            new_model_save_dir = Path(model_cache_path).parent

            if use_merged is True:
                decoder_path = new_model_save_dir / paths["last_decoder_merged_name"]
                decoder_merged_path = new_model_save_dir / paths["last_decoder_merged_name"]
            else:
                decoder_path = new_model_save_dir / paths["last_decoder_model_name"]
                decoder_without_past_path = new_model_save_dir / paths["last_decoder_model_name"]

                if use_cache is True:
                    decoder_with_past_path = new_model_save_dir / paths["last_decoder_with_past_model_name"]

            encoder_path = new_model_save_dir / paths["last_encoder_model_name"]

        encoder_session = InferenceSession(
            encoder_path,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        decoder_session = InferenceSession(
            decoder_path,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )

        decoder_with_past_session = None
        if decoder_with_past_path is not None and (use_merged is False and use_cache is True):
            decoder_with_past_session = InferenceSession(
                decoder_with_past_path,
                providers=providers,
                provider_options=provider_options,
                session_options=session_options,
            )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )

        onnx_paths = [encoder_path]
        if use_merged is False:
            onnx_paths.append(decoder_without_past_path)
            if use_cache is True:
                onnx_paths.append(decoder_with_past_path)
        else:
            onnx_paths.append(decoder_merged_path)

        return cls(
            config=config,
            encoder_session=encoder_session,
            decoder_session=decoder_session,
            decoder_with_past_session=decoder_with_past_session,
            generation_config=generation_config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            onnx_paths=onnx_paths,
            use_cache=use_cache,
        )

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        # export options
        task: Optional[str] = None,
        # hub options
        token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        # inference options
        use_cache: bool = True,
        use_merged: bool = False,
        **kwargs,
    ) -> "ORTModelForConditionalGeneration":
        if use_cache is False and use_merged is True:
            raise ValueError(
                "The incompatible arguments use_cache=False, use_merged=True were passed to"
                " ORTModelForConditionalGeneration.from_pretrained(). Please pass either use_cache=False,"
                " use_merged=False to disable past key value caching, or use_cache=True, use_merged=False"
                " to disable the merging of the decoder not using / using past key and value."
            )

        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

            if use_cache is True:
                task = task + "-with-past"

        model_save_dir = TemporaryDirectory()
        model_save_path = Path(model_save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=model_save_path,
            task=task,
            do_validation=False,
            no_post_process=not use_merged,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        maybe_save_preprocessors(model_id, model_save_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            model_save_path,
            config,
            use_cache=use_cache,
            use_merged=use_merged,
            model_save_dir=model_save_dir,
            **kwargs,
        )

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the model.
        """

        return self.encoder.dtype or self.decoder.dtype

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the device of the model. This method will move the encoder and decoder to the requested device.

        Args:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModelForConditionalGeneration`: The model with the encoder and decoder moved to the requested device.
        """
        device, _ = parse_device(device)

        self.encoder.to(device)
        self.decoder.to(device)
        if self.decoder_with_past is not None:
            self.decoder_with_past.to(device)

        self._device = device

        return self


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForSeq2SeqLM(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports bart, blenderbot, blenderbot_small, longt5, m2m_100, marian, mbart, mt5, pegasus, t5.
    """

    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"

    def _initialize_encoder(self, session: InferenceSession) -> ORTEncoder:
        return ORTEncoder(session, self)

    @add_start_docstrings_to_model_forward(
        SEQ2SEQ_ONNX_MODEL_DOCSTRING
        + TRANSLATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForSeq2SeqLM",
            checkpoint="optimum/t5-small",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        model = (
            self.decoder
            if past_key_values is None or not self.use_cache or self.use_merged
            else self.decoder_with_past
        )
        decoder_outputs = model(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )

        return Seq2SeqLMOutput(
            loss=decoder_outputs.get("loss", None),
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self) -> ORTEncoder:
        return self.encoder

    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        reordered_past = ()
        for layer_past in past:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForSpeechSeq2Seq(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Speech Sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports whisper, speech_to_text.
    """

    main_input_name = "input_features"
    auto_model_class = AutoModelForSpeechSeq2Seq

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Following a breaking change in transformers that relies directly on the mapping name and not on the
        # greedy model mapping (that can be extended), we need to hardcode the ortmodel in this dictionary.
        # Other pipelines do not seem to have controlflow depending on the mapping name.
        # See: https://github.com/huggingface/transformers/pull/24960/files
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES["ort_speechseq2seq"] = self.__class__.__name__

    def _initialize_encoder(self, session: InferenceSession) -> ORTEncoder:
        return ORTEncoderForSpeech(session, self)

    def get_encoder(self) -> ORTEncoder:
        return self.encoder

    @add_start_docstrings_to_model_forward(
        SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING
        + AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForSpeechSeq2Seq",
            checkpoint="optimum/whisper-tiny.en",
        )
    )
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)

        model = (
            self.decoder
            if past_key_values is None or not self.use_cache or self.use_merged
            else self.decoder_with_past
        )
        decoder_outputs = model(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            cache_position=cache_position,
        )

        return Seq2SeqLMOutput(
            loss=decoder_outputs.get("loss", None),
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        reordered_past = ()
        for layer_past in past:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    @classmethod
    def _from_pretrained(cls, model_id: Union[str, Path], config: "PretrainedConfig", **kwargs):
        if config.model_type == "whisper":
            return _ORTModelForWhisper._from_pretrained(model_id, config, **kwargs)
        else:
            return super()._from_pretrained(model_id, config, **kwargs)


class _ORTModelForWhisper(ORTModelForSpeechSeq2Seq, WhisperForConditionalGeneration):
    """
    Whisper implements its own generate() method.
    """

    auto_model_class = WhisperForConditionalGeneration

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = DummyWhisperModel()

    # force the use of the WhisperForConditionalGeneration generate and prepare_inputs_for_generation methods
    def generate(*args, **kwargs):
        return WhisperForConditionalGeneration.generate(*args, **kwargs)

    # force the use of the WhisperForConditionalGeneration prepare_inputs_for_generation method
    def prepare_inputs_for_generation(*args, **kwargs):
        return WhisperForConditionalGeneration.prepare_inputs_for_generation(*args, **kwargs)

    # this is needed to avoid circular calls
    @classmethod
    def _from_pretrained(cls, model_id: Union[str, Path], config: "PretrainedConfig", **kwargs):
        return super(ORTModelForSpeechSeq2Seq, cls)._from_pretrained(model_id, config, **kwargs)


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForVision2Seq(ORTModelForConditionalGeneration, GenerationMixin):
    """
    VisionEncoderDecoder Sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports trocr and vision-encoder-decoder.
    """

    auto_model_class = AutoModelForVision2Seq
    main_input_name = "pixel_values"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # There are probably other archs that do not support cross attention KV cache, but only
        # this one seem popular on the Hub.
        if self.config.decoder.model_type == "gpt2":
            self.no_cross_attention_cache = True

    def _initialize_encoder(self, session: InferenceSession) -> ORTEncoder:
        return ORTEncoderForVisionEncoderDecoder(session, self)

    @add_start_docstrings_to_model_forward(
        VISION_ENCODER_DECODER_SEQ2SEQ_ONNX_MODEL_DOCSTRING
        + IMAGE_TO_TEXT_EXAMPLE.format(
            processor_class=_IMAGE_PROCESSER_FOR_DOC,
            tokenizer_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForVision2Seq",
            checkpoint="nlpconnect/vit-gpt2-image-captioning",
        )
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(pixel_values=pixel_values)

        model = (
            self.decoder
            if past_key_values is None or not self.use_cache or self.use_merged
            else self.decoder_with_past
        )

        decoder_outputs = model(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self) -> ORTEncoder:
        return self.encoder

    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        reordered_past = ()
        for layer_past in past:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForPix2Struct(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Pix2struct model with a language modeling head for ONNX Runtime inference. This class officially supports pix2struct.
    """

    # pix2struct cannot be loaded using AutoModel
    auto_model_class = Pix2StructForConditionalGeneration
    main_input_name = "flattened_patches"

    def _initialize_encoder(self, session: InferenceSession) -> ORTEncoder:
        return ORTEncoderForPix2Struct(session, self)

    @add_start_docstrings_to_model_forward(
        PIX2STRUCT_ONNX_MODEL_DOCSTRING
        + PIX2STRUCT_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForPix2Struct",
            checkpoint="google/pix2struct-ai2d-base",
        )
    )
    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
            )

        model = (
            self.decoder
            if self.use_merged or not self.use_cache or past_key_values is None
            else self.decoder_with_past
        )

        decoder_outputs = model(
            input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self) -> ORTEncoder:
        return self.encoder

    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        ORTModelForSeq2SeqLM._reorder_cache(past, beam_idx)
