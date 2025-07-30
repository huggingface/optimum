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
"""Classes handling causal-lm related architectures in ONNX Runtime."""

import logging
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import onnx
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from onnx.tools import update_model_dims
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import cached_file

from onnxruntime import InferenceSession, SessionOptions

from ..exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from ..exporters.tasks import TasksManager
from ..onnx.utils import check_model_uses_external_data
from ..utils import is_transformers_version
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_save_preprocessors
from .constants import (
    DECODER_MERGED_ONNX_FILE_PATTERN,
    DECODER_ONNX_FILE_PATTERN,
    DECODER_WITH_PAST_ONNX_FILE_PATTERN,
    ONNX_FILE_PATTERN,
)
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import prepare_providers_and_provider_options


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if is_transformers_version(">=", "4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin  # type: ignore # noqa: F401


logger = logging.getLogger(__name__)

DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

CAUSALLM_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

_TOKENIZER_FOR_DOC = "AutoTokenizer"

TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    >>> text = "My name is Arthur and I live in"
    >>> gen = onnx_gen(text)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForCausalLM(ORTModel, GenerationMixin):
    """
    ONNX model with a causal language modeling head for ONNX Runtime inference. This class officially supports bloom, codegen, falcon, gpt2, gpt-bigcode, gpt_neo, gpt_neox, gptj, llama.
    """

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(
        self,
        *args,
        config: "PretrainedConfig" = None,
        session: "InferenceSession" = None,
        use_io_binding: Optional[bool] = None,
        generation_config: Optional["GenerationConfig"] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        # DEPRECATED BEHAVIOR
        if args:
            logger.warning(
                "Instantiating an ORTModelForCausalLM with positional arguments is deprecated and will be removed in the next version. "
                "Please use the keywords arguments {config, session, use_io_binding, generation_config, model_save_dir, use_cache} instead."
            )
            # the old signature is ORTModelForCausalLM(model, config, use_io_binding, model_save_dir, preprocessors, generation_config, use_cache)
            session = args[0]
            if len(args) > 1:
                config = args[1]
            if len(args) > 2:
                use_io_binding = args[2]
            if len(args) > 3:
                model_save_dir = args[3]
            if len(args) > 4:
                _ = args[4]
            if len(args) > 5:
                generation_config = args[5]
            if len(args) > 6:
                _ = args[6]

        if kwargs.get("model", None) is not None:
            logger.warning(
                "Passing the inference session as `model` argument to an ORTModelForCausalLM is deprecated. Please use `session` instead."
            )
            session = kwargs.pop("model")
        if kwargs:
            logger.warning(
                f"Some keyword arguments were passed to the ORTModelForCausalLM constructor that are not part of its signature: {', '.join(kwargs.keys())}. "
                "These arguments will be ignored in the current version and will raise an error in the next version."
            )

        if config is None:
            raise ValueError(
                "The parameter config is required. Please pass a config or use the from_pretrained method."
            )
        if session is None:
            raise ValueError(
                "The parameter session is required. Please pass a session or use the from_pretrained method."
            )
        ## END OF DEPRECATED BEHAVIOR
        super().__init__(config=config, session=session, use_io_binding=use_io_binding, model_save_dir=model_save_dir)

        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        self.can_use_cache = len(self.key_value_input_names) > 0 and len(self.key_value_output_names) > 0
        self.is_merged = "use_cache_branch" in self.input_names
        self.generation_config = generation_config

        # Reference: https://github.com/huggingface/optimum/pull/1381
        model_type = self.config.model_type
        if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS and "position_ids" not in self.input_names:
            logger.warning(
                f"ORTModelForCausalLM loaded a legacy ONNX model with no position_ids input, although the model type {model_type} "
                "requires it. for correct batched generation. We strongly encourage to re-export the model with "
                "a newer version of Optimum for better performance and more reliable generation. "
            )

        if not self.can_use_cache and self.generation_config.use_cache:
            logger.warning(
                "`model.generation_config.use_cache=True` but the loaded model does not support using the past key values cache."
                "Please re-export the original model once again with `use_cache=True` to be able to use it during generation. "
                "Or set `model.generation_config.use_cache=False` to avoid errors from attempting to use the cache. "
                "To re-export your model, simply set `export=True` as in `from_pretrained(..., export=True, use_cache=True)`."
            )

        if self.config.model_type == "gemma":
            self.embed_size_per_head = self.config.head_dim
        elif self.config.model_type == "gpt_bigcode":
            self.embed_size_per_head = self.config.hidden_size // self.config.num_attention_heads * 2
        else:
            self.embed_size_per_head = self.config.hidden_size // self.config.num_attention_heads

        if self.config.model_type in {
            "gemma",
            "mistral",
            "llama",
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "granite",
            "smollm3",
        }:
            self.num_key_value_heads = self.config.num_key_value_heads
        elif self.config.model_type == "falcon":
            if self.config.new_decoder_architecture or not self.config.multi_query:
                self.num_key_value_heads = self.config.num_kv_heads
            else:
                self.num_key_value_heads = 1
        else:
            self.num_key_value_heads = self.config.num_attention_heads

        self.old_bloom_modeling = (
            self.input_shapes.get("past_key_values.0.key", None) is not None
            and self.input_shapes.get("past_key_values.0.value", None) is not None
            and self.input_shapes["past_key_values.0.key"] != self.input_shapes["past_key_values.0.value"]
        )

    @property
    def use_cache(self):
        logger.warning(
            "The `ORTModelForCausalLM.use_cache` property is deprecated and will be removed in a future version. "
            "Please rather use `ORTModelForCausalLM.can_use_cache` to check if a model supports using cache during generation. "
            "And use `ORTModelForCausalLM.generation_config.use_cache` to check if the model is configured to use cache during generation."
        )
        return self.can_use_cache

    @property
    def use_merged(self):
        logger.warning(
            "The `ORTModelForCausalLM.use_merged` property is deprecated and will be removed in a future version. "
            "Please rather use `ORTModelForCausalLM.is_merged` to check if the underlying model is merged or not."
        )
        return self.is_merged

    @add_start_docstrings_to_model_forward(
        CAUSALLM_ONNX_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForCausalLM",
            checkpoint="optimum/gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                f"`use_cache={use_cache}` was passed to the model but the loaded model only supports `use_cache={self.can_use_cache}`. "
                f"Please load your current model with `use_cache={self.can_use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To re-export your model, simply set `export=True` in the `from_pretrained` method."
            )

        # Compute dimensions that will be used afterwards
        batch_size, seq_len = input_ids.shape
        if past_key_values is not None:
            if self.config.model_type == "gpt_bigcode":
                if self.config.multi_query:
                    pkv_seq_len = past_key_values[0].shape[1]
                else:
                    pkv_seq_len = past_key_values[0].shape[2]
            else:
                pkv_seq_len = past_key_values[0][0].shape[2]
        else:
            pkv_seq_len = 0

        if position_ids is None and "position_ids" in self.input_names:
            if self.config.model_type == "opt":
                if attention_mask is not None:
                    # OPT models use a different way to infer position_ids from attention_mask
                    position_ids = attention_mask.cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, -1)
                    position_ids = position_ids[:, pkv_seq_len:]
                else:
                    raise ValueError(
                        "The model OPT requires position_ids for batched generation but none were provided. "
                        "Please provide position_ids or attention_mask (from which position_ids can be inferred)."
                    )
            elif self.config.model_type == "gpt_bigcode":
                if attention_mask is not None:
                    # GPT BigCode models use a different way to infer position_ids from attention_mask
                    position_ids = attention_mask.cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids[:, pkv_seq_len:]
                else:
                    raise ValueError(
                        "The model gpt_bigcode requires position_ids for batched generation but none were provided. "
                        "Please provide position_ids or attention_mask (from which position_ids can be inferred)."
                    )
            else:
                # Create position_ids from input_ids
                position_ids = (
                    torch.arange(pkv_seq_len, pkv_seq_len + seq_len, dtype=torch.long, device=input_ids.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

        use_cache_branch = None
        if self.is_merged:
            # Uses cache branch of merged decoders depending on whether real past key values are passed
            use_cache_branch = torch.full((1,), past_key_values is not None, dtype=torch.bool, device=self.device)

        if len(self.key_value_input_names) > 0:
            if past_key_values is None:
                # Generates the input pkv for the first forward of the model (merged or with past)
                if self.config.model_type == "gpt_bigcode" and self.config.multi_query:
                    k_shape = v_shape = (batch_size, 0, self.embed_size_per_head)
                elif self.config.model_type == "bloom" and self.old_bloom_modeling:
                    k_shape = (batch_size * self.num_key_value_heads, self.embed_size_per_head, 0)
                    v_shape = (batch_size * self.num_key_value_heads, 0, self.embed_size_per_head)
                else:
                    k_shape = v_shape = (batch_size, self.num_key_value_heads, 0, self.embed_size_per_head)
                k_tensor = torch.zeros(k_shape, dtype=self.dtype, device=self.device)
                v_tensor = torch.zeros(v_shape, dtype=self.dtype, device=self.device)
                past_key_values = tuple(
                    k_tensor if ".key" in name else v_tensor for name in self.key_value_input_names
                )
            elif isinstance(past_key_values[0], tuple):
                # Flattens the past_key_values to a single tuple if it is a tuple of tuples
                past_key_values = sum(past_key_values, ())

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache_branch": use_cache_branch,
        }
        if len(self.key_value_input_names) > 0:
            model_inputs.update(zip(self.key_value_input_names, past_key_values))

        known_output_shapes = None
        outputs_to_not_bind = None
        if use_cache and self.use_io_binding:
            # Infers the shape of the output pkv
            batch_size, seq_len = input_ids.shape
            if self.config.model_type == "gpt_bigcode" and self.config.multi_query:
                embed_size_per_head = past_key_values[0].shape[-1]
                k_shape = v_shape = (batch_size, pkv_seq_len + seq_len, embed_size_per_head)
            elif self.config.model_type == "bloom" and self.old_bloom_modeling:
                num_key_value_heads_batch_size, embed_size_per_head = past_key_values[0].shape[:2]
                k_shape = (num_key_value_heads_batch_size, embed_size_per_head, pkv_seq_len + seq_len)
                v_shape = (num_key_value_heads_batch_size, pkv_seq_len + seq_len, embed_size_per_head)
            else:
                embed_size_per_head = past_key_values[0].shape[-1]
                k_shape = v_shape = (batch_size, self.num_key_value_heads, pkv_seq_len + seq_len, embed_size_per_head)
            known_output_shapes = {
                name: k_shape if ".key" in name else v_shape for name in self.key_value_output_names
            }
        else:
            # Don't bind the output pkv if not necessary
            outputs_to_not_bind = self.key_value_output_names

        if self.use_io_binding:
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

            loss = output_buffers.get("loss", None)
            logits = output_buffers["logits"].view(output_shapes["logits"])

            if use_cache:
                past_key_values = tuple(
                    output_buffers.pop(name).view(output_shapes[name]) for name in self.key_value_output_names
                )
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            loss = model_outputs.pop("loss", None)
            logits = model_outputs.pop("logits")

            if use_cache:
                past_key_values = tuple(model_outputs.pop(name) for name in self.key_value_output_names)

        if use_cache and self.config.model_type != "gpt_bigcode":
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and per decoder layer
            past_key_values = tuple(past_key_values[i : i + 2] for i in range(0, len(past_key_values), 2))

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if is_transformers_version("<", "4.46.0"):
            return self._prepare_inputs_for_generation_legacy(*args, **kwargs)
        else:
            return super().prepare_inputs_for_generation(*args, **kwargs)

    # Adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM.prepare_inputs_for_generation
    def _prepare_inputs_for_generation_legacy(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if self.config.model_type == "gpt_bigcode":
                if self.config.multi_query:
                    pkv_seq_len = past_key_values[0].shape[1]
                else:
                    pkv_seq_len = past_key_values[0].shape[2]
            else:
                pkv_seq_len = past_key_values[0][0].shape[2]

            if input_ids.shape[1] > pkv_seq_len:
                remove_prefix_length = pkv_seq_len
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        # falcon, gpt_bigcode, and other models used to override the prepare_inputs_for_generation method to add this logic
        # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L1186
        if "position_ids" in self.input_names and position_ids is None and attention_mask is not None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        if isinstance(past_key_values, tuple) and isinstance(past_key_values[0], tuple):
            if past_key_values[0][0].shape != past_key_values[0][1].shape:
                batch_size_times_num_heads, head_dim, seq_length = past_key_values[0][0].shape
                num_heads = batch_size_times_num_heads // beam_idx.shape[0]
                batch_size = beam_idx.shape[0]

                return tuple(
                    (
                        layer_past[0]
                        .view(batch_size, num_heads, head_dim, seq_length)
                        .index_select(0, beam_idx.to(layer_past[0].device))
                        .view(batch_size * num_heads, head_dim, seq_length),
                        layer_past[1]
                        .view(batch_size, num_heads, seq_length, head_dim)
                        .index_select(0, beam_idx.to(layer_past[1].device))
                        .view(batch_size * num_heads, seq_length, head_dim),
                    )
                    for layer_past in past_key_values
                )
            else:
                # GPT2 style
                return tuple(
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                    for layer_past in past_key_values
                )
        elif isinstance(past_key_values, tuple) and isinstance(past_key_values[0], torch.Tensor):
            # GPT BigCode style
            return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)
        else:
            raise ValueError(
                f"Unexpected past_key_values: {past_key_values}. "
                "Expected tuple of tuples (GPT2 style) or tuple of tensors (GPT BigCode style)."
            )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        # file options
        file_name: Optional[str] = None,
        # session options
        provider: str = "CPUExecutionProvider",
        providers: Optional[Sequence[str]] = None,
        provider_options: Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
        session_options: Optional[SessionOptions] = None,
        # inference options
        use_cache: bool = True,
        use_merged: Optional[bool] = None,
        use_io_binding: Optional[bool] = None,
        generation_config: Optional[GenerationConfig] = None,
        # other arguments
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
    ) -> "ORTModelForCausalLM":
        onnx_files = find_files_matching_pattern(
            model_id,
            ONNX_FILE_PATTERN,
            glob_pattern="**/*.onnx",
            subfolder=subfolder,
            token=token,
            revision=revision,
        )

        if len(onnx_files) == 0:
            raise FileNotFoundError(f"Could not find any ONNX model file in {model_id}")

        if len(onnx_files) == 1:
            subfolder = onnx_files[0].parent
            _file_name = onnx_files[0].name
            if file_name and file_name != _file_name:
                raise FileNotFoundError(f"Trying to load {file_name} but only found {_file_name}")
            file_name = _file_name

        else:
            model_files = []
            # Check first for merged models and then for decoder / decoder_with_past models
            if use_merged is not False:
                model_files = [p for p in onnx_files if re.search(DECODER_MERGED_ONNX_FILE_PATTERN, str(p))]
                use_merged = len(model_files) != 0

            if use_merged is False:
                pattern = DECODER_WITH_PAST_ONNX_FILE_PATTERN if use_cache else DECODER_ONNX_FILE_PATTERN
                model_files = [p for p in onnx_files if re.search(pattern, str(p))]

            # if file_name is specified we don't filter legacy models
            if not model_files or file_name:
                model_files = onnx_files
            else:
                logger.warning(
                    f"Legacy models found in {model_files} will be loaded. "
                    "Legacy models will be deprecated in the next version of optimum, please re-export your model"
                )
            _file_name = model_files[0].name
            subfolder = model_files[0].parent

            defaut_file_name = file_name or "model.onnx"
            for file in model_files:
                if file.name == defaut_file_name:
                    _file_name = file.name
                    subfolder = file.parent
                    break

            file_name = _file_name

            if len(model_files) > 1:
                logger.warning(
                    f"Too many ONNX model files were found in {' ,'.join(map(str, model_files))}. "
                    "specify which one to load by using the `file_name` and/or the `subfolder` arguments. "
                    f"Loading the file {file_name} in the subfolder {subfolder}."
                )

        if os.path.isdir(model_id):
            model_id = subfolder
            subfolder = ""

        if isinstance(subfolder, Path):
            subfolder = subfolder.as_posix()

        model_cache_path = cached_file(
            model_id,
            filename=file_name,
            # hub options
            token=token,
            revision=revision,
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance, in which case we want to keep it
        # instead of the path only.
        if model_save_dir is None:
            model_save_dir = Path(model_cache_path).parent

        try:
            cached_file(
                model_id,
                filename=file_name + "_data",
                # hub options
                token=token,
                revision=revision,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        except EnvironmentError:
            # If the external data file is not found, we assume that the model is not using external data.
            pass

        # This should be removed at some point
        onnx_model = onnx.load(str(model_cache_path), load_external_data=False)
        model_uses_external_data = check_model_uses_external_data(onnx_model)
        if model_uses_external_data:
            onnx_model = onnx.load(str(model_cache_path), load_external_data=True)
        input_dims = {
            node.name: [dim.dim_value or dim.dim_param for dim in node.type.tensor_type.shape.dim]
            for node in onnx_model.graph.input
        }
        output_dims = {
            node.name: [dim.dim_value or dim.dim_param for dim in node.type.tensor_type.shape.dim]
            for node in onnx_model.graph.output
        }
        override_dims = False
        # Since v1.7.0 decoder with past models have fixed sequence length of 1
        # To keep these models compatible we set this dimension to dynamic
        if input_dims["input_ids"][1] == 1:
            input_dims["input_ids"][1] = "sequence_length"
            output_dims["logits"][1] = "sequence_length"
            override_dims = True
        # Since https://github.com/huggingface/optimum/pull/871/
        # changed axis notation/naming during export, we need to update the dims
        for input_name in input_dims.keys():
            if "past" in input_name and input_dims[input_name][2] == "past_sequence_length + sequence_length":
                input_dims[input_name][2] = "past_sequence_length"
                override_dims = True
        if override_dims:
            # this is kinda dangerous, warning the user is the least we can do
            logger.warning(
                "The ONNX model was probably exported with an older version of optimum. "
                "We are updating the input/output dimensions and overwriting the model file "
                "with new dimensions. This is necessary for the model to work correctly with "
                "the current version of optimum. If you encounter any issues, please re-export "
                "the model with the latest version of optimum for optimal performance."
            )
            onnx_model = update_model_dims.update_inputs_outputs_dims(onnx_model, input_dims, output_dims)
            onnx.save(
                onnx_model,
                str(model_cache_path),
                save_as_external_data=model_uses_external_data,
                location=Path(model_cache_path).name + "_data",
                all_tensors_to_one_file=True,
                convert_attribute=True,
                size_threshold=0,
            )
        del onnx_model

        # Important: for encoder-decoder models used with CausalLM, we need to set the is_decoder flag to True
        # and the is_encoder_decoder flag to False. This is needed for the model to work correctly with generation logic.
        if hasattr(config, "is_decoder"):
            config.is_decoder = True
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False

        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except OSError:
                logger.info("Generation config file not found, creating a new one from model config.")
                generation_config = GenerationConfig.from_model_config(config)

        # TODO: not sure if setting config.use_cache is needed for older versions of transformers
        generation_config.use_cache = use_cache
        config.use_cache = use_cache

        if is_transformers_version(">=", "4.45.0"):
            misplaced_generation_parameters = config._get_non_default_generation_parameters()
            if len(misplaced_generation_parameters) > 0:
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(generation_config, param_name, param_value)
                    setattr(config, param_name, None)

        providers, provider_options = prepare_providers_and_provider_options(
            provider=provider, providers=providers, provider_options=provider_options
        )
        session = InferenceSession(
            model_cache_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )

        return cls(
            config=config,
            session=session,
            use_io_binding=use_io_binding,
            generation_config=generation_config,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _export(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        # inference options
        use_cache: bool = True,
        **kwargs,
    ) -> "ORTModelForCausalLM":
        # this is garanteed to work since we it uses a mapping from model classes to task names
        # instead of relying on the hub metadata or the model configuration
        task = TasksManager._infer_task_from_model_or_model_class(model_class=cls.auto_model_class)
        if use_cache:
            task += "-with-past"

        if kwargs.get("task", None) is not None:
            raise ValueError(
                f"The `task` argument is not needed when exporting a model with `{cls.__name__}`. "
                f"The `task` is automatically inferred from the class as `{task}`."
            )

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=False,
            legacy=False,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_cache=use_cache,
            model_save_dir=save_dir,
            **kwargs,
        )

    def _save_config(self, save_directory):
        """
        Save the model and generation configs to the specified directory.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the model and generation configs will be saved.
        """
        self.config.save_pretrained(save_directory)
        self.generation_config.save_pretrained(save_directory)
