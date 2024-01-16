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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import torch
from onnx.tools import update_model_dims
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast

import onnxruntime

from ..exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from ..onnx.utils import check_model_uses_external_data
from ..utils import NormalizedConfigManager, check_if_transformers_greater
from ..utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ..utils.save_utils import maybe_save_preprocessors
from .constants import DECODER_MERGED_ONNX_FILE_PATTERN, DECODER_ONNX_FILE_PATTERN, DECODER_WITH_PAST_ONNX_FILE_PATTERN
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .models.bloom import bloom_convert_to_bloom_cache, bloom_convert_to_standard_cache
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if check_if_transformers_greater("4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin


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
    ONNX model with a causal language modeling head for ONNX Runtime inference. This class officially supports bloom, codegen, falcon, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gptj, llama.
    """

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"

    def __init__(
        self,
        model: onnxruntime.InferenceSession,
        config: "PretrainedConfig",
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if use_io_binding is None:
            use_io_binding = model.get_providers()[0] in ["CPUExecutionProvider", "CUDAExecutionProvider"]

        super().__init__(model, config, use_io_binding, model_save_dir, preprocessors, **kwargs)

        self.num_pkv = 2
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.key_value_input_names = [key for key in self.inputs_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config
        self.onnx_paths = [self.model_path]
        self.use_merged = "use_cache_branch" in self.inputs_names
        self.model_type = self.config.model_type

        self.use_fp16 = False
        for inp in model.get_inputs():
            if inp.name == "past_key_values" and inp.type == "tensor(float16)":
                self.use_fp16 = True
                break

        # Reference: https://github.com/huggingface/optimum/pull/1381
        model_type = config.model_type.replace("_", "-")
        if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS and "position_ids" not in self.inputs_names:
            logger.warning(
                f"ORTModelForCausalLM loaded a legacy ONNX model with no position_ids input, although this input is required for batched generation for the architecture {model_type}. "
                "We strongly encourage to re-export the model with optimum>=1.14 for position_ids and batched inference support."
            )

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )

        if use_io_binding and not use_cache:
            raise ValueError(
                "The parameters combination use_cache=False, use_io_binding=True is not supported. "
                "Please either pass use_cache=True, use_io_binding=True (default), or use_cache=False, use_io_binding=False."
            )

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
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache_branch: bool = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # adding use_cache_branch in the signature here is just a hack for IO Binding
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        inputs = {}
        known_output_shapes = {}
        use_cache_branch = None
        loss = None
        if self.use_cache:
            if past_key_values is not None:
                # Flatten the past_key_values (gpt_bigcode has fused key/value cache, so no need to flatten it)
                if self.model_type != "gpt_bigcode":
                    past_key_values = tuple(
                        past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                    )

            # Create dummy past_key_values for decoder first generation step if none given
            use_cache_branch, past_key_values, known_output_shapes = self.prepare_past_key_values(
                input_ids, past_key_values, use_torch
            )

        if self.use_io_binding:
            # TODO: fix transformers generate to have contiguous input_ids here already
            # For an unknown reason, calling `contiguous()` here is necessary to not have errors
            # on CPU EP with batch size > 1, despite it being also called in _prepare_io_binding.
            # I suspect the reason is the contiguous python list that messes something up?
            model_inputs = [input_ids.contiguous()]

            if "attention_mask" in self.inputs_names:
                model_inputs.append(attention_mask)

            if "position_ids" in self.inputs_names:
                if position_ids is None:
                    raise ValueError("position_ids was not passed but is a required input for this ONNX model.")
                model_inputs.append(position_ids.contiguous())

            if past_key_values is not None:
                model_inputs += past_key_values

            if use_cache_branch is not None:
                model_inputs.append(use_cache_branch)

            if "labels" in self.inputs_names:
                model_inputs.append(labels)
                known_output_shapes.update({"loss": []})

            io_binding, output_shapes, output_buffers = self._prepare_io_binding(
                self.model,
                *model_inputs,
                known_output_shapes=known_output_shapes,
                ordered_input_names=self._ordered_input_names,
            )

            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer(2)
                past_key_values = ()
                for name in self.key_value_output_names:
                    past_key_values += (output_buffers[name].view(output_shapes[name]),)

            logits = output_buffers["logits"].view(output_shapes["logits"])

            if "loss" in self.output_names:
                loss = output_buffers["loss"].view(output_shapes["loss"])
        else:
            inputs["input_ids"] = input_ids.cpu().detach().numpy() if use_torch else input_ids

            if "attention_mask" in self.inputs_names:
                inputs["attention_mask"] = attention_mask.cpu().detach().numpy() if use_torch else attention_mask

            if "labels" in self.inputs_names:
                inputs["labels"] = labels.cpu().detach().numpy() if use_torch else labels

            if "position_ids" in self.inputs_names:
                if position_ids is None:
                    raise ValueError("position_ids was not passed but is a required input for this ONNX model.")
                inputs["position_ids"] = position_ids.cpu().detach().numpy() if use_torch else position_ids

            # Add the past_key_values to the decoder inputs
            if past_key_values is not None:
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                    inputs[input_name] = past_key_value.cpu().detach().numpy() if use_torch else past_key_value

            if use_cache_branch is not None:
                inputs["use_cache_branch"] = use_cache_branch.cpu().detach().numpy() if use_torch else use_cache_branch

            outputs = self.model.run(None, inputs)

            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 for the self-attention)
                past_key_values = tuple(
                    torch.from_numpy(outputs[self.output_names[key]]).to(self.device)
                    for key in self.key_value_output_names
                )

            logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self.device)
            if "loss" in self.output_names:
                loss = torch.from_numpy(outputs[self.output_names["loss"]]).to(self.device)

        if self.use_cache and self.model_type != "gpt_bigcode":
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # per decoder layer
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)

    def prepare_past_key_values(
        self,
        input_ids: Union[None, torch.LongTensor, np.ndarray],
        past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]],
        use_torch: bool,
    ):
        sequence_length = input_ids.shape[1]

        constructor = torch if use_torch else np
        if self.use_merged:
            # Uses without/with branch of a merged decoder depending on whether real past key values are passed
            use_cache_branch = constructor.full((1,), past_key_values is not None)
        else:
            # Uses separate decoders
            use_cache_branch = None

        if use_torch and use_cache_branch is not None:
            use_cache_branch = use_cache_branch.to(self.device)

        pkv_output_shape = {}
        # Generate dummy past for the first forward if uses a merged decoder
        if past_key_values is None:
            batch_size = input_ids.shape[0]
            if self.model_type in {"mistral", "llama"}:
                num_attention_heads = self.normalized_config.num_key_value_heads
            else:
                num_attention_heads = self.normalized_config.num_attention_heads
            embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads

            dtype = constructor.float16 if self.use_fp16 else constructor.float32

            # TODO: find a way to better handle this controlflow, this is EXTREMELY UGLY.
            # "1" is the dummy sequence length
            if self.model_type == "bloom":
                shape_value = (batch_size * num_attention_heads, 0, embed_size_per_head)
                shape_key = (batch_size * num_attention_heads, embed_size_per_head, 0)
                key = constructor.zeros(shape_key, dtype=dtype)
                value = constructor.zeros(shape_value, dtype=dtype)

                if use_torch:
                    key = key.to(self.device)
                    value = value.to(self.device)

                past_key_values = tuple(
                    key_or_value for _ in range(len(self.key_value_input_names) // 2) for key_or_value in [key, value]
                )

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    index = 1 if "value" in name else 2

                    shape[index] += sequence_length
                    pkv_output_shape[name] = shape
            elif self.model_type == "gpt_bigcode":
                # GPT BigCode uses muti-query attention, and has the specificity of putting both key and value in the same cache tensor.
                shape_key_and_value = (batch_size, 0, embed_size_per_head * 2)
                key_and_value = constructor.zeros(shape_key_and_value, dtype=dtype)

                if use_torch:
                    key_and_value = key_and_value.to(self.device)

                past_key_values = tuple(key_and_value for _ in range(len(self.key_value_input_names)))

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    shape[1] += sequence_length
                    pkv_output_shape[name] = shape
            else:
                num_key_value_heads = self.num_key_value_heads if self.model_type == "falcon" else num_attention_heads

                shape = (batch_size, num_key_value_heads, 0, embed_size_per_head)
                key_or_value = constructor.zeros(shape, dtype=dtype)

                if use_torch:
                    key_or_value = key_or_value.to(self.device)

                past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    shape[2] += sequence_length
                    pkv_output_shape[name] = shape

        return use_cache_branch, past_key_values, pkv_output_shape

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        use_cache: bool = True,
        local_files_only: bool = False,
        use_merged: Optional[bool] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ) -> "ORTModelForCausalLM":
        model_path = Path(model_id)

        # We do not implement the logic for use_cache=False, use_merged=True
        if use_cache is False:
            if use_merged is True:
                raise ValueError(
                    "The parameters combination use_cache=False, use_merged=True is not supported."
                    " To use a merged decoder, past key values must be used."
                )
            use_merged = False

        decoder_name = "decoder_file_name" if use_cache else "decoder_with_past_file_name"
        decoder_file_name = kwargs.pop(decoder_name, None)

        if decoder_file_name is not None:
            logger.warning(f"The `{decoder_name}` argument is deprecated, please use `file_name` instead.")
            file_name = file_name or decoder_file_name

        if file_name is None:
            decoder_path = None
            # We use `is not False` here to include two cases: use_merged = None (in which case we auto-detect it),
            # and use_merged = True (explicitely specified by the user)
            if use_merged is not False:
                try:
                    decoder_path = ORTModelForCausalLM.infer_onnx_filename(
                        model_id,
                        [DECODER_MERGED_ONNX_FILE_PATTERN],
                        argument_name=None,
                        subfolder=subfolder,
                        use_auth_token=use_auth_token,
                        revision=revision,
                    )
                    use_merged = True
                    file_name = decoder_path.name
                except FileNotFoundError as e:
                    if use_merged is True:
                        raise FileNotFoundError(
                            "The parameter `use_merged=True` was passed to ORTModelForCausalLM.from_pretrained()"
                            " but no ONNX file for a merged decoder could be found in"
                            f" {str(Path(model_id, subfolder))}, with the error: {e}"
                        )
                    use_merged = False

            if use_merged is False:
                pattern = DECODER_WITH_PAST_ONNX_FILE_PATTERN if use_cache else DECODER_ONNX_FILE_PATTERN
                # exclude decoder file for first iteration
                decoder_path = ORTModelForCausalLM.infer_onnx_filename(
                    model_id,
                    [r"^((?!decoder).)*.onnx", pattern],
                    argument_name=None,
                    subfolder=subfolder,
                    use_auth_token=use_auth_token,
                    revision=revision,
                )
                file_name = decoder_path.name

            if file_name == ONNX_DECODER_WITH_PAST_NAME and config.model_type in MODEL_TO_PATCH_FOR_PAST:
                raise ValueError(
                    f"ONNX Runtime inference using {ONNX_DECODER_WITH_PAST_NAME} has been deprecated for {config.model_type} architecture. Please re-export your model with optimum>=1.14.0 or set use_cache=False. For details about the deprecation, please refer to https://github.com/huggingface/optimum/releases/tag/v1.14.0."
                )

            regular_file_names = []
            for name in [ONNX_WEIGHTS_NAME, ONNX_DECODER_WITH_PAST_NAME if use_cache else ONNX_DECODER_NAME]:
                regular_file_names += ORTModelForCausalLM._generate_regular_names_for_filename(name)

            if file_name not in regular_file_names:
                logger.warning(
                    f"The ONNX file {file_name} is not a regular name used in optimum.onnxruntime that are {regular_file_names}, the "
                    f"{cls.__name__} might not behave as expected."
                )

        model_cache_path, preprocessors = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        new_model_save_dir = model_cache_path.parent

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance, in which case we want to keep it
        # instead of the path only.
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        # Since v1.7.0 decoder with past models have fixed sequence length of 1
        # To keep these models compatible we set this dimension to dynamic
        onnx_model = onnx.load(str(model_cache_path), load_external_data=False)
        model_uses_external_data = check_model_uses_external_data(onnx_model)

        if model_uses_external_data:
            onnx_model = onnx.load(str(model_cache_path), load_external_data=True)

        input_dims = {
            node.name: [dim.dim_value or dim.dim_param for dim in node.type.tensor_type.shape.dim]
            for node in onnx_model.graph.input
        }
        if input_dims["input_ids"][1] == 1:
            input_dims["input_ids"][1] = "sequence_length"
            output_dims = {
                node.name: [dim.dim_value or dim.dim_param for dim in node.type.tensor_type.shape.dim]
                for node in onnx_model.graph.output
            }
            output_dims["logits"][1] = "sequence_length"
            onnx_model = update_model_dims.update_inputs_outputs_dims(onnx_model, input_dims, output_dims)

            onnx.save(
                onnx_model,
                str(model_cache_path),
                save_as_external_data=model_uses_external_data,
                all_tensors_to_one_file=True,
                location=model_cache_path.name + "_data",
                size_threshold=0,
            )
        del onnx_model

        model = ORTModel.load_model(
            model_cache_path,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )

        if config.model_type == "bloom":
            init_cls = ORTBloomForCausalLM
        elif config.model_type == "falcon":
            init_cls = ORTFalconForCausalLM
        elif config.model_type == "mpt":
            init_cls = ORTMPTForCausalLM
        elif config.model_type == "opt":
            init_cls = ORTOPTForCausalLM
        elif config.model_type == "gpt_bigcode":
            init_cls = ORTGPTBigCodeForCausalLM
        else:
            init_cls = ORTModelForCausalLM

        return init_cls(
            model=model,
            config=config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
            use_cache=use_cache,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        use_cache: bool = True,
        use_merged: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTModelForCausalLM":
        file_name = ONNX_WEIGHTS_NAME

        if use_merged:
            logger.warning("The `use_merged` argument is deprecated when the model is exported, and not used anymore.")
            use_merged = False

        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

            if use_cache:
                task += "-with-past"

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
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        config.save_pretrained(save_dir_path)
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_cache=use_cache,
            use_merged=use_merged,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            model_save_dir=save_dir,
            file_name=file_name,
        )

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True


class ORTGPTBigCodeForCausalLM(ORTModelForCausalLM):
    # Adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Omit tokens covered by past_key_values
        if past_key_values:
            if self.config.multi_query:
                past_length = past_key_values[0].shape[1]
            else:
                past_length = past_key_values[0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class ORTBloomForCausalLM(ORTModelForCausalLM):
    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        # only last token for input_ids if past is not None
        if past_key_values:
            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = bloom_convert_to_bloom_cache(past_key_values)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }

    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM._reorder_cache
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        standardized_past = bloom_convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return bloom_convert_to_bloom_cache(reordered_past)


class ORTOPTForCausalLM(ORTModelForCausalLM):
    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }


class ORTMPTForCausalLM(ORTModelForCausalLM):
    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }


class ORTFalconForCausalLM(ORTModelForCausalLM):
    def __init__(
        self,
        model: onnxruntime.InferenceSession,
        config: "PretrainedConfig",
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            config=config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
            generation_config=generation_config,
            use_cache=use_cache,
            **kwargs,
        )
        self.num_key_value_heads = (
            config.num_kv_heads if (config.new_decoder_architecture or not config.multi_query) else 1
        )
        self.use_alibi = config.alibi

    # Copied from transformers.models.falcon.modeling_falcon.FalconForCausalLM._reorder_cache
    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        return reordered_past

    # Adapted from transformers.models.falcon.modeling_falcon.FalconForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # Note: versions of Falcon with alibi do not use position_ids. It is used with RoPE.
        if not self.use_alibi and attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
