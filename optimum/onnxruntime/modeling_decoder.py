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
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.file_utils import add_start_docstrings_to_model_forward, default_cache_path
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.onnx import FeaturesManager, export
from transformers.onnx.utils import get_preprocessor

import onnxruntime
from huggingface_hub import hf_hub_download
from ..onnx.configuration import DecoderOnnxConfigWithPast

from .io_binding import TypeHelper
from .modeling_ort import ORTModel
from .utils import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    get_provider_for_device,
    parse_device,
)
from ..utils.normalized_config import NormalizedConfigManager


logger = logging.getLogger(__name__)

DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
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
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
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
    >>> tokenizer.batch_decode(gen_tokens)
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


class ORTDecoder:
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.
    """

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        config: transformers.PretrainedConfig,
        device: torch.device,
        use_io_binding: bool = True,
    ):
        self.session = session
        self.config = config
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(self.config.model_type)(self.config)
        self._device = device
        self.use_io_binding = use_io_binding
        self.session_inputs = {output_key.name: idx for idx, output_key in enumerate(self.session.get_inputs())}
        self.session_outputs = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.session_input_names = list(self.session_inputs.keys())
        self.session_output_names = list(self.session_outputs.keys())
        self.key_value_input_names = [key for key in self.session_input_names if "key_values" in key]
        self.key_value_output_names = [key for key in self.session_output_names if "key_values" in key]
        self.name_to_np_type = TypeHelper.get_io_numpy_type_map(self.session) if self.use_io_binding else None

    def prepare_output_buffer(
        self,
        output_name,
        batch_size=None,
        sequence_length=None,
        past_sequence_length=None,
    ):
        """
        Prepare the buffer of outputs(`logits`/`key_values`/`loss`) with 1D tensors.
        """
        ort_type = TypeHelper.get_output_type(self.session, output_name)
        torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
        if output_name == "logits":
            output_shape = (batch_size, sequence_length, self.normalized_config.vocab_size)
            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self._device).contiguous()
        elif "key_values" in output_name:
            num_attention_heads = self.normalized_config.num_attention_heads
            hidden_size = self.normalized_config.hidden_size
            embed_size_per_head = hidden_size // num_attention_heads

            if past_sequence_length is not None:
                sequence_length += past_sequence_length
            output_shape = (batch_size, num_attention_heads, sequence_length, embed_size_per_head)

            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self._device).contiguous()

        return output_shape, output_buffer

    def prepare_io_binding(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ):
        io_binding = self.session.io_binding()

        # Bind the inputs

        # Bind input ids
        input_ids = input_ids.contiguous()
        io_binding.bind_input(
            "input_ids",
            input_ids.device.type,
            self._device.index,
            self.name_to_np_type["input_ids"],
            tuple(input_ids.shape),
            input_ids.data_ptr(),
        )

        # Bind the attention mask
        attention_mask = attention_mask.contiguous()
        io_binding.bind_input(
            "attention_mask",
            attention_mask.device.type,
            self._device.index,
            self.name_to_np_type["attention_mask"],
            tuple(attention_mask.shape),
            attention_mask.data_ptr(),
        )

        # Bind the past key values
        if past_key_values is not None:
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                past_key_value = past_key_value.contiguous()
                io_binding.bind_input(
                    input_name,
                    past_key_value.device.type,
                    self._device.index,
                    self.name_to_np_type[input_name],
                    tuple(past_key_value.shape),
                    past_key_value.data_ptr(),
                )

        # Bind the outputs

        # Bind the logits
        logits_shape, logits_buffer = self.prepare_output_buffer(
            output_name="logits",
            batch_size=input_ids.size(0),
            sequence_length=input_ids.size(1),
        )
        io_binding.bind_output(
            "logits",
            logits_buffer.device.type,
            self._device.index,
            self.name_to_np_type["logits"],
            logits_shape,
            logits_buffer.data_ptr(),
        )
        output_shapes = {"logits": logits_shape}
        output_buffers = {"logits": logits_buffer}

        # Bind the past keys values
        for key_value_output_name in self.key_value_output_names:
            self_pkv_shape, self_pkv_buffer = self.prepare_output_buffer(
                output_name=key_value_output_name,
                batch_size=input_ids.size(0),
                sequence_length=input_ids.size(1),
                past_sequence_length=past_key_values[0].size(2)
                if past_key_values
                else None,  # sequence length of self-attention key for layer.0
            )
            io_binding.bind_output(
                key_value_output_name,
                self_pkv_buffer.device.type,
                self._device.index,
                self.name_to_np_type[key_value_output_name],
                self_pkv_shape,
                self_pkv_buffer.data_ptr(),
            )
            # set -1 for sequence_length as it could be larger than the real sequence_length for creating buffer
            self_pkv_shape = self_pkv_shape[:2] + (-1,) + self_pkv_shape[3:]
            output_shapes[key_value_output_name] = self_pkv_shape
            output_buffers[key_value_output_name] = self_pkv_buffer

        return io_binding, output_shapes, output_buffers

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = [past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer]

        if self._device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids, attention_mask, past_key_values
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer(2)
            past_key_values = tuple()
            for name in self.key_value_output_names:
                past_key_values += (output_buffers[name].view(output_shapes[name]),)

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (self-attention key and value per decoder layer)
            num_pkv = 2
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = {
                "input_ids": input_ids.cpu().detach().numpy(),
                "attention_mask": attention_mask.cpu().detach().numpy(),
            }

            if past_key_values is not None:
                # Add the past_key_values to the decoder inputs
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                    onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

            # Run inference
            outputs = self.session.run(None, onnx_inputs)

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 for the self-attention)
            past_key_values = tuple(
                torch.from_numpy(outputs[self.session_outputs[key]]).to(self._device)
                for key in self.key_value_output_names
            )

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # per decoder layer
            num_pkv = 2
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))
            logits = torch.from_numpy(outputs[self.session_outputs["logits"]]).to(self._device)

        return CausalLMOutputWithCrossAttentions(logits=logits, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTModelDecoder(ORTModel):
    """
    Base class for implementing models with a causal language modeling head using ONNX Runtime inference.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        decoder_session: onnxruntime.InferenceSession,
        decoder_with_past_session: Optional[onnxruntime.InferenceSession] = None,
        use_io_binding: bool = True,
        model_save_dir: str = "",
        last_decoder_model_name: str = ONNX_DECODER_NAME,
        last_decoder_with_past_model_name: str = ONNX_DECODER_WITH_PAST_NAME,
    ):
        """
        Args:
            decoder_session (`onnxruntime.InferenceSession`):
                The ONNX Runtime inference session associated to the decoder.
            config (`transformers.PretrainedConfig`):
                An instance of the configuration associated to the model. Initializing with a config file does
                not load the weights associated with the model, only the configuration.
            decoder_with_past_session (`Optional[onnxruntime.InferenceSession]`, *optional*):
                The ONNX Runtime inference session associated to the decoder with past key values.
            use_io_binding (`bool`, *optional*, defaults to `True`):
                Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            model_save_dir (`str`, *optional*, defaults to `""`):
                The directory under which the model exported to ONNX was saved.
            last_decoder_model_name (`str`, *optional*, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_NAME`):
                The name of the ONNX file containing the decoder part of the model.
            last_decoder_with_past_model_name (`str`, *optional*, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_WITH_PAST_NAME`):
                The name of the ONNX file containing the decoder with past key/values part of the model.
        """
        super().__init__(
            decoder_session,
            config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            latest_model_name=last_decoder_model_name,
        )
        self.decoder_file_name = last_decoder_model_name
        self.decoder_file_with_past_name = last_decoder_with_past_model_name

        self.use_cache = decoder_with_past_session is not None
        self.decoder = ORTDecoder(
            session=self.model, config=self.config, device=self._device, use_io_binding=self.use_io_binding
        )
        self.decoder_with_past = None
        if self.use_cache:
            self.decoder_with_past = ORTDecoder(
                session=decoder_with_past_session,
                config=self.config,
                device=self._device,
                use_io_binding=self.use_io_binding,
            )

    @staticmethod
    def load_model(
        decoder_path: Union[str, Path],
        decoder_with_past_path: Optional[Union[str, Path]] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict] = None,
    ):
        """
        Creates an instance of [`~optimum.onnxruntime.ORTModelDecoder`].
        Three inference sessions will be created for respectively the decoder and decoder with past key values
        models. The default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            decoder_path (`str` or `Path`):
                The path of the decoder ONNX model.
            decoder_with_past_path (`str` or `Path`, *optional*):
                The path of the decoder with past key values ONNX model.
            provider(`str`, *optional*, defaults to `"CPUExecutionProvider"`):
                The ONNX Runtime provider to use for loading the model.
            session_options (`Optional[onnxruntime.SessionOptions]`, *optional*),:
                ONNX Runtime session options to use for loading the model.
            provider_options (`Optional[Dict]`, *optional*):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html.
        """
        available_providers = onnxruntime.get_available_providers()
        if provider not in available_providers:
            raise ValueError(
                f"Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}."
            )

        providers = [provider]
        if provider == "TensorrtExecutionProvider":
            # follow advice in https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#python
            providers.append("CUDAExecutionProvider")

        decoder_session = onnxruntime.InferenceSession(
            str(decoder_path),
            providers=providers,
            sess_options=session_options,
            provider_options=None if provider_options is None else [provider_options],
        )
        decoder_with_past_session = None
        # If a decoder_with_past_path is provided, an inference session for the decoder with past key/values as inputs
        # will be enabled
        if decoder_with_past_path is not None:
            decoder_with_past_session = onnxruntime.InferenceSession(
                str(decoder_with_past_path),
                providers=providers,
                sess_options=session_options,
                provider_options=None if provider_options is None else [provider_options],
            )
        return decoder_session, decoder_with_past_session

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        decoder_file_name: str = ONNX_DECODER_NAME,
        decoder_with_past_file_name: str = ONNX_DECODER_WITH_PAST_NAME,
    ):
        """
        Saves the model decoder and decoder with past key values as well as its configuration file to a
        directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_causal.ORTModelDecoder.from_pretrained`] class method.

        Args:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            decoder_file_name (`str`, *optional*, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_NAME`):
                The decoder model file name. Overwrites the default file name and allows one to save the decoder model
                with a different name.
            decoder_with_past_file_name (`str`, *optional*, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_WITH_PAST_NAME`):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
        """
        src_file_names = [self.decoder_file_name]
        dst_file_names = [decoder_file_name]
        if self.use_cache:
            src_file_names.append(self.decoder_file_with_past_name)
            dst_file_names.append(decoder_with_past_file_name)

        for src_file_name, dst_file_name in zip(src_file_names, dst_file_names):
            src_path = self.model_save_dir.joinpath(src_file_name)
            dst_path = Path(save_directory).joinpath(dst_file_name)
            shutil.copyfile(src_path, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        decoder_file_name: str = ONNX_DECODER_NAME,
        decoder_with_past_file_name: str = ONNX_DECODER_WITH_PAST_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        use_io_binding: bool = True,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ):
        file_names = {}
        # Load model from a local directory
        if os.path.isdir(os.path.join(model_id, subfolder)):
            decoder_with_past_path = (
                os.path.join(model_id, subfolder, decoder_with_past_file_name) if use_cache else None
            )
            model = cls.load_model(
                decoder_path=os.path.join(model_id, subfolder, decoder_file_name),
                decoder_with_past_path=decoder_with_past_path,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )
            model_save_dir = Path(model_id).joinpath(subfolder)
            file_names["last_decoder_model_name"] = decoder_file_name
            file_names["last_decoder_with_past_model_name"] = decoder_with_past_file_name
        # Load model from hub
        else:
            default_file_names = [ONNX_DECODER_NAME]
            model_file_names = [decoder_file_name]
            if use_cache:
                default_file_names.append(ONNX_DECODER_WITH_PAST_NAME)
                model_file_names.append(decoder_with_past_file_name)
            # Download the decoder and decoder_with_past forming the model
            for file_name, default_file_name in zip(model_file_names, default_file_names):
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=file_name,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                file_names[f"last_{default_file_name.split('.')[0]}_name"] = Path(model_cache_path).name
            model_save_dir = Path(model_cache_path).parent

            last_decoder_with_past_name = file_names.get("last_decoder_with_past_model_name", None)
            if last_decoder_with_past_name is not None:
                last_decoder_with_past_name = model_save_dir.joinpath(last_decoder_with_past_name)
            model = cls.load_model(
                decoder_path=model_save_dir.joinpath(file_names["last_decoder_model_name"]),
                decoder_with_past_path=last_decoder_with_past_name,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )

        return cls(
            config,
            *model,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            last_decoder_model_name=file_names["last_decoder_model_name"],
            last_decoder_with_past_model_name=file_names.get("last_decoder_with_past_model_name", None),
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        subfolder: Optional[str] = "",
        save_dir: Union[str, Path] = default_cache_path,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        # Create local save dir in cache dir
        save_dir = Path(save_dir).joinpath(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = get_preprocessor(model_id)
        framework = FeaturesManager.determine_framework(os.path.join(model_id, subfolder))
        model_class = FeaturesManager.get_model_class_for_feature(cls.export_feature, framework)
        model = model_class.from_pretrained(model_id, subfolder=subfolder, config=config, cache_dir=cache_dir)

        # Export the decoder without the past key values
        onnx_config = DecoderOnnxConfigWithPast(model.config, task=cls.export_feature, use_past=False)
        onnx_opset = onnx_config.default_onnx_opset
        export(
            preprocessor=preprocessor,
            model=model,
            config=onnx_config,
            opset=onnx_opset,
            output=save_dir.joinpath(ONNX_DECODER_NAME),
        )

        # Export the decoder with the past key values
        if use_cache:
            onnx_config_with_past = DecoderOnnxConfigWithPast(model.config, task=cls.export_feature, use_past=True)
            export(
                preprocessor=preprocessor,
                model=model,
                config=onnx_config_with_past,
                opset=onnx_opset,
                output=save_dir.joinpath(ONNX_DECODER_WITH_PAST_NAME),
            )

        return cls._from_pretrained(save_dir, config=config, use_cache=use_cache, **kwargs)

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`Union[torch.device, str, int]`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModel`: the model placed on the requested device.
        """
        device, provider_options = parse_device(device)

        provider = get_provider_for_device(device)
        self.device = device
        self.decoder._device = device
        self.decoder.session.set_providers([provider], provider_options=[provider_options])
        if self.decoder_with_past is not None:
            self.decoder_with_past._device = device
            self.decoder_with_past.session.set_providers([provider], provider_options=[provider_options])
        self.providers = self.decoder.session.get_providers()

        return self


class ORTModelForCausalLM(ORTModelDecoder, GenerationMixin):
    """
    ONNX model with a causal language modeling head for ONNX Runtime inference.
    """

    # Used to export the model to ONNX
    export_feature = "causal-lm"
    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"

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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:

        if past_key_values is None or self.decoder_with_past is None:
            outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.decoder_with_past(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

        return CausalLMOutputWithCrossAttentions(logits=outputs.logits, past_key_values=outputs.past_key_values)

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly

        attention_mask = kwargs.get("attention_mask", None)  # input_ids.new_ones(input_ids.shape)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
