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

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoTokenizer, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, default_cache_path
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.onnx import FeaturesManager, export

import onnx
import onnxruntime
from huggingface_hub import HfApi, hf_hub_download

from ..exporters.onnx.model_configs import SpeechSeq2SeqDecoderOnnxConfig, SpeechSeq2SeqEncoderOnnxConfig
from ..onnx.configuration import DecoderOnnxConfig, EncoderOnnxConfig
from ..onnx.modeling_seq2seq import _DecoderWithLMhead
from .io_binding import TypeHelper
from .modeling_ort import ORTModel
from .utils import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    ORTConfigManager,
    get_device_for_provider,
    get_provider_for_device,
    parse_device,
)


logger = logging.getLogger(__name__)

ONNX_INPUTS_DOCSTRING = r"""
    Arguments:
        encoder_session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
        decoder_session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the decoder.
        decoder_with_past_session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the decoder with past key values.
        config (`transformers.PretrainedConfig`):
            [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is an instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
        encoder_file_name(`str`, *optional*):
            The encoder model file name. Overwrites the default file name and allows one to save the encoder model with
            a different name.
        decoder_file_name(`str`, *optional*):
            The decoder model file name. Overwrites the default file name and allows one to save the decoder model with
            a different name.
        decoder_with_past_file_name(`str`, *optional*):
            The decoder with past key values model file name overwriting the default file name, allowing to save
            the decoder model with a different name.
        use_io_binding (`bool`, *optional*):
            Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to `True`
            if the device is CUDA, otherwise defaults to `False`.
"""

SEQ2SEQ_ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
"""

WHISPER_ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_features (`torch.FloatTensor`):
            Mel features extracted from the raw speech waveform. `(batch_size, feature_size, encoder_sequence_length)`.
"""


DECODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""


SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Arguments:
        input_features (`torch.FloatTensor`):
            Mel features extracted from the raw speech waveform.
            `(batch_size, feature_size, encoder_sequence_length)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

_TOKENIZER_FOR_DOC = "AutoTokenizer"
_PROCESSOR_FOR_DOC = "AutoProcessor"

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
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> speech_recognition = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> pred = speech_recognition(ds[0]["audio"]["array"])
    ```
"""


@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.
    """,
    ONNX_INPUTS_DOCSTRING,
)
class ORTModelForConditionalGeneration(ORTModel):
    # Used in from_transformers to export model to onnxORTEncoder
    base_model_prefix = "onnx_model"

    def __init__(
        self,
        encoder_session: onnxruntime.InferenceSession = None,
        decoder_session: onnxruntime.InferenceSession = None,
        decoder_with_past_session: onnxruntime.InferenceSession = None,
        config: transformers.PretrainedConfig = None,
        use_io_binding: bool = True,
        **kwargs
    ):
        self.config = config
        self.use_io_binding = use_io_binding
        self.model_save_dir = kwargs.get("model_save_dir", None)

        self.providers = encoder_session.get_providers()
        self._device = get_device_for_provider(encoder_session.get_providers()[0])

        if "TensorrtExecutionProvider" in self.providers and self.use_io_binding:
            logger.warning(
                "There is no need to do IO binding for TensorrtExecutionProvider, `use_io_binding` will be set to False."
            )
            self.use_io_binding = False

        self.encoder = self._initialize_encoder(
            session=encoder_session, config=self.config, device=self._device, use_io_binding=self.use_io_binding
        )
        self.decoder = ORTDecoder(
            session=decoder_session, config=self.config, device=self._device, use_io_binding=self.use_io_binding
        )

        self.use_cache = decoder_with_past_session is not None

        # If a decoder_with_past_path is provided, an inference session for the decoder with past key/values as inputs
        # will be enabled
        self.decoder_with_past = (
            ORTDecoder(
                session=decoder_with_past_session,
                config=self.config,
                device=self._device,
                use_io_binding=self.use_io_binding,
            )
            if self.use_cache
            else None
        )

        self.encoder_file_name = kwargs.get("last_encoder_model_name", ONNX_ENCODER_NAME)
        self.decoder_file_name = kwargs.get("last_decoder_model_name", ONNX_DECODER_NAME)
        self.decoder_file_with_past_name = kwargs.get("last_decoder_with_past_model_name", ONNX_DECODER_WITH_PAST_NAME)
        # registers the ORTModelForXXX classes into the transformers AutoModel classes
        # to avoid warnings when create a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    @staticmethod
    def load_model(
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        decoder_with_past_path: Union[str, Path] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict] = None,
        **kwargs
    ):
        """
        Creates an instance of [`~optimum.onnxruntime.modeling_seq2seq.ORTModelForConditionalGeneration`].
        Three inference sessions will be created for respectively the encoder, decoder and decoder with past key values
        models. The default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Arguments:
            encoder_path (`str` or `Path`):
                The path of the encoder ONNX model.
            decoder_path (`str` or `Path`):
                The path of the decoder ONNX model.
            decoder_with_past_path (`str` or `Path`, *optional*):
                The path of the decoder with past key values ONNX model.
            provider (`str`, *optional*):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers. Defaults to `CPUExecutionProvider`.
            session_options (`onnxruntime.SessionOptions`, *optional*),:
                ONNX Runtime session options to use for loading the model. Defaults to `None`.
            provider_options (`Dict`, **optional**):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html . Defaults to `None`.
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

        encoder_session = onnxruntime.InferenceSession(
            str(encoder_path),
            providers=providers,
            sess_options=session_options,
            provider_options=None if provider_options is None else [provider_options],
        )
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

        return encoder_session, decoder_session, decoder_with_past_session

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        **kwargs
    ):
        """
        Saves the model encoder, decoder and decoder with past key values as well as its configuration file to a
        directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_seq2seq.ORTModelForSeq2SeqLM.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name and allows one to save the encoder model
                with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name and allows one to save the decoder model
                with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
        """
        src_file_names = [self.encoder_file_name, self.decoder_file_name]
        dst_file_names = [encoder_file_name or ONNX_ENCODER_NAME, decoder_file_name or ONNX_DECODER_NAME]
        if self.use_cache:
            src_file_names.append(self.decoder_file_with_past_name)
            dst_file_names.append(decoder_with_past_file_name or ONNX_DECODER_WITH_PAST_NAME)

        for src_file_name, dst_file_name in zip(src_file_names, dst_file_names):
            src_path = self.model_save_dir.joinpath(src_file_name)
            dst_path = Path(save_directory).joinpath(dst_file_name)
            shutil.copyfile(src_path, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        subfolder: Optional[str] = "",
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.
        Implements: https://github.com/huggingface/huggingface_hub/blob/e67de48368bc1843e40afc1cc9d236402b9609ee/src/huggingface_hub/hub_mixin.py#L73

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name and allows one to save the encoder model
                with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name and allows one to save the decoder model
                with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not to use the pre-computed key/values hidden-states in order to speed up sequential decoding.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        use_cache = kwargs.pop("use_cache", True)
        local_files_only = kwargs.pop("local_files_only", False)
        config = kwargs.pop("config", {})
        encoder_file_name = encoder_file_name or ONNX_ENCODER_NAME
        decoder_file_name = decoder_file_name or ONNX_DECODER_NAME
        decoder_with_past_file_name = decoder_with_past_file_name or ONNX_DECODER_WITH_PAST_NAME

        # Load model from a local directory
        if os.path.isdir(os.path.join(model_id, subfolder)):
            decoder_with_past_path = (
                os.path.join(model_id, subfolder, decoder_with_past_file_name) if use_cache else None
            )
            model = cls.load_model(
                encoder_path=os.path.join(model_id, subfolder, encoder_file_name),
                decoder_path=os.path.join(model_id, subfolder, decoder_file_name),
                decoder_with_past_path=decoder_with_past_path,
                **kwargs,
            )
            kwargs["model_save_dir"] = Path(model_id).joinpath(subfolder)
            kwargs["last_encoder_name"] = encoder_file_name
            kwargs["last_decoder_name"] = decoder_file_name
            kwargs["last_decoder_with_past_name"] = decoder_with_past_file_name
        # Load model from hub
        else:
            default_file_names = [ONNX_ENCODER_NAME, ONNX_DECODER_NAME]
            model_file_names = [encoder_file_name, decoder_file_name]
            if use_cache:
                default_file_names.append(ONNX_DECODER_WITH_PAST_NAME)
                model_file_names.append(decoder_with_past_file_name)
            # Download the encoder, decoder and decoder_with_past forming the model
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
                kwargs[f"last_{default_file_name.split('.')[0]}_name"] = Path(model_cache_path).name
            kwargs["model_save_dir"] = Path(model_cache_path).parent

            last_decoder_with_past_name = kwargs.get("last_decoder_with_past_model_name", None)
            if last_decoder_with_past_name is not None:
                last_decoder_with_past_name = kwargs["model_save_dir"].joinpath(last_decoder_with_past_name)
            model = cls.load_model(
                encoder_path=kwargs["model_save_dir"].joinpath(kwargs["last_encoder_model_name"]),
                decoder_path=kwargs["model_save_dir"].joinpath(kwargs["last_decoder_model_name"]),
                decoder_with_past_path=last_decoder_with_past_name,
                **kwargs,
            )

        return cls(*model, config=config, **kwargs)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        subfolder: Optional[str] = "",
        save_dir: Union[str, Path] = default_cache_path,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Exports through the ONNX format a vanilla Transformers model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
            save_dir (`str` or `Path`):
                The directory where the ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache dir for transformers.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not to use the pre-computed key/values hidden-states in order to speed up sequential decoding.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization.
        """
        # Create local save dir in cache dir
        save_dir = Path(save_dir).joinpath(model_id, subfolder)
        save_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model_save_dir"] = save_dir
        config = kwargs.get("config", {})
        use_cache = kwargs.get("use_cache", True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        framework = FeaturesManager.determine_framework(os.path.join(model_id, subfolder))
        model_class = FeaturesManager.get_model_class_for_feature(cls.export_feature, framework)
        model = model_class.from_pretrained(model_id, subfolder=subfolder, config=config, cache_dir=cache_dir)

        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=cls.export_feature)
        onnx_config = model_onnx_config(model.config)
        onnx_opset = onnx_config.default_onnx_opset

        # Extract the encoder and decoder for ONNX export
        encoder = model.get_encoder()
        decoder = model.get_decoder()

        # Concatenate the decoder with the language model head for ONNX export
        decoder_with_lm_head = _DecoderWithLMhead(model)

        # Get the encoder and decoder ONNX configs
        onnx_config_encoder = cls.get_encoder_onnx_config(encoder.config)
        onnx_config_decoder = cls.get_decoder_onnx_config(decoder.config, cls.export_feature, use_past=False)
        if use_cache:
            onnx_config_decoder_with_past = cls.get_decoder_onnx_config(
                decoder.config, cls.export_feature, use_past=True
            )

        if config.model_type == "whisper":
            from ..exporters.onnx.convert import export as export_optimum

            # Export the encoder
            export_optimum(
                encoder,
                onnx_config_encoder,
                onnx_opset,
                save_dir.joinpath(ONNX_ENCODER_NAME),
            )

            # Export the decoder without the past key values
            export_optimum(
                model,
                onnx_config_decoder,
                onnx_opset,
                save_dir.joinpath(ONNX_DECODER_NAME),
            )

            # Export the decoder with the past key values
            if use_cache:
                export_optimum(
                    model,
                    onnx_config_decoder_with_past,
                    onnx_opset,
                    save_dir.joinpath(ONNX_DECODER_WITH_PAST_NAME),
                )
        else:
            # Export the encoder
            export(
                preprocessor=tokenizer,
                model=encoder,
                config=onnx_config_encoder,
                opset=onnx_opset,
                output=save_dir.joinpath(ONNX_ENCODER_NAME),
            )

            # Export the decoder without the past key values
            export(
                preprocessor=tokenizer,
                model=decoder_with_lm_head,
                config=onnx_config_decoder,
                opset=onnx_opset,
                output=save_dir.joinpath(ONNX_DECODER_NAME),
            )

            # Export the decoder with the past key values
            if use_cache:
                export(
                    preprocessor=tokenizer,
                    model=decoder_with_lm_head,
                    config=onnx_config_decoder_with_past,
                    opset=onnx_opset,
                    output=save_dir.joinpath(ONNX_DECODER_WITH_PAST_NAME),
                )

        kwargs["config"] = model.config
        return cls._from_pretrained(save_dir, **kwargs)

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Arguments:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModel`: the model placed on the requested device.
        """
        device, provider_options = parse_device(device)

        provider = get_provider_for_device(device)
        self.device = device
        self.encoder._device = device
        self.encoder.session.set_providers([provider], provider_options=[provider_options])
        self.decoder._device = device
        self.decoder.session.set_providers([provider], provider_options=[provider_options])
        if self.decoder_with_past is not None:
            self.decoder_with_past._device = device
            self.decoder_with_past.session.set_providers([provider], provider_options=[provider_options])
        self.providers = self.encoder.session.get_providers()

        return self


class ORTEncoder:
    """
    Encoder model for ONNX Runtime inference.

    Arguments:
        session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        config: transformers.PretrainedConfig,
        device: torch.device,
        use_io_binding: bool = True,
        main_input_name: str = "input_ids",
        **kwargs
    ):
        self.session = session
        self.config = config
        self._device = device
        self.use_io_binding = use_io_binding
        self.main_input_name = main_input_name
        self.normalized_config = ORTConfigManager.get_normalized_config_class(self.config.model_type)(self.config)
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.name_to_np_type = TypeHelper.get_io_numpy_type_map(self.session) if self.use_io_binding else None

    def prepare_output_buffer(self, batch_size, sequence_length):
        """Prepare the buffer of output(`last_hidden_state`) with a 1D tensor on shape: (batch_size, sequence_length, hidden_size)."""
        ort_type = TypeHelper.get_output_type(self.session, "last_hidden_state")
        torch_type = TypeHelper.ort_type_to_torch_type(ort_type)

        hidden_size = self.normalized_config.hidden_size
        output_shape = (batch_size, sequence_length, hidden_size)
        output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self._device).contiguous()

        return output_shape, output_buffer

    def prepare_io_binding(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        io_binding = self.session.io_binding()

        # bind input ids
        input_ids = input_ids.contiguous()
        io_binding.bind_input(
            "input_ids",
            input_ids.device.type,
            self._device.index,
            self.name_to_np_type["input_ids"],
            tuple(input_ids.shape),
            input_ids.data_ptr(),
        )
        if "attention_mask" in self.input_names:
            # bind attention mask
            attention_mask = attention_mask.contiguous()
            io_binding.bind_input(
                "attention_mask",
                attention_mask.device.type,
                self._device.index,
                self.name_to_np_type["attention_mask"],
                tuple(attention_mask.shape),
                attention_mask.data_ptr(),
            )

        # bind last_hidden_state
        output_shape, output_buffer = self.prepare_output_buffer(
            batch_size=input_ids.size(0),
            sequence_length=input_ids.size(1),
        )
        io_binding.bind_output(
            "last_hidden_state",
            output_buffer.device.type,
            self._device.index,
            self.name_to_np_type["last_hidden_state"],
            output_shape,
            output_buffer.data_ptr(),
        )
        output_shapes = {"last_hidden_state": output_shape}
        output_buffers = {"last_hidden_state": output_buffer}

        return io_binding, output_shapes, output_buffers

    @add_start_docstrings_to_model_forward(SEQ2SEQ_ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:

        if self._device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_ids, attention_mask)

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return BaseModelOutput(
                last_hidden_state=output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
            )
        else:
            onnx_inputs = {"input_ids": input_ids.cpu().detach().numpy()}

            # Add the attention_mask inputs when needed
            if "attention_mask" in self.input_names:
                onnx_inputs["attention_mask"] = attention_mask.cpu().detach().numpy()

            # Run inference
            outputs = self.session.run(None, onnx_inputs)
            last_hidden_state = torch.from_numpy(outputs[self.output_names["last_hidden_state"]]).to(self._device)

            return BaseModelOutput(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTEncoderForWhisper(ORTEncoder):
    """
    Encoder model for ONNX Runtime inference for Whisper model.

    Arguments:
        session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    def prepare_io_binding(
        self,
        input_features: torch.FloatTensor = None,
    ):
        io_binding = self.session.io_binding()

        # bind input ids
        io_binding.bind_input(
            "input_features",
            input_features.device.type,
            self._device.index,
            self.name_to_np_type["input_features"],
            tuple(input_features.shape),
            input_features.data_ptr(),
        )

        # bind logits
        output_shape, output_buffer = self.prepare_output_buffer(
            batch_size=input_features.size(0),
            sequence_length=input_features.size(2) // 2,
        )
        io_binding.bind_output(
            "last_hidden_state",
            output_buffer.device.type,
            self._device.index,
            self.name_to_np_type["last_hidden_state"],
            output_shape,
            output_buffer.data_ptr(),
        )
        output_shapes = {"last_hidden_state": output_shape}
        output_buffers = {"last_hidden_state": output_buffer}

        return io_binding, output_shapes, output_buffers

    @add_start_docstrings_to_model_forward(WHISPER_ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.FloatTensor,
        **kwargs,
    ) -> BaseModelOutput:
        if self._device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_features)

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return BaseModelOutput(
                last_hidden_state=output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
            )
        else:
            onnx_inputs = {"input_features": input_features.cpu().detach().numpy()}

            # Run inference
            outputs = self.session.run(None, onnx_inputs)
            last_hidden_state = torch.from_numpy(outputs[self.output_names["last_hidden_state"]]).to(self._device)

            return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoder:
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.

    Arguments:
        session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the decoder.
    """

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        config: transformers.PretrainedConfig,
        device: torch.device,
        use_io_binding: bool = True,
        **kwargs
    ):
        self.session = session
        self.config = config
        self.normalized_config = ORTConfigManager.get_normalized_config_class(self.config.model_type)(self.config)
        self._device = device
        self.use_io_binding = use_io_binding
        self.session_inputs = {output_key.name: idx for idx, output_key in enumerate(self.session.get_inputs())}
        self.session_outputs = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.session_input_names = list(self.session_inputs.keys())
        self.session_output_names = list(self.session_outputs.keys())
        self.key_value_input_names = [
            key for key in self.session_input_names if ("key_values" in key or ".key" in key or ".value" in key)
        ]
        self.key_value_output_names = [
            key for key in self.session_output_names if ("key_values" in key or ".key" in key or ".value" in key)
        ]
        self.name_to_np_type = TypeHelper.get_io_numpy_type_map(self.session) if self.use_io_binding else None

    def prepare_output_buffer(
        self,
        output_name,
        batch_size=None,
        sequence_length=None,
        encoder_sequence_length=None,
        past_sequence_length=None,
        is_self_attn=False,
    ):
        """
        Prepare the buffer of outputs(`logits`/`key_values`/`loss`) with 1D tensors.
        """
        ort_type = TypeHelper.get_output_type(self.session, output_name)
        torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
        if output_name == "loss":
            output_shape = (1,)
            output_buffer = torch.empty(1, dtype=torch_type, device=self._device).contiguous()
        elif output_name == "logits":
            output_shape = (batch_size, sequence_length, self.normalized_config.vocab_size)
            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self._device).contiguous()
        elif "key_values" in output_name or ".key" in output_name or ".value" in output_name:
            num_attention_heads = self.normalized_config.num_attention_heads
            hidden_size = self.normalized_config.hidden_size
            embed_size_per_head = hidden_size // num_attention_heads
            if is_self_attn:
                if past_sequence_length is not None:
                    sequence_length += past_sequence_length
                output_shape = (batch_size, num_attention_heads, sequence_length, embed_size_per_head)
            else:
                output_shape = (batch_size, num_attention_heads, encoder_sequence_length, embed_size_per_head)

            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self._device).contiguous()

        return output_shape, output_buffer

    def prepare_io_binding(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        io_binding = self.session.io_binding()

        # bind input ids
        input_ids = input_ids.contiguous()
        io_binding.bind_input(
            "input_ids",
            input_ids.device.type,
            self._device.index,
            self.name_to_np_type["input_ids"],
            list(input_ids.size()),
            input_ids.data_ptr(),
        )

        # bind encoder attention mask
        if "encoder_attention_mask" in self.session_input_names:
            encoder_attention_mask = encoder_attention_mask.contiguous()
            io_binding.bind_input(
                "encoder_attention_mask",
                encoder_attention_mask.device.type,
                self._device.index,
                self.name_to_np_type["encoder_attention_mask"],
                list(encoder_attention_mask.size()),
                encoder_attention_mask.data_ptr(),
            )

        # bind encoder hidden states
        if "encoder_hidden_states" in self.session_input_names:
            encoder_hidden_states = encoder_hidden_states.contiguous()
            io_binding.bind_input(
                "encoder_hidden_states",
                encoder_hidden_states.device.type,
                self._device.index,
                self.name_to_np_type["encoder_hidden_states"],
                list(encoder_hidden_states.size()),
                encoder_hidden_states.data_ptr(),
            )

        # bind past key values
        if past_key_values is not None:
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                past_key_value = past_key_value.contiguous()
                io_binding.bind_input(
                    input_name,
                    past_key_value.device.type,
                    self._device.index,
                    self.name_to_np_type[input_name],
                    list(past_key_value.size()),
                    past_key_value.data_ptr(),
                )

        # bind labels
        if "labels" in self.session_input_names:
            labels = labels.contiguous()
            io_binding.bind_input(
                "labels",
                labels.device.type,
                self._device.index,
                self.name_to_np_type["labels"],
                list(labels.size()),
                labels.data_ptr(),
            )

        # bind outputs
        # bind logits
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
        # bind loss
        if "loss" in self.session_output_names:
            loss_shape, loss_buffer = self.prepare_output_buffer(output_name="loss")
            io_binding.bind_output(
                "loss",
                loss_buffer.device.type,
                self._device.index,
                self.name_to_np_type["loss"],
                loss_shape,
                loss_buffer.data_ptr(),
            )
            output_shapes["loss"] = loss_shape
            output_buffers["loss"] = loss_buffer

        # bind past key values
        num_pkv = 4  # number of self-attention and cross-attention per decoder layer
        for pkv_names_per_layer in [
            self.key_value_output_names[i : i + num_pkv] for i in range(0, len(self.key_value_output_names), num_pkv)
        ]:
            # bind a self attention and a cross-attention each time(2)
            for i in range(2):
                # bind self-attention past key values(2)
                self_name = pkv_names_per_layer[i]
                self_pkv_shape, self_pkv_buffer = self.prepare_output_buffer(
                    output_name=self_name,
                    batch_size=input_ids.size(0),
                    sequence_length=input_ids.size(1),
                    past_sequence_length=past_key_values[0].size(2)
                    if past_key_values
                    else None,  # sequence length of self-attention key for layer.0
                    is_self_attn=True,
                )
                io_binding.bind_output(
                    self_name,
                    self_pkv_buffer.device.type,
                    self._device.index,
                    self.name_to_np_type[self_name],
                    self_pkv_shape,
                    self_pkv_buffer.data_ptr(),
                )
                # set -1 for sequence_length as it could be larger than the real sequence_length for creating buffer
                self_pkv_shape = self_pkv_shape[:2] + (-1,) + self_pkv_shape[3:]
                output_shapes[self_name] = self_pkv_shape
                output_buffers[self_name] = self_pkv_buffer

                # bind cross-attention past key values(2)
                cross_name = pkv_names_per_layer[i + 2]
                cross_pkv_shape, cross_pkv_buffer = self.prepare_output_buffer(
                    output_name=cross_name,
                    batch_size=input_ids.size(0),
                    encoder_sequence_length=encoder_hidden_states.size(1),
                )
                io_binding.bind_output(
                    cross_name,
                    cross_pkv_buffer.device.type,
                    self._device.index,
                    self.name_to_np_type[cross_name],
                    cross_pkv_shape,
                    cross_pkv_buffer.data_ptr(),
                )
                # set -1 for sequence_length as it could be larger than the real sequence_length for creating buffer
                cross_pkv_shape = cross_pkv_shape[:2] + (-1,) + cross_pkv_shape[3:]
                output_shapes[cross_name] = cross_pkv_shape
                output_buffers[cross_name] = cross_pkv_buffer

        return io_binding, output_shapes, output_buffers

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        # Flatten the past_key_values
        if past_key_values is not None:
            past_key_values = [past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer]

        if self._device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids, encoder_hidden_states, encoder_attention_mask, past_key_values, labels
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            past_key_values = tuple()
            for name in self.session_output_names:
                if "key_values" in name or ".key" in name or ".value" in name:
                    past_key_values += (output_buffers[name].view(output_shapes[name]),)

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # cross-attention per decoder layer
            num_pkv = 4
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))

            logits = output_buffers["logits"].view(output_shapes["logits"])

            loss = None
            if "loss" in self.session_output_names:
                loss = output_buffers["loss"].view(output_shapes["loss"])
        else:
            onnx_inputs = {
                "input_ids": input_ids.cpu().detach().numpy(),
            }

            # Add the encoder_attention_mask inputs when needed
            if "encoder_attention_mask" in self.session_input_names:
                onnx_inputs["encoder_attention_mask"] = encoder_attention_mask.cpu().detach().numpy()

            # Add the encoder_hidden_states inputs when needed
            if "encoder_hidden_states" in self.session_input_names:
                onnx_inputs["encoder_hidden_states"] = encoder_hidden_states.cpu().detach().numpy()

            if past_key_values is not None:
                # Add the past_key_values to the decoder inputs
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                    onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

            if "labels" in self.session_input_names:
                # TODO: Any preprocessing like  `self._shift_right(labels)`?
                onnx_inputs["labels"] = labels.cpu().detach().numpy()

            # Run inference
            outputs = self.session.run(None, onnx_inputs)
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            past_key_values = tuple(
                torch.from_numpy(outputs[self.session_outputs[key]]).to(self._device)
                for key in self.session_output_names
                if "key_values" in key or ".key" in key or ".value" in key
            )

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # cross-attention per decoder layer
            num_pkv = 4
            past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))
            logits = torch.from_numpy(outputs[self.session_outputs["logits"]]).to(self._device)

            loss = None
            if "loss" in self.session_output_names:
                loss = torch.from_numpy(outputs[self.session_outputs["loss"]]).to(self._device)

        # converts output to namedtuple for pipelines post-processing
        return Seq2SeqLMOutput(loss=loss, logits=logits, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTModelForSeq2SeqLM(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.
    """

    export_feature = "seq2seq-lm"
    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_encoder(
        self,
        session: onnxruntime.InferenceSession,
        config: transformers.PretrainedConfig,
        device: torch.device,
        use_io_binding: bool = True,
    ) -> ORTEncoder:
        return ORTEncoder(
            session=session,
            config=config,
            device=device,
            use_io_binding=use_io_binding,
            main_input_name=self.main_input_name,
        )

    def get_encoder_onnx_config(encoder_config: PretrainedConfig) -> EncoderOnnxConfig:
        return EncoderOnnxConfig(encoder_config, task="default")

    def get_decoder_onnx_config(
        decoder_config: PretrainedConfig, export_feature: str, use_past: bool = False
    ) -> DecoderOnnxConfig:
        return DecoderOnnxConfig(decoder_config, export_feature, use_past=use_past)

    @add_start_docstrings_to_model_forward(
        SEQ2SEQ_ONNX_MODEL_DOCSTRING.format("batch_size, sequence_length")
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
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:

        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Decode
        if past_key_values is None or self.decoder_with_past is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                labels=labels,
            )
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids[:, -1:],  # Cut decoder_input_ids if past is used
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                labels=labels,
            )

        return Seq2SeqLMOutput(
            loss=decoder_outputs.get("loss", None),
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ) -> Dict:

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
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


class ORTModelForSpeechSeq2Seq(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Speech Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.
    """

    export_feature = "speech2seq-lm"
    auto_model_class = AutoModelForSpeechSeq2Seq
    main_input_name = "input_features"

    _MODEL_TYPE_TO_ORTENCODER = {
        "whisper": ORTEncoderForWhisper,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_encoder(
        self,
        session: onnxruntime.InferenceSession,
        config: transformers.PretrainedConfig,
        device: torch.device,
        use_io_binding: bool = True,
    ) -> ORTEncoder:
        if config.model_type not in self._MODEL_TYPE_TO_ORTENCODER:
            raise KeyError(
                f"{config.model_type} is not supported yet. "
                f"Only {list(self._MODEL_TYPE_TO_ORTENCODER.keys())} are supported. "
                f"If you want to support {config.model_type} please propose a PR or open up an issue."
            )
        return self._MODEL_TYPE_TO_ORTENCODER[config.model_type](
            session=session,
            config=config,
            device=device,
            use_io_binding=use_io_binding,
            main_input_name=self.main_input_name,
        )

    def get_encoder_onnx_config(encoder_config: PretrainedConfig) -> SpeechSeq2SeqEncoderOnnxConfig:
        return SpeechSeq2SeqEncoderOnnxConfig(encoder_config, task="default")

    def get_decoder_onnx_config(
        decoder_config: PretrainedConfig, export_feature: str, use_past: bool = False
    ) -> SpeechSeq2SeqDecoderOnnxConfig:
        return SpeechSeq2SeqDecoderOnnxConfig(decoder_config, export_feature, use_past=use_past)

    @add_start_docstrings_to_model_forward(
        SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING.format("batch_size, feature_size, sequence_length")
        + AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForSpeechSeq2Seq",
            checkpoint="optimum/whisper-tiny.en",
        )
    )
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:

        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_features=input_features)

        # Decode
        if past_key_values is None or self.decoder_with_past is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                labels=labels,
            )
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids[:, -1:],  # Cut decoder_input_ids if past is used
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                labels=labels,
            )

        return Seq2SeqLMOutput(
            loss=decoder_outputs.get("loss", None),
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ) -> Dict:

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
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
