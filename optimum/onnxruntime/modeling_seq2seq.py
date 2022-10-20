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
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Set, Tuple, Union

import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, default_cache_path
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.onnx import FeaturesManager, export

import onnx
import onnxruntime
from huggingface_hub import HfApi, hf_hub_download

from ..onnx.configuration import DecoderOnnxConfig, EncoderOnnxConfig
from ..onnx.modeling_seq2seq import _DecoderWithLMhead
from .modeling_ort import ORTModel
from .utils import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    _is_gpu_available,
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
"""

ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
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

_TOKENIZER_FOR_DOC = "AutoTokenizer"

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


@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.
    """,
    ONNX_INPUTS_DOCSTRING,
)
class ORTModelForConditionalGeneration(ORTModel):
    # Used in from_transformers to export model to onnx
    base_model_prefix = "onnx_model"
    export_feature = "seq2seq-lm"
    auto_model_class = AutoModelForSeq2SeqLM

    def __init__(
        self,
        encoder_session: onnxruntime.InferenceSession = None,
        decoder_session: onnxruntime.InferenceSession = None,
        decoder_with_past_session: onnxruntime.InferenceSession = None,
        config: transformers.PretrainedConfig = None,
        **kwargs
    ):
        self.config = config
        self.model_save_dir = kwargs.get("model_save_dir", None)

        self.providers = encoder_session.get_providers()
        self._device = get_device_for_provider(encoder_session.get_providers()[0])

        self.encoder = ORTEncoder(session=encoder_session, device=self._device)
        self.decoder = ORTDecoder(session=decoder_session, device=self._device)
        self.use_cache = decoder_with_past_session is not None
        # If a decoder_with_past_path is provided, an inference session for the decoder with past key/values as inputs
        # will be enabled
        self.decoder_with_past = (
            ORTDecoder(session=decoder_with_past_session, device=self._device) if self.use_cache else None
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
                Provider option dictionaries corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html . Defaults to `None`.
        """
        available_providers = onnxruntime.get_available_providers()
        if provider not in available_providers:
            raise ValueError(
                f"Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}."
            )

        encoder_session = onnxruntime.InferenceSession(
            str(encoder_path),
            providers=[provider],
            sess_options=session_options,
            provider_options=None if provider_options is None else [provider_options],
        )
        decoder_session = onnxruntime.InferenceSession(
            str(decoder_path),
            providers=[provider],
            sess_options=session_options,
            provider_options=None if provider_options is None else [provider_options],
        )

        decoder_with_past_session = None
        # If a decoder_with_past_path is provided, an inference session for the decoder with past key/values as inputs
        # will be enabled
        if decoder_with_past_path is not None:
            decoder_with_past_session = onnxruntime.InferenceSession(
                str(decoder_with_past_path),
                providers=[provider],
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
        config_dict = kwargs.pop("config", {})
        config = PretrainedConfig.from_dict(config_dict)
        encoder_file_name = encoder_file_name or ONNX_ENCODER_NAME
        decoder_file_name = decoder_file_name or ONNX_DECODER_NAME
        decoder_with_past_file_name = decoder_with_past_file_name or ONNX_DECODER_WITH_PAST_NAME

        # Load model from a local directory
        if os.path.isdir(model_id):
            decoder_with_past_path = os.path.join(model_id, decoder_with_past_file_name) if use_cache else None
            model = cls.load_model(
                encoder_path=os.path.join(model_id, encoder_file_name),
                decoder_path=os.path.join(model_id, decoder_file_name),
                decoder_with_past_path=decoder_with_past_path,
                **kwargs,
            )
            kwargs["model_save_dir"] = Path(model_id)
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
        save_dir = Path(save_dir).joinpath(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model_save_dir"] = save_dir
        use_cache = kwargs.get("use_cache", True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FeaturesManager.get_model_from_feature(cls.export_feature, model_id)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=cls.export_feature)
        onnx_config = model_onnx_config(model.config)
        onnx_opset = onnx_config.default_onnx_opset
        onnx_config_encoder = EncoderOnnxConfig(model.config, task="default")
        onnx_config_decoder = DecoderOnnxConfig(model.config, task=cls.export_feature, use_past=False)
        onnx_config_decoder_with_past = DecoderOnnxConfig(model.config, task=cls.export_feature, use_past=True)

        # Extract the encoder for ONNX export
        encoder = model.get_encoder()
        # Concatenate the decoder with the language model head for ONNX export
        decoder_with_lm_head = _DecoderWithLMhead(model)

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

        kwargs["config"] = model.config.__dict__
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
        self.encoder._device = device
        self.encoder.session.set_providers([provider], provider_options=[provider_options])
        self.decoder._device = device
        self.decoder.session.set_providers([provider], provider_options=[provider_options])
        if self.decoder_with_past is not None:
            self.decoder_with_past._device = device
            self.decoder_with_past.session.set_providers([provider], provider_options=[provider_options])
        return self


class ORTEncoder:
    """
    Encoder model for ONNX Runtime inference.

    Arguments:
        session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    def __init__(self, session: onnxruntime.InferenceSession, device: torch.device):
        self.session = session
        self._device = device
        self.main_input_name = "input_ids"
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    @add_start_docstrings_to_model_forward(ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:

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


class ORTDecoder:
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.

    Arguments:
        session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the decoder.
    """

    def __init__(self, session: onnxruntime.InferenceSession, device: torch.device):
        self.session = session
        self._device = device
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:

        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "encoder_attention_mask": encoder_attention_mask.cpu().detach().numpy(),
        }

        # Add the encoder_hidden_states inputs when needed
        if "encoder_hidden_states" in self.input_names:
            onnx_inputs["encoder_hidden_states"] = encoder_hidden_states.cpu().detach().numpy()

        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = [past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer]
            # Add the past_key_values to the decoder inputs
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

        if "labels" in self.input_names:
            # TODO: Any preprocessing like  `self._shift_right(labels)`?
            onnx_inputs["labels"] = labels.cpu().detach().numpy()

        # Run inference
        outputs = self.session.run(None, onnx_inputs)

        # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
        # self-attention layer and 2 to the cross-attention layer)
        past_key_values = tuple(
            torch.from_numpy(outputs[self.output_names[key]]).to(self._device)
            for key in self.output_names
            if "key_values" in key
        )

        # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
        # cross-attention per decoder layer
        num_pkv = 4
        past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))
        logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self._device)

        loss = None
        if "loss" in self.output_names:
            loss = torch.from_numpy(outputs[self.output_names["loss"]]).to(self._device)
        return Seq2SeqLMOutput(loss=loss, logits=logits, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTModelForSeq2SeqLM(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_input_name = "input_ids"

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
