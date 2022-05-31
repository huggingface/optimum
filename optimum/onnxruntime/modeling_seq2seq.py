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
from typing import Any, DefaultDict, Dict, Mapping, Optional, Set, Tuple, Union

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
from optimum.onnx.configuration import DecoderOnnxConfig, EncoderOnnxConfig
from optimum.onnx.modeling_seq2seq import _DecoderWithLMhead

from .modeling_ort import ORTModel
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME, _is_gpu_available


logger = logging.getLogger(__name__)


SEQ2SEQ_ONNX_MODEL_START_DOCSTRING = r"""
    Arguments:
        config (`transformers.PretrainedConfig`):
            [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is the model configuration class with all the parameters of the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
        encoder_session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
        decoder_session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the decoder.
        decoder_with_past_session (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the decoder with past key values.
        config (`transformers.PretrainedConfig`):
            An instance of the configuration associated to the model.
        encoder_file_name(`str`, *optional*):
            The encoder model file name overwriting the default file name, allowing to save the encoder model with
            a different name.
        decoder_file_name(`str`, *optional*):
            The decoder model file name overwriting the default file name, allowing to save the decoder model with
            a different name.
        decoder_with_past_file_name(`str`, *optional*):
            The decoder with past key values model file name overwriting the default file name, allowing to save
            the decoder model with a different name.
"""

ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
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

@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.
    """,
    SEQ2SEQ_ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForConditionalGeneration(ORTModel):
    # Used in from_transformers to export model to onnx
    pipeline_task = "seq2seq-lm"
    auto_model_class = AutoModelForSeq2SeqLM

    def __init__(self, encoder=None, decoder=None, decoder_with_past=None, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.encoder = ORTEncoder(encoder)
        self.decoder = ORTDecoder(decoder)
        self.decoder_with_past = ORTDecoder(decoder_with_past)
        self.encoder_file_name = kwargs.get("encoder", ONNX_ENCODER_NAME)
        self.decoder_file_name = kwargs.get("decoder", ONNX_DECODER_NAME)
        self.decoder_file_with_past_name = kwargs.get("decoder_with_past", ONNX_DECODER_WITH_PAST_NAME)

    @staticmethod
    def load_model(
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        decoder_with_past_path: Union[str, Path],
        config: transformers.PretrainedConfig,
        provider=None,
    ):
        """
        Create an instance of [`~optimum.onnxruntime.modeling_seq2seq.ORTModelForConditionalGeneration`].
        Three inference sessions will be created for respectively the encoder, decoder and decoder with past key values
        models. The default execution provider of the inference sessions is `"CUDAExecutionProvider"` if a GPU is
        available else `"CPUExecutionProvider"`.

        Arguments:
            encoder_path (`str` or `Path`):
                The path of the encoder ONNX model.
            decoder_path (`str` or `Path`):
                The path of the decoder ONNX model.
            decoder_with_past_path (`str` or `Path`):
                The path of the decoder with past key values ONNX model.
            config (`transformers.PretrainedConfig`):
                An instance of the configuration associated to the model.
            provider(`str`, *optional*):
                The ONNX Runtime provider to use for loading the model, defaults to `CUDAExecutionProvider` if GPU is
                available else `CPUExecutionProvider`
        """
        if provider is None:
            provider = "CUDAExecutionProvider" if _is_gpu_available() else "CPUExecutionProvider"

        encoder_session = onnxruntime.InferenceSession(str(encoder_path), providers=[provider])
        decoder_session = onnxruntime.InferenceSession(str(decoder_path), providers=[provider])
        decoder_with_past_session = onnxruntime.InferenceSession(str(decoder_with_past_path), providers=[provider])
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
        Save the model encoder, decoder and decoder with past key values as well as its configuration file to a
        directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_seq2seq.ORTModelForSeq2SeqLM.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            encoder_file_name(`str`, *optional*):
                The encoder model file name overwriting the default file name, allowing to save the encoder model with
                a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name overwriting the default file name, allowing to save the decoder model with
                a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
        """
        src_file_names = [self.encoder_file_name, self.decoder_file_name, self.decoder_file_with_past_name]

        dst_file_names = [
            encoder_file_name or ONNX_ENCODER_NAME,
            decoder_file_name or ONNX_DECODER_NAME,
            decoder_with_past_file_name or ONNX_DECODER_WITH_PAST_NAME,
        ]

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
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a model and its configuration file from a directory or the HF Hub.
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
                The encoder model file name overwriting the default file name, allowing to save the encoder model with
                a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name overwriting the default file name, allowing to save the decoder model with
                a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization.
        """
        config_dict = kwargs.pop("config", {})
        config = PretrainedConfig.from_dict(config_dict)
        encoder_file_name = encoder_file_name if encoder_file_name is not None else ONNX_ENCODER_NAME
        decoder_file_name = decoder_file_name if decoder_file_name is not None else ONNX_DECODER_NAME
        decoder_with_past_file_name = (
            decoder_with_past_file_name if decoder_with_past_file_name is not None else ONNX_DECODER_WITH_PAST_NAME
        )

        # Load model from a local directory
        if os.path.isdir(model_id):
            model = cls.load_model(
                encoder_path=os.path.join(model_id, encoder_file_name),
                decoder_path=os.path.join(model_id, decoder_file_name),
                decoder_with_past_path=os.path.join(model_id, decoder_with_past_file_name),
                config=config,
            )
            kwargs["model_save_dir"] = Path(model_id)
        # Load model from hub
        else:
            default_file_names = [ONNX_ENCODER_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME]
            model_file_names = [encoder_file_name, decoder_file_name, decoder_with_past_file_name]
            # Download the encoder, decoder and decoder_with_past forming the model
            for file_name, default_file_name in zip(model_file_names, default_file_names):
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                )
                kwargs[default_file_name.split(".")[0]] = Path(model_cache_path).name
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            model = cls.load_model(
                encoder_path=kwargs["model_save_dir"].joinpath(kwargs["encoder"]),
                decoder_path=kwargs["model_save_dir"].joinpath(kwargs["decoder"]),
                decoder_with_past_path=kwargs["model_save_dir"].joinpath(kwargs["decoder_with_past"]),
                config=config,
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
        Export through the ONNX format a vanilla Transformers model using `transformers.onnx.export_onnx`.

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
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization.
        """
        # Create local save dir in cache dir
        save_dir = Path(save_dir).joinpath(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model_save_dir"] = save_dir
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FeaturesManager.get_model_from_feature(cls.pipeline_task, model_id)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=cls.pipeline_task)
        onnx_config = model_onnx_config(model.config)
        onnx_opset = onnx_config.default_onnx_opset
        onnx_config_encoder = EncoderOnnxConfig(model.config, task="default")
        onnx_config_decoder = DecoderOnnxConfig(model.config, task=cls.pipeline_task, use_past=False)
        onnx_config_decoder_with_past = DecoderOnnxConfig(model.config, task=cls.pipeline_task, use_past=True)

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
        export(
            preprocessor=tokenizer,
            model=decoder_with_lm_head,
            config=onnx_config_decoder_with_past,
            opset=onnx_opset,
            output=save_dir.joinpath(ONNX_DECODER_WITH_PAST_NAME),
        )

        kwargs["config"] = model.config.__dict__
        return cls._from_pretrained(save_dir.as_posix(), **kwargs)


class ORTEncoder:
    """
    Encoder model for ONNX Runtime inference.

    Arguments:
        encoder (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    def __init__(self, encoder: onnxruntime.InferenceSession):
        self.encoder = encoder
        self.main_input_name = "input_ids"
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.encoder.get_outputs())}

    @add_start_docstrings_to_model_forward(ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:

        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }

        # Run inference
        outputs = self.encoder.run(None, onnx_inputs)

        return BaseModelOutput(last_hidden_state=torch.from_numpy(outputs[self.output_names["last_hidden_state"]]))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTDecoder:
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.

    Arguments:
        decoder (`onnxruntime.InferenceSession`):
            The ONNX Runtime inference session associated to the decoder.
    """

    def __init__(self, decoder: onnxruntime.InferenceSession):
        self.decoder = decoder
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.decoder.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.decoder.get_outputs())}

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
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
            past_key_values = [mha for past_key_value in past_key_values for mha in past_key_value]

            # Add the past_key_values to the decoder inputs
            for i, past_key_value in enumerate(past_key_values):
                onnx_inputs[f"past_key_values_{i}.1"] = past_key_value.cpu().detach().numpy()

        # Run inference
        outputs = self.decoder.run(None, onnx_inputs)

        # Tuple of length equal to : number of layer * number of multi-head attention
        past_key_values = tuple(
            torch.from_numpy(outputs[self.output_names[key]]) for key in self.output_names if "past_key_values" in key
        )

        # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number on multi-head attention
        # (number of self-attention and cross-attention per decoder layer)
        num_mha = 4
        past_key_values = tuple(past_key_values[i : i + num_mha] for i in range(0, len(past_key_values), num_mha))

        return Seq2SeqLMOutput(
            logits=torch.from_numpy(outputs[self.output_names["logits"]]), past_key_values=past_key_values
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTModelForSeq2SeqLM(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_input_name = "input_ids"

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

        # Decode
        if past_key_values is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids[:, -1:],  # Cut decoder_input_ids if past is used
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )

        return Seq2SeqLMOutput(logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values)

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

        # Cut input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

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

    @property
    def device(self) -> torch.device:
        # By default, ONNX Runtime places inputs and outputs on CPU (even though not optimal when the latter are
        # consumed by a device other than CPU. When preparing the decoder_input_ids for generation, the latter should
        # be placed on the same device as the other inputs (CPU). Needed for version of transformers < 4.19.0
        return torch.device("cpu")

    # Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1425
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        reordered_past = ()
        for layer_past in past:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
