#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

import importlib
import logging
import os
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    LatentConsistencyModelImg2ImgPipeline,
    LatentConsistencyModelPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.schedulers import SchedulerMixin
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, is_invisible_watermark_available
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.file_utils import add_end_docstrings
from transformers.modeling_outputs import ModelOutput

import onnxruntime as ort

from ..exporters.onnx import main_export
from ..onnx.utils import _get_model_external_data_paths
from ..utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)
from .base import ORTModelPart
from .io_binding import TypeHelper
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    get_provider_for_device,
    np_to_pt_generators,
    parse_device,
    validate_provider_availability,
)


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

logger = logging.getLogger(__name__)


class ORTPipeline(ORTModel):
    auto_model_class = None
    model_type = "onnx_pipeline"

    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    def __init__(
        self,
        config: Dict[str, Any],
        scheduler: SchedulerMixin,
        unet: ort.InferenceSession,
        vae_encoder: Optional[ort.InferenceSession] = None,
        vae_decoder: Optional[ort.InferenceSession] = None,
        text_encoder: Optional[ort.InferenceSession] = None,
        text_encoder_2: Optional[ort.InferenceSession] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        """
        Args:
            config (`Dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            tokenizer (`CLIPTokenizer`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
                for the text encoder.
            scheduler (`Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]`):
                A scheduler to be used in combination with the U-NET component to denoise the encoded image latents.
            unet (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the U-NET.
            feature_extractor (`Optional[CLIPFeatureExtractor]`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
            vae_encoder (`Optional[ort.InferenceSession]`, defaults to `None`):
                The ONNX Runtime inference session associated to the VAE encoder.
            text_encoder (`Optional[ort.InferenceSession]`, defaults to `None`):
                The ONNX Runtime inference session associated to the text encoder.
            tokenizer_2 (`Optional[CLIPTokenizer]`, defaults to `None`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
                for the second text encoder.
            use_io_binding (`Optional[bool]`, defaults to `None`):
                Whether to use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            model_save_dir (`Optional[str]`, defaults to `None`):
                The directory under which the model exported to ONNX was saved.
        """

        self.unet = ORTModelUnet(unet, self)
        self.vae_encoder = ORTModelVaeEncoder(vae_encoder, self) if vae_encoder is not None else None
        self.vae_decoder = ORTModelVaeDecoder(vae_decoder, self) if vae_decoder is not None else None
        self.text_encoder = ORTModelTextEncoder(text_encoder, self) if text_encoder is not None else None
        self.text_encoder_2 = ORTModelTextEncoder(text_encoder_2, self) if text_encoder_2 is not None else None

        # We create VAE encoder & decoder and wrap them in one object to
        # be used by the pipeline mixins with minimal code changes (simulating the diffusers API)
        self.vae = ORTVaeWrapper(self.vae_encoder, self.vae_decoder, self)

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.feature_extractor = feature_extractor
        self.safety_checker = kwargs.get("safety_checker", None)

        if hasattr(self.vae.config, "block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        sub_models = {
            self.unet: DIFFUSION_MODEL_UNET_SUBFOLDER,
            self.vae_decoder: DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            self.vae_encoder: DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            self.text_encoder: DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
            self.text_encoder_2: DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
        }

        # Modify config to keep the resulting model compatible with diffusers pipelines
        for model, model_name in sub_models.items():
            config[model_name] = ("optimum", model.__class__.__name__) if model is not None else (None, None)

        self._internal_dict = FrozenDict(config)
        self.shared_attributes_init(model=unet, use_io_binding=use_io_binding, model_save_dir=model_save_dir)

    @staticmethod
    def load_model(
        vae_decoder_path: Union[str, Path],
        text_encoder_path: Union[str, Path],
        unet_path: Union[str, Path],
        vae_encoder_path: Optional[Union[str, Path]] = None,
        text_encoder_2_path: Optional[Union[str, Path]] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict] = None,
    ):
        """
        Creates three inference sessions for respectively the VAE decoder, the text encoder and the U-NET models.
        The default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            vae_decoder_path (`Union[str, Path]`):
                The path to the VAE decoder ONNX model.
            text_encoder_path (`Union[str, Path]`):
                The path to the text encoder ONNX model.
            unet_path (`Union[str, Path]`):
                The path to the U-NET ONNX model.
            vae_encoder_path (`Union[str, Path]`, defaults to `None`):
                The path to the VAE encoder ONNX model.
            text_encoder_2_path (`Union[str, Path]`, defaults to `None`):
                The path to the second text decoder ONNX model.
            provider (`str`, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[ort.SessionOptions]`, defaults to `None`):
                ONNX Runtime session options to use for loading the model. Defaults to `None`.
            provider_options (`Optional[Dict]`, defaults to `None`):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html . Defaults to `None`.
        """
        vae_decoder = ORTModel.load_model(vae_decoder_path, provider, session_options, provider_options)
        unet = ORTModel.load_model(unet_path, provider, session_options, provider_options)

        sessions = {
            "vae_encoder": vae_encoder_path,
            "text_encoder": text_encoder_path,
            "text_encoder_2": text_encoder_2_path,
        }

        for key, value in sessions.items():
            if value is not None and value.is_file():
                sessions[key] = ORTModel.load_model(value, provider, session_options, provider_options)
            else:
                sessions[key] = None

        return vae_decoder, sessions["text_encoder"], unet, sessions["vae_encoder"], sessions["text_encoder_2"]

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)

        sub_models_to_save = {
            self.unet: DIFFUSION_MODEL_UNET_SUBFOLDER,
            self.vae_decoder: DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            self.vae_encoder: DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            self.text_encoder: DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
            self.text_encoder_2: DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
        }

        for model, model_folder in sub_models_to_save.items():
            if model is not None:
                model_path = Path(model.session._model_path)
                model_save_path = save_directory / model_folder / ONNX_WEIGHTS_NAME
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                # copy onnx model
                shutil.copyfile(model_path, model_save_path)
                # copy external data
                external_data_paths = _get_model_external_data_paths(model_path)
                for external_data_path in external_data_paths:
                    shutil.copyfile(external_data_path, model_save_path.parent / external_data_path.name)
                # copy config
                shutil.copyfile(
                    model_path.parent / self.sub_component_config_name,
                    model_save_path.parent / self.sub_component_config_name,
                )

        self.scheduler.save_pretrained(save_directory / "scheduler")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory / "tokenizer_2")
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory / "feature_extractor")

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
        vae_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_2_file_name: str = ONNX_WEIGHTS_NAME,
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if provider == "TensorrtExecutionProvider":
            raise ValueError("The provider `'TensorrtExecutionProvider'` is not supported")

        model_id = str(model_id)
        patterns = set(config.keys())
        sub_models_to_load = patterns.intersection({"feature_extractor", "tokenizer", "tokenizer_2", "scheduler"})

        if not os.path.isdir(model_id):
            patterns.update({"vae_encoder", "vae_decoder"})
            allow_patterns = {os.path.join(k, "*") for k in patterns if not k.startswith("_")}
            allow_patterns.update(
                {
                    vae_decoder_file_name,
                    text_encoder_file_name,
                    unet_file_name,
                    vae_encoder_file_name,
                    text_encoder_2_file_name,
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                    cls.config_name,
                }
            )
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin", "*.xml"],
            )
        new_model_save_dir = Path(model_id)

        sub_models = {}
        for name in sub_models_to_load:
            library_name, library_classes = config[name]
            if library_classes is not None:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (new_model_save_dir / name).is_dir():
                    sub_models[name] = load_method(new_model_save_dir / name)
                else:
                    sub_models[name] = load_method(new_model_save_dir)

        vae_decoder, text_encoder, unet, vae_encoder, text_encoder_2 = cls.load_model(
            vae_decoder_path=new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            text_encoder_path=new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            unet_path=new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
            vae_encoder_path=new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            text_encoder_2_path=(
                new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name
            ),
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        if use_io_binding:
            raise ValueError(
                "IOBinding is not yet available for stable diffusion model, please set `use_io_binding` to False."
            )

        return cls(
            unet=unet,
            config=config,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            scheduler=sub_models.get("scheduler"),
            tokenizer=sub_models.get("tokenizer", None),
            tokenizer_2=sub_models.get("tokenizer_2", None),
            feature_extractor=sub_models.get("feature_extractor", None),
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTPipeline":
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        return cls._from_pretrained(
            save_dir_path,
            config=config,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            model_save_dir=save_dir,
        )

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModel`: the model placed on the requested device.
        """

        device, provider_options = parse_device(device)
        provider = get_provider_for_device(device)
        validate_provider_availability(provider)  # raise error if the provider is not available

        if device.type == "cuda" and self.providers[0] == "TensorrtExecutionProvider":
            return self

        self.unet.session.set_providers([provider], provider_options=[provider_options])

        if self.vae_encoder is not None:
            self.vae_encoder.session.set_providers([provider], provider_options=[provider_options])

        if self.vae_decoder is not None:
            self.vae_decoder.session.set_providers([provider], provider_options=[provider_options])

        if self.text_encoder is not None:
            self.text_encoder.session.set_providers([provider], provider_options=[provider_options])

        if self.text_encoder_2 is not None:
            self.text_encoder_2.session.set_providers([provider], provider_options=[provider_options])

        self.providers = self.vae_decoder.session.get_providers()
        self._device = device

        return self

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)

    @property
    def _execution_device(self):
        return self.device

    def __call__(self, *args, **kwargs):
        device = self._execution_device

        for i in range(len(args)):
            args[i] = np_to_pt_generators(args[i], device)

        for k, v in kwargs.items():
            kwargs[k] = np_to_pt_generators(v, device)

        return self.auto_model_class.__call__(self, *args, **kwargs)


class ORTPipelinePart(ORTModelPart):
    def __init__(self, session: ort.InferenceSession, parent_model: ORTPipeline):
        super().__init__(session, parent_model)

        config_path = Path(session._model_path).parent / "config.json"
        config_dict = parent_model._dict_from_json_file(config_path) if config_path.is_file() else {}
        self.config = FrozenDict(config_dict)

    @property
    def input_dtype(self):
        logger.warning(
            "The `input_dtype` property is deprecated and will be removed in the next release. "
            "Please use `input_dtypes` along with `TypeHelper` to get the `numpy` types."
        )
        return {name: TypeHelper.ort_type_to_numpy_type(ort_type) for name, ort_type in self.input_dtypes.items()}


class ORTModelTextEncoder(ORTPipelinePart):
    def forward(
        self,
        input_ids: Union[np.ndarray, torch.Tensor],
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)

        model_inputs = {"input_ids": input_ids}

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        if output_hidden_states:
            model_outputs["hidden_states"] = []
            for i in range(self.config.num_hidden_layers):
                model_outputs["hidden_states"].append(model_outputs.pop(f"hidden_states.{i}"))
            model_outputs["hidden_states"].append(model_outputs.get("last_hidden_state"))
        else:
            for i in range(self.config.num_hidden_layers):
                model_outputs.pop(f"hidden_states.{i}", None)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTModelUnet(ORTPipelinePart):
    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        timestep: Union[np.ndarray, torch.Tensor],
        encoder_hidden_states: Union[np.ndarray, torch.Tensor],
        text_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        time_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        timestep_cond: Optional[Union[np.ndarray, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(sample, torch.Tensor)

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
            "timestep_cond": timestep_cond,
            **(cross_attention_kwargs or {}),
            **(added_cond_kwargs or {}),
        }

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTModelVaeEncoder(ORTPipelinePart):
    def forward(self, sample: Union[np.ndarray, torch.Tensor], return_dict: bool = False):
        use_torch = isinstance(sample, torch.Tensor)

        model_inputs = {"sample": sample}

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if "latent_parameters" in model_outputs:
            model_outputs["latent_dist"] = DiagonalGaussianDistribution(
                parameters=model_outputs.pop("latent_parameters")
            )

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTModelVaeDecoder(ORTPipelinePart):
    def forward(
        self,
        latent_sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(latent_sample, torch.Tensor)

        model_inputs = {"latent_sample": latent_sample}

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTVaeWrapper(ORTPipelinePart):
    def __init__(self, vae_encoder: ORTModelVaeEncoder, vae_decoder: ORTModelVaeDecoder, parent_model: ORTPipeline):
        super().__init__(vae_decoder.session, parent_model)
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

    def encode(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        return_dict: bool = False,
    ):
        return self.vae_encoder(sample, return_dict)

    def decode(
        self,
        latent_sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        return self.vae_decoder(latent_sample, generator, return_dict)

    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        latent_sample = self.encode(sample).latent_dist.sample(generator=generator)
        return self.decode(latent_sample, generator, return_dict)


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionPipeline(ORTPipeline, StableDiffusionPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionImg2ImgPipeline(ORTPipeline, StableDiffusionImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    main_input_name = "image"
    auto_model_class = StableDiffusionImg2ImgPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionInpaintPipeline(ORTPipeline, StableDiffusionInpaintPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionInpaintPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLPipeline(ORTPipeline, StableDiffusionXLPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLPipeline

    def __init__(self, *args, add_watermarker: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        requires_aesthetics_score = kwargs.get("requires_aesthetics_score", False)
        force_zeros_for_empty_prompt = kwargs.get("force_zeros_for_empty_prompt", True)
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)

        return add_time_ids


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLImg2ImgPipeline(ORTPipeline, StableDiffusionXLImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLImg2ImgPipeline

    def __init__(self, *args, add_watermarker: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        requires_aesthetics_score = kwargs.get("requires_aesthetics_score", False)
        force_zeros_for_empty_prompt = kwargs.get("force_zeros_for_empty_prompt", True)
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLInpaintPipeline(ORTPipeline, StableDiffusionXLInpaintPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline).
    """

    main_input_name = "image"
    auto_model_class = StableDiffusionXLInpaintPipeline

    def __init__(self, *args, add_watermarker: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        requires_aesthetics_score = kwargs.get("requires_aesthetics_score", False)
        force_zeros_for_empty_prompt = kwargs.get("force_zeros_for_empty_prompt", True)
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTLatentConsistencyModelPipeline(ORTPipeline, LatentConsistencyModelPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = LatentConsistencyModelPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTLatentConsistencyModelImg2ImgPipeline(ORTPipeline, LatentConsistencyModelImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency_img2img#diffusers.LatentConsistencyModelImg2ImgPipeline).
    """

    main_input_name = "image"
    auto_model_class = LatentConsistencyModelImg2ImgPipeline


SUPPORTED_ORT_PIPELINES = [
    ORTStableDiffusionPipeline,
    ORTStableDiffusionImg2ImgPipeline,
    ORTStableDiffusionInpaintPipeline,
    ORTStableDiffusionXLPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
    ORTStableDiffusionXLInpaintPipeline,
    ORTLatentConsistencyModelPipeline,
    ORTLatentConsistencyModelImg2ImgPipeline,
]


def _get_pipeline_class(pipeline_class_name: str, throw_error_if_not_exist: bool = True):
    for ort_pipeline_class in SUPPORTED_ORT_PIPELINES:
        if (
            ort_pipeline_class.__name__ == pipeline_class_name
            or ort_pipeline_class.auto_model_class.__name__ == pipeline_class_name
        ):
            return ort_pipeline_class

    if throw_error_if_not_exist:
        raise ValueError(f"ORTDiffusionPipeline can't find a pipeline linked to {pipeline_class_name}")


class ORTDiffusionPipeline(ConfigMixin):
    config_name = "model_index.json"

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs) -> ORTPipeline:
        load_config_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download", None),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir", None),
            "revision": kwargs.get("revision", None),
            "proxies": kwargs.get("proxies", None),
            "token": kwargs.get("token", None),
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        config = config[0] if isinstance(config, tuple) else config
        class_name = config["_class_name"]

        ort_pipeline_class = _get_pipeline_class(class_name)

        return ort_pipeline_class.from_pretrained(pretrained_model_or_path, **kwargs)


ORT_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", ORTStableDiffusionPipeline),
        ("stable-diffusion-xl", ORTStableDiffusionXLPipeline),
        ("latent-consistency", ORTLatentConsistencyModelPipeline),
    ]
)

ORT_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", ORTStableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", ORTStableDiffusionXLImg2ImgPipeline),
        ("latent-consistency", ORTLatentConsistencyModelImg2ImgPipeline),
    ]
)

ORT_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", ORTStableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", ORTStableDiffusionXLInpaintPipeline),
    ]
)

SUPPORTED_ORT_PIPELINES_MAPPINGS = [
    ORT_TEXT2IMAGE_PIPELINES_MAPPING,
    ORT_IMAGE2IMAGE_PIPELINES_MAPPING,
    ORT_INPAINT_PIPELINES_MAPPING,
]


def _get_task_class(mapping, pipeline_class_name):
    def _get_model_name(pipeline_class_name):
        for ort_pipelines_mapping in SUPPORTED_ORT_PIPELINES_MAPPINGS:
            for model_name, ort_pipeline_class in ort_pipelines_mapping.items():
                if (
                    ort_pipeline_class.__name__ == pipeline_class_name
                    or ort_pipeline_class.auto_model_class.__name__ == pipeline_class_name
                ):
                    return model_name

    model_name = _get_model_name(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    raise ValueError(f"ORTPipelineForTask can't find a pipeline linked to {pipeline_class_name} for {model_name}")


class ORTPipelineForTask(ConfigMixin):
    config_name = "model_index.json"

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs) -> ORTPipeline:
        load_config_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download", None),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir", None),
            "revision": kwargs.get("revision", None),
            "proxies": kwargs.get("proxies", None),
            "token": kwargs.get("token", None),
        }
        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        config = config[0] if isinstance(config, tuple) else config
        class_name = config["_class_name"]

        ort_pipeline_class = _get_task_class(cls.ort_pipelines_mapping, class_name)

        return ort_pipeline_class.from_pretrained(pretrained_model_or_path, **kwargs)


class ORTPipelineForText2Image(ORTPipelineForTask):
    auto_model_class = AutoPipelineForText2Image
    ort_pipelines_mapping = ORT_TEXT2IMAGE_PIPELINES_MAPPING


class ORTPipelineForImage2Image(ORTPipelineForTask):
    auto_model_class = AutoPipelineForImage2Image
    ort_pipelines_mapping = ORT_IMAGE2IMAGE_PIPELINES_MAPPING


class ORTPipelineForInpainting(ORTPipelineForTask):
    auto_model_class = AutoPipelineForInpainting
    ort_pipelines_mapping = ORT_INPAINT_PIPELINES_MAPPING
