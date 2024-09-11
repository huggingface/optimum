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
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ConfigMixin,
    DDIMScheduler,
    LatentConsistencyModelPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
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
from ..onnx.utils import _get_external_data_paths
from ..pipelines.diffusers.pipeline_latent_consistency import LatentConsistencyPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin
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
    parse_device,
    validate_provider_availability,
)


logger = logging.getLogger(__name__)


class ORTPipeline(ORTModel):
    auto_model_class = None
    model_type = "onnx_pipeline"

    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    def __init__(
        self,
        vae_decoder_session: ort.InferenceSession,
        unet_session: ort.InferenceSession,
        tokenizer: CLIPTokenizer,
        config: Dict[str, Any],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_2_session: Optional[ort.InferenceSession] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
    ):
        """
        Args:
            vae_decoder_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the VAE decoder
            unet_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the U-NET.
            tokenizer (`CLIPTokenizer`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
                for the text encoder.
            config (`Dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            scheduler (`Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]`):
                A scheduler to be used in combination with the U-NET component to denoise the encoded image latents.
            feature_extractor (`Optional[CLIPFeatureExtractor]`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
            vae_encoder_session (`Optional[ort.InferenceSession]`, defaults to `None`):
                The ONNX Runtime inference session associated to the VAE encoder.
            text_encoder_session (`Optional[ort.InferenceSession]`, defaults to `None`):
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
        self.shared_attributes_init(
            model=vae_decoder_session,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )
        self._internal_dict = config
        self.vae_decoder = ORTModelVaeDecoder(vae_decoder_session, self)
        self.vae_decoder_model_path = Path(vae_decoder_session._model_path)
        self.unet = ORTModelUnet(unet_session, self)
        self.unet_model_path = Path(unet_session._model_path)

        if text_encoder_session is not None:
            self.text_encoder_model_path = Path(text_encoder_session._model_path)
            self.text_encoder = ORTModelTextEncoder(text_encoder_session, self)
        else:
            self.text_encoder_model_path = None
            self.text_encoder = None

        if vae_encoder_session is not None:
            self.vae_encoder_model_path = Path(vae_encoder_session._model_path)
            self.vae_encoder = ORTModelVaeEncoder(vae_encoder_session, self)
        else:
            self.vae_encoder_model_path = None
            self.vae_encoder = None

        if text_encoder_2_session is not None:
            self.text_encoder_2_model_path = Path(text_encoder_2_session._model_path)
            self.text_encoder_2 = ORTModelTextEncoder(text_encoder_2_session, self)
        else:
            self.text_encoder_2_model_path = None
            self.text_encoder_2 = None

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None

        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
            DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER: self.vae_encoder,
            DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER: self.text_encoder_2,
        }

        # Modify config to keep the resulting model compatible with diffusers pipelines
        for name in sub_models.keys():
            self._internal_dict[name] = (
                ("diffusers", "OnnxRuntimeModel") if sub_models[name] is not None else (None, None)
            )
        self._internal_dict.pop("vae", None)

        self.vae_scale_factor = 2 ** (len(self.vae_decoder.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

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
        src_to_dst_path = {
            self.vae_decoder_model_path: save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.text_encoder_model_path: save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.unet_model_path: save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER / ONNX_WEIGHTS_NAME,
        }

        sub_models_to_save = {
            self.vae_encoder_model_path: DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            self.text_encoder_2_model_path: DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
        }
        for path, subfolder in sub_models_to_save.items():
            if path is not None:
                src_to_dst_path[path] = save_directory / subfolder / ONNX_WEIGHTS_NAME

        # TODO: Modify _get_external_data_paths to give dictionnary
        src_paths = list(src_to_dst_path.keys())
        dst_paths = list(src_to_dst_path.values())
        # Add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            config_path = src_path.parent / self.sub_component_config_name
            if config_path.is_file():
                shutil.copyfile(config_path, dst_path.parent / self.sub_component_config_name)

        self.scheduler.save_pretrained(save_directory / "scheduler")

        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory / "feature_extractor")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory / "tokenizer_2")

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
            vae_decoder_session=vae_decoder,
            text_encoder_session=text_encoder,
            unet_session=unet,
            config=config,
            tokenizer=sub_models.get("tokenizer", None),
            scheduler=sub_models.get("scheduler"),
            feature_extractor=sub_models.get("feature_extractor", None),
            tokenizer_2=sub_models.get("tokenizer_2", None),
            vae_encoder_session=vae_encoder,
            text_encoder_2_session=text_encoder_2,
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

        self.vae_decoder.session.set_providers([provider], provider_options=[provider_options])
        self.text_encoder.session.set_providers([provider], provider_options=[provider_options])
        self.unet.session.set_providers([provider], provider_options=[provider_options])

        if self.vae_encoder is not None:
            self.vae_encoder.session.set_providers([provider], provider_options=[provider_options])

        self.providers = self.vae_decoder.session.get_providers()
        self._device = device

        return self

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)


class ORTPipelinePart(ORTModelPart):
    CONFIG_NAME = "config.json"

    def __init__(self, session: ort.InferenceSession, parent_model: ORTPipeline):
        config_path = Path(session._model_path).parent / self.CONFIG_NAME

        if config_path.is_file():
            self.config = FrozenDict(parent_model._dict_from_json_file(config_path))
        else:
            self.config = FrozenDict({})

        super().__init__(session, parent_model)

    @property
    def input_dtype(self):
        # for backward compatibility and diffusion mixins (will be standardized in the future)
        return {name: TypeHelper.ort_type_to_numpy_type(ort_type) for name, ort_type in self.input_dtypes.items()}


class ORTModelTextEncoder(ORTPipelinePart):
    def forward(self, input_ids: Union[np.ndarray, torch.Tensor]):
        use_torch = isinstance(input_ids, torch.Tensor)

        model_inputs = {"input_ids": input_ids}

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        if any("hidden_states" in model_output for model_output in model_outputs):
            model_outputs["hidden_states"] = []

            for i in range(self.config.num_hidden_layers):
                model_outputs["hidden_states"].append(model_outputs.pop(f"hidden_states.{i}"))

            # exporter doesnt duplicate last hidden state for some reason
            # (only returned once as last_hidden_state and not part of the list of hidden_states)
            model_outputs["hidden_states"].append(model_outputs.get("last_hidden_state"))

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
    ):
        use_torch = isinstance(sample, torch.Tensor)

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
            "timestep_cond": timestep_cond,
        }

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        return ModelOutput(**model_outputs)


class ORTModelVaeDecoder(ORTPipelinePart):
    def forward(self, latent_sample: Union[np.ndarray, torch.Tensor]):
        use_torch = isinstance(latent_sample, torch.Tensor)

        model_inputs = {"latent_sample": latent_sample}

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        return ModelOutput(**model_outputs)


class ORTModelVaeEncoder(ORTPipelinePart):
    def forward(self, sample: Union[np.ndarray, torch.Tensor]):
        use_torch = isinstance(sample, torch.Tensor)

        model_inputs = {"sample": sample}

        onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        return ModelOutput(**model_outputs)


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionPipeline(ORTPipeline, StableDiffusionPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionPipeline

    __call__ = StableDiffusionPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionImg2ImgPipeline(ORTPipeline, StableDiffusionImg2ImgPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionImg2ImgPipeline

    __call__ = StableDiffusionImg2ImgPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionInpaintPipeline(ORTPipeline, StableDiffusionInpaintPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionInpaintPipeline

    __call__ = StableDiffusionInpaintPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTLatentConsistencyModelPipeline(ORTPipeline, LatentConsistencyPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = LatentConsistencyModelPipeline

    __call__ = LatentConsistencyPipelineMixin.__call__


class ORTStableDiffusionXLPipelineBase(ORTPipeline):
    def __init__(
        self,
        vae_decoder_session: ort.InferenceSession,
        text_encoder_session: ort.InferenceSession,
        unet_session: ort.InferenceSession,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_2_session: Optional[ort.InferenceSession] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae_decoder_session=vae_decoder_session,
            text_encoder_session=text_encoder_session,
            unet_session=unet_session,
            config=config,
            tokenizer=tokenizer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            vae_encoder_session=vae_encoder_session,
            text_encoder_2_session=text_encoder_2_session,
            tokenizer_2=tokenizer_2,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            if not is_invisible_watermark_available():
                raise ImportError(
                    "`add_watermarker` requires invisible-watermark to be installed, which can be installed with `pip install invisible-watermark`."
                )

            from ..pipelines.diffusers.watermark import StableDiffusionXLWatermarker

            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLPipeline(ORTStableDiffusionXLPipelineBase, StableDiffusionXLPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLPipeline

    __call__ = StableDiffusionXLPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLImg2ImgPipeline(ORTStableDiffusionXLPipelineBase, StableDiffusionXLImg2ImgPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLImg2ImgPipeline

    __call__ = StableDiffusionXLImg2ImgPipelineMixin.__call__


SUPPORTED_ORT_PIPELINES = [
    ORTStableDiffusionPipeline,
    ORTStableDiffusionImg2ImgPipeline,
    ORTStableDiffusionInpaintPipeline,
    ORTLatentConsistencyModelPipeline,
    ORTStableDiffusionXLPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
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
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
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
    ]
)

ORT_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", ORTStableDiffusionInpaintPipeline),
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
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
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
