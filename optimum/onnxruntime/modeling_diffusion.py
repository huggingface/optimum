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
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, is_invisible_watermark_available
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.file_utils import add_end_docstrings

import onnxruntime as ort

from ..exporters.onnx import main_export
from ..onnx.utils import _get_external_data_paths
from ..pipelines.diffusers.pipeline_latent_consistency import LatentConsistencyPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin
from ..pipelines.diffusers.pipeline_utils import VaeImageProcessor
from ..utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
    _ORT_TO_NP_TYPE,
    ONNX_WEIGHTS_NAME,
    get_provider_for_device,
    parse_device,
    validate_provider_availability,
)


logger = logging.getLogger(__name__)


class ORTStableDiffusionPipelineBase(ORTModel):
    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    base_model_prefix = "onnx_model"
    config_name = "model_index.json"
    sub_component_config_name = "config.json"

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
    ):
        """
        Args:
            vae_decoder_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the VAE decoder.
            text_encoder_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the text encoder.
            unet_session (`ort.InferenceSession`):
                The ONNX Runtime inference session associated to the U-NET.
            config (`Dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            tokenizer (`CLIPTokenizer`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            scheduler (`Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]`):
                A scheduler to be used in combination with the U-NET component to denoise the encoded image latents.
            feature_extractor (`Optional[CLIPFeatureExtractor]`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
            vae_encoder_session (`Optional[ort.InferenceSession]`, defaults to `None`):
                The ONNX Runtime inference session associated to the VAE encoder.
            use_io_binding (`Optional[bool]`, defaults to `None`):
                Whether to use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            model_save_dir (`Optional[str]`, defaults to `None`):
                The directory under which the model exported to ONNX was saved.
        """
        self.shared_attributes_init(
            vae_decoder_session,
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

        if "block_out_channels" in self.vae_decoder.config:
            self.vae_scale_factor = 2 ** (len(self.vae_decoder.config["block_out_channels"]) - 1)
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

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
            text_encoder_2_path=new_model_save_dir
            / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER
            / text_encoder_2_file_name,
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
    ) -> "ORTStableDiffusionPipeline":
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


# TODO : Use ORTModelPart once IOBinding support is added
class _ORTDiffusionModelPart:
    """
    For multi-file ONNX models, represents a part of the model.
    It has its own `onnxruntime.InferenceSession`, and can perform a forward pass.
    """

    CONFIG_NAME = "config.json"

    def __init__(self, session: ort.InferenceSession, parent_model: ORTModel):
        self.session = session
        self.parent_model = parent_model
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        config_path = Path(session._model_path).parent / self.CONFIG_NAME
        self.config = self.parent_model._dict_from_json_file(config_path) if config_path.is_file() else {}
        self.input_dtype = {inputs.name: _ORT_TO_NP_TYPE[inputs.type] for inputs in self.session.get_inputs()}

    @property
    def device(self):
        return self.parent_model.device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ORTModelTextEncoder(_ORTDiffusionModelPart):
    def forward(self, input_ids: np.ndarray):
        onnx_inputs = {
            "input_ids": input_ids,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelUnet(_ORTDiffusionModelPart):
    def __init__(self, session: ort.InferenceSession, parent_model: ORTModel):
        super().__init__(session, parent_model)

    def forward(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        text_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
        timestep_cond: Optional[np.ndarray] = None,
    ):
        onnx_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if text_embeds is not None:
            onnx_inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            onnx_inputs["time_ids"] = time_ids
        if timestep_cond is not None:
            onnx_inputs["timestep_cond"] = timestep_cond
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelVaeDecoder(_ORTDiffusionModelPart):
    def forward(self, latent_sample: np.ndarray):
        onnx_inputs = {
            "latent_sample": latent_sample,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelVaeEncoder(_ORTDiffusionModelPart):
    def forward(self, sample: np.ndarray):
        onnx_inputs = {
            "sample": sample,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionPipeline(ORTStableDiffusionPipelineBase, StableDiffusionPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline).
    """

    __call__ = StableDiffusionPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionImg2ImgPipeline(ORTStableDiffusionPipelineBase, StableDiffusionImg2ImgPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    __call__ = StableDiffusionImg2ImgPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionInpaintPipeline(ORTStableDiffusionPipelineBase, StableDiffusionInpaintPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    __call__ = StableDiffusionInpaintPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTLatentConsistencyModelPipeline(ORTStableDiffusionPipelineBase, LatentConsistencyPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    __call__ = LatentConsistencyPipelineMixin.__call__


class ORTStableDiffusionXLPipelineBase(ORTStableDiffusionPipelineBase):
    auto_model_class = StableDiffusionXLImg2ImgPipeline

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

    __call__ = StableDiffusionXLPipelineMixin.__call__


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLImg2ImgPipeline(ORTStableDiffusionXLPipelineBase, StableDiffusionXLImg2ImgPipelineMixin):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    __call__ = StableDiffusionXLImg2ImgPipelineMixin.__call__
