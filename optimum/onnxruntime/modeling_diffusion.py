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
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTokenizer

import onnxruntime as ort

from ..exporters.onnx import (
    export_models,
    get_stable_diffusion_models_for_export,
)
from ..exporters.tasks import TasksManager
from ..onnx.utils import _get_external_data_paths
from ..pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from ..utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)
from .modeling_ort import ORTModel
from .utils import (
    _ORT_TO_NP_TYPE,
    ONNX_WEIGHTS_NAME,
    get_provider_for_device,
    parse_device,
    validate_provider_availability,
)


logger = logging.getLogger(__name__)


class ORTStableDiffusionPipeline(ORTModel, StableDiffusionPipelineMixin):
    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    base_model_prefix = "onnx_model"
    config_name = "model_index.json"

    def __init__(
        self,
        vae_decoder_session: ort.InferenceSession,
        text_encoder_session: ort.InferenceSession,
        unet_session: ort.InferenceSession,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
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
        self.text_encoder = ORTModelTextEncoder(text_encoder_session, self)
        self.text_encoder_model_path = Path(text_encoder_session._model_path)
        self.unet = ORTModelUnet(unet_session, self)
        self.unet_model_path = Path(unet_session._model_path)
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
        }
        for name in sub_models.keys():
            self._internal_dict[name] = ("optimum", sub_models[name].__class__.__name__)
        self._internal_dict.pop("vae", None)

    @staticmethod
    def load_model(
        vae_decoder_path: Union[str, Path],
        text_encoder_path: Union[str, Path],
        unet_path: Union[str, Path],
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
            provider (`str`, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[ort.SessionOptions]`, defaults to `None`):
                ONNX Runtime session options to use for loading the model. Defaults to `None`.
            provider_options (`Optional[Dict]`, defaults to `None`):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html . Defaults to `None`.
        """
        vae_decoder_session = ORTModel.load_model(vae_decoder_path, provider, session_options, provider_options)
        text_encoder_session = ORTModel.load_model(text_encoder_path, provider, session_options, provider_options)
        unet_session = ORTModel.load_model(unet_path, provider, session_options, provider_options)

        return vae_decoder_session, text_encoder_session, unet_session

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
    ):
        save_directory = Path(save_directory)
        src_to_dst_path = {
            self.vae_decoder_model_path: save_directory
            / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER
            / vae_decoder_file_name,
            self.text_encoder_model_path: save_directory
            / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER
            / text_encoder_file_name,
            self.unet_model_path: save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
        }

        # TODO: Modify _get_external_data_paths to give dictionnary
        src_paths = list(src_to_dst_path.keys())
        dst_paths = list(src_to_dst_path.values())
        # Add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        self.tokenizer.save_pretrained(save_directory.joinpath("tokenizer"))
        self.scheduler.save_pretrained(save_directory.joinpath("scheduler"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory.joinpath("feature_extractor"))

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        if provider == "TensorrtExecutionProvider":
            raise ValueError("The provider `'TensorrtExecutionProvider'` is not supported")

        model_id = str(model_id)
        sub_models_to_load, _, _ = cls.extract_init_dict(config)
        sub_models_names = set(sub_models_to_load.keys()).intersection({"feature_extractor", "tokenizer", "scheduler"})
        sub_models = {}

        if not os.path.isdir(model_id):
            allow_patterns = [os.path.join(k, "*") for k in config.keys() if not k.startswith("_")]
            allow_patterns += list(
                {
                    vae_decoder_file_name,
                    text_encoder_file_name,
                    unet_file_name,
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
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],
            )
        new_model_save_dir = Path(model_id)
        for name in sub_models_names:
            library_name, library_classes = sub_models_to_load[name]
            if library_classes is not None:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (new_model_save_dir / name).is_dir():
                    sub_models[name] = load_method(new_model_save_dir / name)
                else:
                    sub_models[name] = load_method(new_model_save_dir)

        inference_sessions = cls.load_model(
            vae_decoder_path=new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            text_encoder_path=new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            unet_path=new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
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
            *inference_sessions,
            config=config,
            tokenizer=sub_models["tokenizer"],
            scheduler=sub_models["scheduler"],
            feature_extractor=sub_models.pop("feature_extractor", None),
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTStableDiffusionPipeline":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        model = TasksManager.get_model_from_task(
            task,
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        output_names = [
            os.path.join(DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
            os.path.join(DIFFUSION_MODEL_UNET_SUBFOLDER, ONNX_WEIGHTS_NAME),
            os.path.join(DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
            os.path.join(DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
        ]
        models_and_onnx_configs = get_stable_diffusion_models_for_export(model)

        model.save_config(save_dir_path)
        model.tokenizer.save_pretrained(save_dir_path.joinpath("tokenizer"))
        model.scheduler.save_pretrained(save_dir_path.joinpath("scheduler"))
        if model.feature_extractor is not None:
            model.feature_extractor.save_pretrained(save_dir_path.joinpath("feature_extractor"))

        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=save_dir_path,
            output_names=output_names,
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
        self.device = device
        self.vae_decoder.session.set_providers([provider], provider_options=[provider_options])
        self.text_encoder.session.set_providers([provider], provider_options=[provider_options])
        self.unet.session.set_providers([provider], provider_options=[provider_options])
        self.providers = self.vae_decoder.session.get_providers()
        return self

    def __call__(self, *args, **kwargs):
        return StableDiffusionPipelineMixin.__call__(self, *args, **kwargs)

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

    def __init__(self, session: ort.InferenceSession, parent_model: ORTModel):
        self.session = session
        self.parent_model = parent_model
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

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
        self.input_dtype = {inputs.name: _ORT_TO_NP_TYPE[inputs.type] for inputs in self.session.get_inputs()}

    def forward(self, sample: np.ndarray, timestep: np.ndarray, encoder_hidden_states: np.ndarray):
        onnx_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelVaeDecoder(_ORTDiffusionModelPart):
    def forward(self, latent_sample: np.ndarray):
        onnx_inputs = {
            "latent_sample": latent_sample,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs
