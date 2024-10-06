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
import inspect
import logging
import os
import shutil
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, FrozenDict
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
from diffusers.utils.constants import CONFIG_NAME
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
from .io_binding import TypeHelper
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    get_provider_for_device,
    np_to_pt_generators,
    parse_device,
    validate_provider_availability,
)


logger = logging.getLogger(__name__)


class ORTPipeline(ORTModel, ConfigMixin):
    config_name = "model_index.json"

    def __init__(
        self,
        # diffusers mandatory arguments
        tokenizer: Optional["CLIPTokenizer"],
        scheduler: Optional["SchedulerMixin"],
        unet_session: Optional[ort.InferenceSession],
        vae_decoder_session: Optional[ort.InferenceSession],
        # diffusers optional arguments
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_2_session: Optional[ort.InferenceSession] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        # stable diffusion xl specific arguments
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        # onnxruntime specific arguments
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        if kwargs:
            logger.warning(f"{self.__class__.__name__} received additional arguments that are not used.")

        # mandatory components
        self.unet = ORTModelUnet(unet_session, self, subfolder=DIFFUSION_MODEL_UNET_SUBFOLDER)
        self.vae_decoder = ORTModelVaeDecoder(
            vae_decoder_session, self, subfolder=DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER
        )

        # optional components
        self.vae_encoder = (
            ORTModelVaeEncoder(vae_encoder_session, self, subfolder=DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER)
            if vae_encoder_session is not None
            else None
        )
        self.text_encoder = (
            ORTModelTextEncoder(text_encoder_session, self, subfolder=DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER)
            if text_encoder_session is not None
            else None
        )
        self.text_encoder_2 = (
            ORTModelTextEncoder(text_encoder_2_session, self, subfolder=DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER)
            if text_encoder_2_session is not None
            else None
        )

        # We wrap the VAE encoder and decoder in a single object to simplify the API
        self.vae = ORTWrapperVae(self.vae_encoder, self.vae_decoder)

        self.image_encoder = None  # TODO: maybe implement ORTModelImageEncoder
        self.safety_checker = None  # TODO: maybe implement ORTModelSafetyChecker

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.feature_extractor = feature_extractor

        all_possible_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "image_encoder": self.image_encoder,
            "safety_checker": self.safety_checker,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }

        diffusers_pipeline_args = {}
        for key in inspect.signature(self.auto_model_class).parameters.keys():
            if key in all_possible_init_args:
                diffusers_pipeline_args[key] = all_possible_init_args[key]

        # inits stuff like config, vae_scale_factor, image_processor, etc.
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)

        # not registered correctly in the config
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        self.shared_attributes_init(model=unet_session, use_io_binding=use_io_binding, model_save_dir=model_save_dir)

    @staticmethod
    def load_model(
        unet_path: Union[str, Path],
        vae_encoder_path: Optional[Union[str, Path]] = None,
        vae_decoder_path: Optional[Union[str, Path]] = None,
        text_encoder_path: Optional[Union[str, Path]] = None,
        text_encoder_2_path: Optional[Union[str, Path]] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates three inference sessions for the components of a Diffusion Pipeline (U-NET, VAE, Text Encoders).
        The default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            unet_path (`Union[str, Path]`):
                The path to the U-NET ONNX model.
            vae_encoder_path (`Union[str, Path]`, defaults to `None`):
                The path to the VAE encoder ONNX model.
            vae_decoder_path (`Union[str, Path]`, defaults to `None`):
                The path to the VAE decoder ONNX model.
            text_encoder_path (`Union[str, Path]`, defaults to `None`):
                The path to the text encoder ONNX model.
            text_encoder_2_path (`Union[str, Path]`, defaults to `None`):
                The path to the second text decoder ONNX model.
            provider (`str`, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[ort.SessionOptions]`, defaults to `None`):
                ONNX Runtime session options to use for loading the model. Defaults to `None`.
            provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html . Defaults to `None`.
        """
        paths = {
            "unet": unet_path,
            "vae_encoder": vae_encoder_path,
            "vae_decoder": vae_decoder_path,
            "text_encoder": text_encoder_path,
            "text_encoder_2": text_encoder_2_path,
        }

        sessions = {}
        for model_name, model_path in paths.items():
            if model_path is not None and model_path.is_file():
                sessions[model_name] = ORTModel.load_model(model_path, provider, session_options, provider_options)
            else:
                sessions[model_name] = None

        return sessions

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)

        models_to_save_paths = {
            self.unet: save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.vae_encoder: save_directory / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.vae_decoder: save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.text_encoder: save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.text_encoder_2: save_directory / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / ONNX_WEIGHTS_NAME,
        }
        for model, model_save_path in models_to_save_paths.items():
            if model is not None:
                model_path = Path(model.session._model_path)
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                # copy onnx model
                shutil.copyfile(model_path, model_save_path)
                # copy external onnx data
                external_data_paths = _get_model_external_data_paths(model_path)
                for external_data_path in external_data_paths:
                    shutil.copyfile(external_data_path, model_save_path.parent / external_data_path.name)
                # copy model config
                config_path = model_path.parent / CONFIG_NAME
                if config_path.is_file():
                    config_save_path = model_save_path.parent / CONFIG_NAME
                    shutil.copyfile(config_path, config_save_path)

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
        subfolder: str = "",
        force_download: bool = False,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
        vae_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_2_file_name: str = ONNX_WEIGHTS_NAME,
        use_io_binding: Optional[bool] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        all_components = {key for key in config.keys() if not key.startswith("_")}
        all_components.update({"vae_encoder", "vae_decoder"})

        if not os.path.isdir(str(model_id)):
            allow_patterns = {os.path.join(component, "*") for component in all_components}
            allow_patterns.update(
                {
                    unet_file_name,
                    vae_encoder_file_name,
                    vae_decoder_file_name,
                    text_encoder_file_name,
                    text_encoder_2_file_name,
                    SCHEDULER_CONFIG_NAME,
                    cls.config_name,
                    CONFIG_NAME,
                }
            )
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                revision=revision,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin", "*.xml"],
            )

        model_save_path = Path(model_id)

        sub_models = {}
        for name in {"feature_extractor", "tokenizer", "tokenizer_2", "scheduler"}:
            library_name, library_classes = config.get(name, (None, None))
            if library_classes is not None:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (model_save_path / name).is_dir():
                    sub_models[name] = load_method(model_save_path / name)
                else:
                    sub_models[name] = load_method(model_save_path)

        paths = {
            "unet_path": model_save_path / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
            "vae_encoder_path": model_save_path / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            "vae_decoder_path": model_save_path / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "text_encoder_path": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "text_encoder_2_path": model_save_path
            / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER
            / text_encoder_2_file_name,
        }
        models = cls.load_model(
            **paths, provider=provider, session_options=session_options, provider_options=provider_options
        )

        if use_io_binding:
            raise ValueError(
                "IOBinding is not yet available for stable diffusion model, please set `use_io_binding` to False."
            )

        return cls(
            **models,
            **sub_models,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir or model_save_path,
        )

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: Dict[str, Any],
        subfolder: str = "",
        force_download: bool = False,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        trust_remote_code: bool = False,
        use_io_binding: Optional[bool] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> "ORTPipeline":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        # we continue passing the model_save_dir from here on to avoid it being cleaned up
        # might be better to use a persistent temporary directory such as the one implemented in
        # https://gist.github.com/twolfson/2929dc1163b0a76d2c2b66d51f9bc808
        model_save_dir = TemporaryDirectory()
        model_save_path = Path(model_save_dir.name)

        main_export(
            model_id,
            output=model_save_path,
            do_validation=False,
            no_post_process=True,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
            subfolder=subfolder,
            force_download=force_download,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            library_name="diffusers",
            task=task,
        )

        return cls._from_pretrained(
            model_save_path,
            config=config,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
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

        self.providers = self.unet.session.get_providers()
        self._device = device

        return self

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)

    @property
    def components(self) -> Dict[str, Any]:
        components = {
            "vae": self.vae,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "image_encoder": self.image_encoder,
            "safety_checker": self.safety_checker,
        }
        components = {k: v for k, v in components.items() if v is not None}
        return components

    def __call__(self, *args, **kwargs):
        # we keep numpy random states support for now

        args = list(args)
        for i in range(len(args)):
            args[i] = np_to_pt_generators(args[i], self.device)

        for k, v in kwargs.items():
            kwargs[k] = np_to_pt_generators(v, self.device)

        return self.auto_model_class.__call__(self, *args, **kwargs)


class ORTPipelinePart:
    def __init__(self, session: ort.InferenceSession, parent_pipeline: ORTPipeline, subfolder: str):
        self.session = session
        self.subfolder = subfolder
        self.parent_pipeline = parent_pipeline

        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.input_dtypes = {input_key.name: input_key.type for input_key in session.get_inputs()}
        self.output_dtypes = {output_key.name: output_key.type for output_key in session.get_outputs()}

        self.model_save_dir = Path(self.session._model_path).parent
        config_path = self.model_save_dir / CONFIG_NAME

        if not config_path.is_file():
            # config is necessary for the model to work
            raise ValueError(f"Configuration file for {self.__class__.__name__} not found at {config_path}")

        config_dict = parent_pipeline._dict_from_json_file(config_path)
        self.config = FrozenDict(**config_dict)

    @property
    def device(self):
        return self.parent_pipeline.device

    @property
    def dtype(self):
        for dtype in self.input_dtypes.values():
            torch_dtype = TypeHelper.ort_type_to_torch_type(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = TypeHelper.ort_type_to_torch_type(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    def to(self, *args, device: Optional[Union[torch.device, str, int]] = None, dtype: Optional[torch.dtype] = None):
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, (int, str)):
                device = torch.device(arg)
            elif isinstance(arg, torch.dtype):
                dtype = arg

        if device is not None and device != self.device:
            raise ValueError(
                "Cannot change the device of a pipeline part without changing the device of the parent pipeline. "
                "Please use the `to` method of the parent pipeline to change the device."
            )

        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                f"Cannot change the dtype of the pipeline from {self.dtype} to {dtype}. "
                f"Please export the pipeline with the desired dtype."
            )

    def prepare_onnx_inputs(self, use_torch: bool, **inputs: Union[torch.Tensor, np.ndarray]) -> Dict[str, np.ndarray]:
        onnx_inputs = {}

        # converts pytorch inputs into numpy inputs for onnx
        for input_name in self.input_names.keys():
            onnx_inputs[input_name] = inputs.pop(input_name)

            if use_torch:
                onnx_inputs[input_name] = onnx_inputs[input_name].numpy(force=True)

            if onnx_inputs[input_name].dtype != self.input_dtypes[input_name]:
                onnx_inputs[input_name] = onnx_inputs[input_name].astype(
                    TypeHelper.ort_type_to_numpy_type(self.input_dtypes[input_name])
                )

        return onnx_inputs

    def prepare_onnx_outputs(
        self, use_torch: bool, *onnx_outputs: np.ndarray
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        model_outputs = {}

        # converts onnxruntime outputs into tensor for standard outputs
        for output_name, idx in self.output_names.items():
            model_outputs[output_name] = onnx_outputs[idx]

            if use_torch:
                model_outputs[output_name] = torch.from_numpy(model_outputs[output_name]).to(self.device)

        return model_outputs

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


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

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

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
    # def __init__(self, session: ort.InferenceSession, parent_pipeline: ORTPipeline, subfolder: str):
    #     super().__init__(session, parent_pipeline, subfolder)

    #     if not hasattr(self.config, "time_cond_proj_dim"):
    #         self.config = FrozenDict(**self.config, time_cond_proj_dim=None)

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

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)

    @property
    def add_embedding(self):
        return FrozenDict(
            linear_1=FrozenDict(
                # this is a hacky way to get the attribute in add_embedding.linear_1.in_features
                # (StableDiffusionXLImg2ImgPipeline/StableDiffusionXLInpaintPipeline)._get_add_time_ids
                in_features=self.config.addition_time_embed_dim
                * (
                    5  # list(original_size + crops_coords_top_left + (aesthetic_score,))
                    if self.parent_pipeline.config.requires_aesthetics_score
                    else 6  # list(original_size + crops_coords_top_left + target_size)
                )
                + self.parent_pipeline.text_encoder.config.projection_dim
            )
        )


class ORTModelVaeEncoder(ORTPipelinePart):
    # def __init__(self, session: ort.InferenceSession, parent_pipeline: ORTPipeline, subfolder: str):
    #     super().__init__(session, parent_pipeline, subfolder)

    #     if not hasattr(self.config, "scaling_factor"):
    #         scaling_factor = 2 ** (len(self.config.block_out_channels) - 1)
    #         self.config = FrozenDict(**self.config, scaling_factor=scaling_factor)

    def forward(self, sample: Union[np.ndarray, torch.Tensor], return_dict: bool = False):
        use_torch = isinstance(sample, torch.Tensor)

        model_inputs = {"sample": sample}

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

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
    # def __init__(self, session: ort.InferenceSession, parent_pipeline: ORTPipeline, subfolder: str):
    #     super().__init__(session, parent_pipeline, subfolder)

    #     if not hasattr(self.config, "scaling_factor"):
    #         scaling_factor = 2 ** (len(self.config.block_out_channels) - 1)
    #         self.config = FrozenDict(**self.config, scaling_factor=scaling_factor)

    def forward(
        self,
        latent_sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(latent_sample, torch.Tensor)

        model_inputs = {"latent_sample": latent_sample}

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTWrapperVae(ORTPipelinePart):
    def __init__(self, encoder: ORTModelVaeEncoder, decoder: ORTModelVaeDecoder):
        if encoder is not None:
            self.encoder = encoder

        self.decoder = decoder

    @property
    def config(self):
        return self.decoder.config

    @property
    def dtype(self):
        return self.decoder.dtype

    @property
    def device(self):
        return self.decoder.device

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def to(self, *args, **kwargs):
        if self.encoder is not None:
            self.encoder.to(*args, **kwargs)

        self.decoder.to(*args, **kwargs)


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


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLImg2ImgPipeline(ORTPipeline, StableDiffusionXLImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLImg2ImgPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLInpaintPipeline(ORTPipeline, StableDiffusionXLInpaintPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline).
    """

    main_input_name = "image"
    auto_model_class = StableDiffusionXLInpaintPipeline


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


def _get_ort_class(pipeline_class_name: str, throw_error_if_not_exist: bool = True):
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

        ort_pipeline_class = _get_ort_class(class_name)

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


def _get_task_ort_class(mapping, pipeline_class_name):
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

        ort_pipeline_class = _get_task_ort_class(cls.ort_pipelines_mapping, class_name)

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
