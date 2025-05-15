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
from diffusers.configuration_utils import ConfigMixin
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
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import SchedulerMixin
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils.constants import CONFIG_NAME
from huggingface_hub import HfApi
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.file_utils import add_end_docstrings
from transformers.modeling_outputs import ModelOutput
from transformers.utils import http_user_agent

import onnxruntime as ort
from optimum.utils import is_diffusers_version

from ..exporters.onnx import main_export
from ..onnx.utils import _get_model_external_data_paths
from ..utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
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


if is_diffusers_version(">=", "0.25.0"):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
else:
    from diffusers.models.vae import DiagonalGaussianDistribution  # type: ignore


logger = logging.getLogger(__name__)


# TODO: support from_pipe()
# TODO: Instead of ORTModel, it makes sense to have a compositional ORTMixin
# TODO: instead of one bloated __init__, we should consider an __init__ per pipeline
class ORTDiffusionPipeline(ORTModel, DiffusionPipeline):
    config_name = "model_index.json"
    auto_model_class = DiffusionPipeline

    def __init__(
        self,
        scheduler: "SchedulerMixin",
        vae_decoder_session: ort.InferenceSession,
        # optional pipeline models
        unet_session: Optional[ort.InferenceSession] = None,
        transformer_session: Optional[ort.InferenceSession] = None,
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_2_session: Optional[ort.InferenceSession] = None,
        text_encoder_3_session: Optional[ort.InferenceSession] = None,
        # optional pipeline submodels
        tokenizer: Optional["CLIPTokenizer"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        tokenizer_3: Optional["CLIPTokenizer"] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        # stable diffusion xl specific arguments
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetics_score: bool = False,
        add_watermarker: Optional[bool] = None,
        # onnxruntime specific arguments
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.unet = ORTModelUnet(unet_session, self) if unet_session is not None else None
        self.transformer = ORTModelTransformer(transformer_session, self) if transformer_session is not None else None
        self.text_encoder = (
            ORTModelTextEncoder(text_encoder_session, self) if text_encoder_session is not None else None
        )
        self.text_encoder_2 = (
            ORTModelTextEncoder(text_encoder_2_session, self) if text_encoder_2_session is not None else None
        )
        self.text_encoder_3 = (
            ORTModelTextEncoder(text_encoder_3_session, self) if text_encoder_3_session is not None else None
        )
        # We wrap the VAE Decoder & Encoder in a single object to simulate diffusers API
        self.vae_encoder = ORTModelVaeEncoder(vae_encoder_session, self) if vae_encoder_session is not None else None
        self.vae_decoder = ORTModelVaeDecoder(vae_decoder_session, self) if vae_decoder_session is not None else None
        self.vae = ORTWrapperVae(self.vae_encoder, self.vae_decoder)

        # we allow passing these as torch models for now
        self.image_encoder = kwargs.pop("image_encoder", None)  # TODO: maybe implement ORTModelImageEncoder
        self.safety_checker = kwargs.pop("safety_checker", None)  # TODO: maybe implement ORTModelSafetyChecker

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.feature_extractor = feature_extractor

        all_pipeline_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "tokenizer_3": self.tokenizer_3,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }

        diffusers_pipeline_args = {}
        for key in inspect.signature(self.auto_model_class).parameters.keys():
            if key in all_pipeline_init_args:
                diffusers_pipeline_args[key] = all_pipeline_init_args[key]
        # inits diffusers pipeline specific attributes (registers modules and config)
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)

        # inits ort specific attributes
        self.shared_attributes_init(
            model=unet_session if unet_session is not None else transformer_session,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            **kwargs,
        )

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)

        models_to_save_paths = {
            (self.unet, save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER),
            (self.transformer, save_directory / DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER),
            (self.vae_decoder, save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER),
            (self.vae_encoder, save_directory / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER),
            (self.text_encoder, save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER),
            (self.text_encoder_2, save_directory / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER),
            (self.text_encoder_3, save_directory / DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER),
        }
        for model, save_path in models_to_save_paths:
            if model is not None:
                model_path = Path(model.session._model_path)
                save_path.mkdir(parents=True, exist_ok=True)
                # copy onnx model
                shutil.copyfile(model_path, save_path / ONNX_WEIGHTS_NAME)
                # copy external onnx data
                external_data_paths = _get_model_external_data_paths(model_path)
                for external_data_path in external_data_paths:
                    shutil.copyfile(external_data_path, save_path / external_data_path.name)
                # copy model config
                config_path = model_path.parent / CONFIG_NAME
                if config_path.is_file():
                    config_save_path = save_path / CONFIG_NAME
                    shutil.copyfile(config_path, config_save_path)

        self.scheduler.save_pretrained(save_directory / "scheduler")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory / "tokenizer_2")
        if self.tokenizer_3 is not None:
            self.tokenizer_3.save_pretrained(save_directory / "tokenizer_3")
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
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
        transformer_file_name: str = ONNX_WEIGHTS_NAME,
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        vae_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_2_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_3_file_name: str = ONNX_WEIGHTS_NAME,
        use_io_binding: Optional[bool] = None,
        provider: str = "CPUExecutionProvider",
        provider_options: Optional[Dict[str, Any]] = None,
        session_options: Optional[ort.SessionOptions] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        if use_io_binding:
            raise ValueError(
                "IOBinding is not yet available for diffusion pipelines, please set `use_io_binding` to False."
            )

        if not os.path.isdir(str(model_id)):
            all_components = {key for key in config.keys() if not key.startswith("_")} | {"vae_encoder", "vae_decoder"}
            allow_patterns = {os.path.join(component, "*") for component in all_components}
            allow_patterns.update(
                {
                    unet_file_name,
                    transformer_file_name,
                    vae_decoder_file_name,
                    vae_encoder_file_name,
                    text_encoder_file_name,
                    text_encoder_2_file_name,
                    text_encoder_3_file_name,
                    SCHEDULER_CONFIG_NAME,
                    cls.config_name,
                    CONFIG_NAME,
                }
            )
            model_save_folder = HfApi(user_agent=http_user_agent(), token=token).snapshot_download(
                model_id,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin", "*.xml"],
                allow_patterns=allow_patterns,
            )
        else:
            model_save_folder = str(model_id)

        model_save_path = Path(model_save_folder)

        if model_save_dir is None:
            model_save_dir = model_save_path

        model_paths = {
            "unet": model_save_path / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
            "transformer": model_save_path / DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER / transformer_file_name,
            "vae_decoder": model_save_path / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "vae_encoder": model_save_path / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            "text_encoder": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "text_encoder_2": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name,
            "text_encoder_3": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER / text_encoder_3_file_name,
        }

        sessions = {}
        for model, path in model_paths.items():
            if kwargs.get(model, None) is not None:
                # this allows passing a model directly to from_pretrained
                sessions[f"{model}_session"] = kwargs.pop(model)
            else:
                sessions[f"{model}_session"] = (
                    ORTModel.load_model(path, provider, session_options, provider_options) if path.is_file() else None
                )

        submodels = {}
        for submodel in {"scheduler", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor"}:
            if kwargs.get(submodel, None) is not None:
                submodels[submodel] = kwargs.pop(submodel)
            elif config.get(submodel, (None, None))[0] is not None:
                library_name, library_classes = config.get(submodel)
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (model_save_path / submodel).is_dir():
                    submodels[submodel] = load_method(model_save_path / submodel)
                else:
                    submodels[submodel] = load_method(model_save_path)

        # same as DiffusionPipeline.from_pretraoned, if called directly, it loads the class in the config
        if cls.__name__ == "ORTDiffusionPipeline":
            class_name = config["_class_name"]
            ort_pipeline_class = _get_ort_class(class_name)
        else:
            ort_pipeline_class = cls

        ort_pipeline = ort_pipeline_class(
            **sessions,
            **submodels,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            **kwargs,
        )

        # same as in DiffusionPipeline.from_pretrained, we save where the model was instantiated from
        ort_pipeline.register_to_config(_name_or_path=config.get("_name_or_path", str(model_id)))

        return ort_pipeline

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: Dict[str, Any],
        subfolder: str = "",
        force_download: bool = False,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        use_io_binding: Optional[bool] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> "ORTDiffusionPipeline":
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
            provider_options=provider_options,
            session_options=session_options,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            **kwargs,
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
        validate_provider_availability(provider)

        if device.type == "cuda" and self.providers[0] == "TensorrtExecutionProvider":
            return self

        self.vae_decoder.session.set_providers([provider], provider_options=[provider_options])

        if self.unet is not None:
            self.unet.session.set_providers([provider], provider_options=[provider_options])
        if self.transformer is not None:
            self.transformer.session.set_providers([provider], provider_options=[provider_options])
        if self.vae_encoder is not None:
            self.vae_encoder.session.set_providers([provider], provider_options=[provider_options])
        if self.text_encoder is not None:
            self.text_encoder.session.set_providers([provider], provider_options=[provider_options])
        if self.text_encoder_2 is not None:
            self.text_encoder_2.session.set_providers([provider], provider_options=[provider_options])
        if self.text_encoder_3 is not None:
            self.text_encoder_3.session.set_providers([provider], provider_options=[provider_options])

        self.providers = (
            self.unet.session.get_providers() if self.unet is not None else self.transformer.session.get_providers()
        )
        self._device = device

        return self

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory: Union[str, Path]):
        model_dir = (
            self.model_save_dir
            if not isinstance(self.model_save_dir, TemporaryDirectory)
            else self.model_save_dir.name
        )
        save_dir = Path(save_directory)
        original_config = Path(model_dir) / self.config_name
        if original_config.exists():
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            shutil.copy(original_config, save_dir)
        else:
            self.save_config(save_directory)

    @property
    def components(self) -> Dict[str, Any]:
        components = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
        }
        components = {k: v for k, v in components.items() if v is not None}
        return components

    def __call__(self, *args, **kwargs):
        # we do this to keep numpy random states support for now
        # TODO: deprecate and add warnings when a random state is passed

        args = list(args)
        for i in range(len(args)):
            args[i] = np_to_pt_generators(args[i], self.device)

        for k, v in kwargs.items():
            kwargs[k] = np_to_pt_generators(v, self.device)

        return self.auto_model_class.__call__(self, *args, **kwargs)


class ORTPipelinePart(ConfigMixin):
    config_name: str = CONFIG_NAME

    def __init__(self, session: ort.InferenceSession, parent_pipeline: ORTDiffusionPipeline):
        self.session = session
        self.parent_pipeline = parent_pipeline

        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

        self.input_dtypes = {input_key.name: input_key.type for input_key in self.session.get_inputs()}
        self.output_dtypes = {output_key.name: output_key.type for output_key in self.session.get_outputs()}

        self.input_shapes = {input_key.name: input_key.shape for input_key in self.session.get_inputs()}
        self.output_shapes = {output_key.name: output_key.shape for output_key in self.session.get_outputs()}

        config_file_path = Path(session._model_path).parent / self.config_name
        if not config_file_path.is_file():
            # config is mandatory for the model part to be used for inference
            raise ValueError(f"Configuration file for {self.__class__.__name__} not found at {config_file_path}")
        config_dict = self._dict_from_json_file(config_file_path)
        self.register_to_config(**config_dict)

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


class ORTModelUnet(ORTPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # can be missing from models exported long ago
        if not hasattr(self.config, "time_cond_proj_dim"):
            logger.warning(
                "The `time_cond_proj_dim` attribute is missing from the UNet configuration. "
                "Please re-export the model with newer version of optimum and diffusers."
            )
            self.register_to_config(time_cond_proj_dim=None)

        if len(self.input_shapes["timestep"]) > 0:
            logger.warning(
                "The exported unet onnx model expects a non scalar timestep input. "
                "We will have to unsqueeze the timestep input at each iteration which might be inefficient. "
                "Please re-export the pipeline with newer version of optimum and diffusers to avoid this warning."
            )

    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        timestep: Union[np.ndarray, torch.Tensor],
        encoder_hidden_states: Union[np.ndarray, torch.Tensor],
        timestep_cond: Optional[Union[np.ndarray, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(sample, torch.Tensor)

        if len(self.input_shapes["timestep"]) > 0:
            timestep = timestep.unsqueeze(0)

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep_cond": timestep_cond,
            **(cross_attention_kwargs or {}),
            **(added_cond_kwargs or {}),
        }

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if not return_dict:
            return tuple(model_outputs.values())

        return ModelOutput(**model_outputs)


class ORTModelTransformer(ORTPipelinePart):
    def forward(
        self,
        hidden_states: Union[np.ndarray, torch.Tensor],
        encoder_hidden_states: Union[np.ndarray, torch.Tensor],
        pooled_projections: Union[np.ndarray, torch.Tensor],
        timestep: Union[np.ndarray, torch.Tensor],
        guidance: Optional[Union[np.ndarray, torch.Tensor]] = None,
        txt_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        img_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(hidden_states, torch.Tensor)

        model_inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "guidance": guidance,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            **(joint_attention_kwargs or {}),
        }

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if not return_dict:
            return tuple(model_outputs.values())

        return ModelOutput(**model_outputs)


class ORTModelTextEncoder(ORTPipelinePart):
    def forward(
        self,
        input_ids: Union[np.ndarray, torch.Tensor],
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)

        model_inputs = {"input_ids": input_ids}

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if output_hidden_states:
            model_outputs["hidden_states"] = []
            num_layers = self.num_hidden_layers if hasattr(self, "num_hidden_layers") else self.num_decoder_layers
            for i in range(num_layers):
                model_outputs["hidden_states"].append(model_outputs.pop(f"hidden_states.{i}"))
            model_outputs["hidden_states"].append(model_outputs.get("last_hidden_state"))
        else:
            num_layers = self.num_hidden_layers if hasattr(self, "num_hidden_layers") else self.num_decoder_layers
            for i in range(num_layers):
                model_outputs.pop(f"hidden_states.{i}", None)

        if not return_dict:
            return tuple(model_outputs.values())

        return ModelOutput(**model_outputs)


class ORTModelVaeEncoder(ORTPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # can be missing from models exported long ago
        if not hasattr(self.config, "scaling_factor"):
            logger.warning(
                "The `scaling_factor` attribute is missing from the VAE encoder configuration. "
                "Please re-export the model with newer version of optimum and diffusers to avoid this warning."
            )
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
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

        if not return_dict:
            return tuple(model_outputs.values())

        return ModelOutput(**model_outputs)


class ORTModelVaeDecoder(ORTPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # can be missing from models exported long ago
        if not hasattr(self.config, "scaling_factor"):
            logger.warning(
                "The `scaling_factor` attribute is missing from the VAE decoder configuration. "
                "Please re-export the model with newer version of optimum and diffusers to avoid this warning."
            )
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        latent_sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(latent_sample, torch.Tensor)

        model_inputs = {"latent_sample": latent_sample}

        onnx_inputs = self.prepare_onnx_inputs(use_torch, **model_inputs)
        onnx_outputs = self.session.run(None, onnx_inputs)
        model_outputs = self.prepare_onnx_outputs(use_torch, *onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if not return_dict:
            return tuple(model_outputs.values())

        return ModelOutput(**model_outputs)


class ORTWrapperVae(ORTPipelinePart):
    def __init__(self, encoder: ORTModelVaeEncoder, decoder: ORTModelVaeDecoder):
        self.decoder = decoder
        self.encoder = encoder

    @property
    def config(self):
        return self.decoder.config

    @property
    def dtype(self):
        return self.decoder.dtype

    @property
    def device(self):
        return self.decoder.device

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.decoder.to(*args, **kwargs)
        if self.encoder is not None:
            self.encoder.to(*args, **kwargs)


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionPipeline(ORTDiffusionPipeline, StableDiffusionPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusionPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionImg2ImgPipeline(ORTDiffusionPipeline, StableDiffusionImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = StableDiffusionImg2ImgPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionInpaintPipeline(ORTDiffusionPipeline, StableDiffusionInpaintPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    main_input_name = "prompt"
    export_feature = "inpainting"
    auto_model_class = StableDiffusionInpaintPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLPipeline(ORTDiffusionPipeline, StableDiffusionXLPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusionXLPipeline

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTStableDiffusionXLImg2ImgPipeline(ORTDiffusionPipeline, StableDiffusionXLImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    main_input_name = "prompt"
    export_feature = "image-to-image"
    auto_model_class = StableDiffusionXLImg2ImgPipeline

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
class ORTStableDiffusionXLInpaintPipeline(ORTDiffusionPipeline, StableDiffusionXLInpaintPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline).
    """

    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = StableDiffusionXLInpaintPipeline

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
class ORTLatentConsistencyModelPipeline(ORTDiffusionPipeline, LatentConsistencyModelPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = LatentConsistencyModelPipeline


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTLatentConsistencyModelImg2ImgPipeline(ORTDiffusionPipeline, LatentConsistencyModelImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency_img2img#diffusers.LatentConsistencyModelImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = LatentConsistencyModelImg2ImgPipeline


class ORTUnavailablePipeline:
    MIN_VERSION = None

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"The pipeline {self.__class__.__name__} is not available in the current version of `diffusers`. "
            f"Please upgrade `diffusers` to {self.MIN_VERSION} or later."
        )


if is_diffusers_version(">=", "0.29.0"):
    from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline

    @add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
    class ORTStableDiffusion3Pipeline(ORTDiffusionPipeline, StableDiffusion3Pipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusion3Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusion3Pipeline).
        """

        main_input_name = "prompt"
        export_feature = "text-to-image"
        auto_model_class = StableDiffusion3Pipeline

    @add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
    class ORTStableDiffusion3Img2ImgPipeline(ORTDiffusionPipeline, StableDiffusion3Img2ImgPipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusion3Img2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusion3Img2ImgPipeline).
        """

        main_input_name = "image"
        export_feature = "image-to-image"
        auto_model_class = StableDiffusion3Img2ImgPipeline

else:

    class ORTStableDiffusion3Pipeline(ORTUnavailablePipeline):
        MIN_VERSION = "0.29.0"

    class ORTStableDiffusion3Img2ImgPipeline(ORTUnavailablePipeline):
        MIN_VERSION = "0.29.0"


if is_diffusers_version(">=", "0.30.0"):
    from diffusers import FluxPipeline, StableDiffusion3InpaintPipeline

    @add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
    class ORTStableDiffusion3InpaintPipeline(ORTDiffusionPipeline, StableDiffusion3InpaintPipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusion3InpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusion3InpaintPipeline).
        """

        main_input_name = "prompt"
        export_feature = "inpainting"
        auto_model_class = StableDiffusion3InpaintPipeline

    @add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
    class ORTFluxPipeline(ORTDiffusionPipeline, FluxPipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.FluxPipeline](https://huggingface.co/docs/diffusers/api/pipelines/flux/text2img#diffusers.FluxPipeline).
        """

        main_input_name = "prompt"
        export_feature = "text-to-image"
        auto_model_class = FluxPipeline

else:

    class ORTStableDiffusion3InpaintPipeline(ORTUnavailablePipeline):
        MIN_VERSION = "0.30.0"

    class ORTFluxPipeline(ORTUnavailablePipeline):
        MIN_VERSION = "0.30.0"


SUPPORTED_ORT_PIPELINES = [
    ORTStableDiffusionPipeline,
    ORTStableDiffusionImg2ImgPipeline,
    ORTStableDiffusionInpaintPipeline,
    ORTStableDiffusionXLPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
    ORTStableDiffusionXLInpaintPipeline,
    ORTLatentConsistencyModelPipeline,
    ORTLatentConsistencyModelImg2ImgPipeline,
    ORTStableDiffusion3Pipeline,
    ORTStableDiffusion3Img2ImgPipeline,
    ORTStableDiffusion3InpaintPipeline,
    ORTFluxPipeline,
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


ORT_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("flux", ORTFluxPipeline),
        ("latent-consistency", ORTLatentConsistencyModelPipeline),
        ("stable-diffusion", ORTStableDiffusionPipeline),
        ("stable-diffusion-3", ORTStableDiffusion3Pipeline),
        ("stable-diffusion-xl", ORTStableDiffusionXLPipeline),
    ]
)

ORT_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("latent-consistency", ORTLatentConsistencyModelImg2ImgPipeline),
        ("stable-diffusion", ORTStableDiffusionImg2ImgPipeline),
        ("stable-diffusion-3", ORTStableDiffusion3Img2ImgPipeline),
        ("stable-diffusion-xl", ORTStableDiffusionXLImg2ImgPipeline),
    ]
)

ORT_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", ORTStableDiffusionInpaintPipeline),
        ("stable-diffusion-3", ORTStableDiffusion3InpaintPipeline),
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
    def from_pretrained(cls, pretrained_model_or_path, **kwargs) -> ORTDiffusionPipeline:
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
