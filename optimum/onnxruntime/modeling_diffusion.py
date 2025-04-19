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
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.file_utils import add_end_docstrings
from transformers.modeling_outputs import ModelOutput
from transformers.utils import http_user_agent

from onnxruntime import InferenceSession, SessionOptions

from ..exporters.onnx import main_export
from ..onnx.utils import _get_model_external_data_paths
from ..utils import (
    DIFFUSION_MODEL_ONNX_FILE_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
    DIFFUSION_PIPELINE_CONFIG_FILE_NAME,
    is_diffusers_version,
)
from .base import ORTSessionMixin, ORTSessionsWrapper
from .utils import np_to_pt_generators


if is_diffusers_version(">=", "0.25.0"):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
else:
    from diffusers.models.vae import DiagonalGaussianDistribution  # type: ignore


logger = logging.getLogger(__name__)


# TODO: support from_pipe()
class ORTDiffusionPipeline(ORTSessionsWrapper, DiffusionPipeline):
    config_name = DIFFUSION_PIPELINE_CONFIG_FILE_NAME

    task = "auto"
    library = "diffusers"
    auto_model_class = DiffusionPipeline

    def __init__(
        self,
        scheduler: "SchedulerMixin",
        # optional pipeline models
        unet_session: Optional[InferenceSession] = None,
        transformer_session: Optional[InferenceSession] = None,
        vae_decoder_session: Optional[InferenceSession] = None,
        vae_encoder_session: Optional[InferenceSession] = None,
        text_encoder_session: Optional[InferenceSession] = None,
        text_encoder_2_session: Optional[InferenceSession] = None,
        text_encoder_3_session: Optional[InferenceSession] = None,
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
        # We initialize all ort session mixins first
        self.unet = ORTUnet(unet_session, use_io_binding) if unet_session is not None else None
        self.transformer = (
            ORTTransformer(transformer_session, use_io_binding) if transformer_session is not None else None
        )
        self.text_encoder = (
            ORTTextEncoder(text_encoder_session, use_io_binding) if text_encoder_session is not None else None
        )
        self.text_encoder_2 = (
            ORTTextEncoder(text_encoder_2_session, use_io_binding) if text_encoder_2_session is not None else None
        )
        self.text_encoder_3 = (
            ORTTextEncoder(text_encoder_3_session, use_io_binding) if text_encoder_3_session is not None else None
        )
        self.vae_encoder = (
            ORTVaeEncoder(vae_encoder_session, use_io_binding) if vae_encoder_session is not None else None
        )
        self.vae_decoder = (
            ORTVaeDecoder(vae_decoder_session, use_io_binding) if vae_decoder_session is not None else None
        )

        # We register ort session mixins to the wrapper
        ort_session_mixins = {
            self.unet,
            self.transformer,
            self.vae_encoder,
            self.vae_decoder,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
        }
        super().__init__(sessions=list(filter(lambda x: x is not None, ort_session_mixins)))

        # We wrap the VAE Encoder & Decoder in a single object for convenience
        self.vae = (
            ORTVae(self.vae_encoder, self.vae_decoder)
            if self.vae_encoder is not None or self.vae_decoder is not None
            else None
        )

        # we allow passing these as torch models for now
        self.image_encoder = kwargs.pop("image_encoder", None)  # TODO: maybe implement ORTImageEncoder
        self.safety_checker = kwargs.pop("safety_checker", None)  # TODO: maybe implement ORTSafetyChecker

        # We register the submodels to the pipeline
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.feature_extractor = feature_extractor

        # We initialize diffusers pipeline specific attributes (registers modules and config)
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
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying ONNX model (and thus failing inference).
        self.model_save_dir = model_save_dir

    @property
    def components(self) -> Dict[str, Optional[Union[ORTSessionMixin, torch.nn.Module]]]:
        # TODO: all compoenents should be ORTSessionMixin's at some point
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

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTParentMixin`: The updated ORT model.
        """

        for component in self.components.values():
            if isinstance(component, (ORTSessionMixin, ORTSessionsWrapper)):
                component.to(device)

        return self

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        # export related arguments
        export: bool = False,
        # load related arguments
        unet_file_name_or_path: Optional[Union[str, Path]] = None,
        transformer_file_name_or_path: Optional[Union[str, Path]] = None,
        vae_encoder_file_name_or_path: Optional[Union[str, Path]] = None,
        vae_decoder_file_name_or_path: Optional[Union[str, Path]] = None,
        text_encoder_file_name_or_path: Optional[Union[str, Path]] = None,
        text_encoder_2_file_name_or_path: Optional[Union[str, Path]] = None,
        text_encoder_3_file_name_or_path: Optional[Union[str, Path]] = None,
        # inference related arguments
        use_io_binding: Optional[bool] = None,
        provider: str = "CPUExecutionProvider",
        provider_option: Optional[Dict[str, Any]] = None,
        session_options: Optional[SessionOptions] = None,
        # hub related arguments
        **kwargs,
    ):
        hub_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download", None),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir", None),
            "revision": kwargs.get("revision", None),
            "proxies": kwargs.get("proxies", None),
            "token": kwargs.get("token", None),
        }

        # get the pipeline config
        config = cls.load_config(model_name_or_path, **hub_kwargs)
        config = config[0] if isinstance(config, tuple) else config

        model_save_tmpdir = None
        model_save_path = Path(model_name_or_path)

        # export the model if requested
        if export:
            model_save_tmpdir = TemporaryDirectory()
            model_save_path = Path(model_save_tmpdir.name)
            main_export(
                model_name_or_path=model_name_or_path,
                # export related arguments
                output=model_save_path,
                no_post_process=True,
                do_validation=False,
                task=cls.task,
                library_name=cls.library,
                # hub related arguments
                **hub_kwargs,
            )

        # download the model if needed
        if not model_save_path.is_dir():
            # everything in components subfolders
            all_components = {key for key in config.keys() if not key.startswith("_")} | {"vae_encoder", "vae_decoder"}
            allow_patterns = {os.path.join(component, "*") for component in all_components}
            # plus custom file names
            allow_patterns.update(
                {
                    DIFFUSION_PIPELINE_CONFIG_FILE_NAME,
                    DIFFUSION_MODEL_ONNX_FILE_NAME,
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                }
            )
            model_save_folder = HfApi(user_agent=http_user_agent()).snapshot_download(
                repo_id=str(model_name_or_path),
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin", "*.xml"],
                **hub_kwargs,
            )
            model_save_path = Path(model_save_folder)

        # onnx files to load
        if unet_file_name_or_path is None:
            unet_file_name_or_path = model_save_path / DIFFUSION_MODEL_UNET_SUBFOLDER / DIFFUSION_MODEL_ONNX_FILE_NAME
        if transformer_file_name_or_path is None:
            transformer_file_name_or_path = (
                model_save_path / DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER / DIFFUSION_MODEL_ONNX_FILE_NAME
            )
        if vae_encoder_file_name_or_path is None:
            vae_encoder_file_name_or_path = (
                model_save_path / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / DIFFUSION_MODEL_ONNX_FILE_NAME
            )
        if vae_decoder_file_name_or_path is None:
            vae_decoder_file_name_or_path = (
                model_save_path / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / DIFFUSION_MODEL_ONNX_FILE_NAME
            )
        if text_encoder_file_name_or_path is None:
            text_encoder_file_name_or_path = (
                model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / DIFFUSION_MODEL_ONNX_FILE_NAME
            )
        if text_encoder_2_file_name_or_path is None:
            text_encoder_2_file_name_or_path = (
                model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / DIFFUSION_MODEL_ONNX_FILE_NAME
            )
        if text_encoder_3_file_name_or_path is None:
            text_encoder_3_file_name_or_path = (
                model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER / DIFFUSION_MODEL_ONNX_FILE_NAME
            )

        model_paths = {
            "unet": unet_file_name_or_path,
            "transformer": transformer_file_name_or_path,
            "vae_encoder": vae_encoder_file_name_or_path,
            "vae_decoder": vae_decoder_file_name_or_path,
            "text_encoder": text_encoder_file_name_or_path,
            "text_encoder_2": text_encoder_2_file_name_or_path,
            "text_encoder_3": text_encoder_3_file_name_or_path,
        }

        sessions = {}
        for model, path in model_paths.items():
            if kwargs.get(model, None) is not None:
                # this allows passing a model directly to from_pretrained
                sessions[f"{model}_session"] = kwargs.pop(model)
            elif kwargs.get(f"{model}_session", None) is not None:
                # this allows passing a session directly to from_pretrained
                sessions[f"{model}_session"] = kwargs.pop(f"{model}_session")
            else:
                sessions[f"{model}_session"] = (
                    ORTSessionMixin.load_model(path, provider, provider_option, session_options)
                    if path.is_file()
                    else None
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

        # same as DiffusionPipeline.from_pretrained,
        # if ORTDiffusionPipeline is called directly,
        # it loads (the ort equivalent of) the class name in the config
        if cls.__name__ == "ORTDiffusionPipeline":
            class_name = config["_class_name"]
            ort_pipeline_class = _get_ort_class(class_name)
        else:
            ort_pipeline_class = cls

        ort_pipeline = ort_pipeline_class(
            **sessions,
            **submodels,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_tmpdir,
            **kwargs,
        )

        # same as DiffusionPipeline.from_pretrained,
        # we save where the model was instantiated from
        ort_pipeline.register_to_config(_name_or_path=config.get("_name_or_path", model_name_or_path))

        return ort_pipeline

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        push_to_hub: Optional[bool] = False,
        **kwargs,
    ):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.

        Args:
            save_directory (`Union[str, os.PathLike]`):
                Directory to which to save. Will be created if it doesn't exist.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

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
                shutil.copyfile(model_path, save_path / DIFFUSION_MODEL_ONNX_FILE_NAME)
                # copy external onnx data if any
                external_data_paths = _get_model_external_data_paths(model_path)
                for external_data_path in external_data_paths:
                    if external_data_path.is_file():
                        external_data_save_path = save_path / external_data_path.name
                        shutil.copyfile(external_data_path, external_data_save_path)
                # copy model config if any
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

        # finally save the config
        self.save_config(save_directory)

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token, is_pipeline=True)
            model_card = populate_model_card(model_card)
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    def __call__(self, *args, **kwargs):
        # we do this to keep numpy random states support for now

        args = list(args)
        for i in range(len(args)):
            new_args = np_to_pt_generators(args[i], self.device)
            if args[i] is not new_args:
                logger.warning(
                    "Converting numpy random state to torch generator is deprecated. "
                    "Please pass a torch generator directly to the pipeline."
                )

        for key, value in kwargs.items():
            new_value = np_to_pt_generators(value, self.device)
            if value is not new_value:
                logger.warning(
                    "Converting numpy random state to torch generator is deprecated. "
                    "Please pass a torch generator directly to the pipeline."
                )
                kwargs[key] = new_value

        return self.auto_model_class.__call__(self, *args, **kwargs)


class ORTModelMixin(ORTSessionMixin, ConfigMixin):
    config_name: str = CONFIG_NAME

    def __init__(self, session: InferenceSession, use_io_binding: Optional[bool] = None):
        super().__init__(session, use_io_binding=use_io_binding)

        config_file_path = Path(session._model_path).parent / self.config_name
        if not config_file_path.is_file():
            # config is mandatory for the model part to be used for inference
            raise ValueError(f"Configuration file for {self.__class__.__name__} not found at {config_file_path}")
        config_dict = self._dict_from_json_file(config_file_path)
        self.register_to_config(**config_dict)


class ORTUnet(ORTModelMixin):
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
        return_dict: bool = False,
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

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTTransformer(ORTModelMixin):
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
        return_dict: bool = False,
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

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTTextEncoder(ORTModelMixin):
    def forward(
        self,
        input_ids: Union[np.ndarray, torch.Tensor],
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = False,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)

        model_inputs = {
            "input_ids": input_ids,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

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

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTVaeEncoder(ORTModelMixin):
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
        return_dict: bool = False,
    ):
        use_torch = isinstance(sample, torch.Tensor)

        model_inputs = {
            "sample": sample,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if "latent_parameters" in model_outputs:
            model_outputs["latent_dist"] = DiagonalGaussianDistribution(
                parameters=model_outputs.pop("latent_parameters")
            )

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTVaeDecoder(ORTModelMixin):
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
        return_dict: bool = False,
    ):
        use_torch = isinstance(latent_sample, torch.Tensor)

        model_inputs = {
            "latent_sample": latent_sample,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class ORTVae(ORTSessionsWrapper):
    def __init__(self, encoder: Optional[ORTVaeEncoder] = None, decoder: Optional[ORTVaeDecoder] = None):
        super().__init__(sessions=list(filter(lambda x: x is not None, [encoder, decoder])))
        self.encoder, self.decoder = encoder, decoder

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @property
    def config(self):
        return self.decoder.config


ORT_PIPELINE_DOCSTRING = r"""
    This Pipeline inherits from [`ORTDiffusionPipeline`] and is used to run inference with the ONNX Runtime.
    The pipeline can be loaded from a pretrained pipeline using the [`ORTDiffusionPipeline.from_pretrained`] method.
"""


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTStableDiffusionPipeline(ORTDiffusionPipeline, StableDiffusionPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline).
    """

    task = "text-to-image"
    main_input_name = "prompt"
    auto_model_class = StableDiffusionPipeline


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTStableDiffusionImg2ImgPipeline(ORTDiffusionPipeline, StableDiffusionImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    task = "image-to-image"
    main_input_name = "image"
    auto_model_class = StableDiffusionImg2ImgPipeline


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTStableDiffusionInpaintPipeline(ORTDiffusionPipeline, StableDiffusionInpaintPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    task = "inpainting"
    main_input_name = "prompt"
    auto_model_class = StableDiffusionInpaintPipeline


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTStableDiffusionXLPipeline(ORTDiffusionPipeline, StableDiffusionXLPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline).
    """

    task = "text-to-image"
    main_input_name = "prompt"
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


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTStableDiffusionXLImg2ImgPipeline(ORTDiffusionPipeline, StableDiffusionXLImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    task = "image-to-image"
    main_input_name = "prompt"
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


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTStableDiffusionXLInpaintPipeline(ORTDiffusionPipeline, StableDiffusionXLInpaintPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline).
    """

    main_input_name = "image"
    task = "inpainting"
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


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTLatentConsistencyModelPipeline(ORTDiffusionPipeline, LatentConsistencyModelPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    task = "text-to-image"
    main_input_name = "prompt"
    auto_model_class = LatentConsistencyModelPipeline


@add_end_docstrings(ORT_PIPELINE_DOCSTRING)
class ORTLatentConsistencyModelImg2ImgPipeline(ORTDiffusionPipeline, LatentConsistencyModelImg2ImgPipeline):
    """
    ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency_img2img#diffusers.LatentConsistencyModelImg2ImgPipeline).
    """

    task = "image-to-image"
    main_input_name = "image"
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

    @add_end_docstrings(ORT_PIPELINE_DOCSTRING)
    class ORTStableDiffusion3Pipeline(ORTDiffusionPipeline, StableDiffusion3Pipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusion3Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusion3Pipeline).
        """

        task = "text-to-image"
        main_input_name = "prompt"
        auto_model_class = StableDiffusion3Pipeline

    @add_end_docstrings(ORT_PIPELINE_DOCSTRING)
    class ORTStableDiffusion3Img2ImgPipeline(ORTDiffusionPipeline, StableDiffusion3Img2ImgPipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusion3Img2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusion3Img2ImgPipeline).
        """

        task = "image-to-image"
        main_input_name = "image"
        auto_model_class = StableDiffusion3Img2ImgPipeline

else:

    class ORTStableDiffusion3Pipeline(ORTUnavailablePipeline):
        MIN_VERSION = "0.29.0"

    class ORTStableDiffusion3Img2ImgPipeline(ORTUnavailablePipeline):
        MIN_VERSION = "0.29.0"


if is_diffusers_version(">=", "0.30.0"):
    from diffusers import FluxPipeline, StableDiffusion3InpaintPipeline

    @add_end_docstrings(ORT_PIPELINE_DOCSTRING)
    class ORTStableDiffusion3InpaintPipeline(ORTDiffusionPipeline, StableDiffusion3InpaintPipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusion3InpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusion3InpaintPipeline).
        """

        task = "inpainting"
        main_input_name = "prompt"
        auto_model_class = StableDiffusion3InpaintPipeline

    @add_end_docstrings(ORT_PIPELINE_DOCSTRING)
    class ORTFluxPipeline(ORTDiffusionPipeline, FluxPipeline):
        """
        ONNX Runtime-powered stable diffusion pipeline corresponding to [diffusers.FluxPipeline](https://huggingface.co/docs/diffusers/api/pipelines/flux/text2img#diffusers.FluxPipeline).
        """

        task = "text-to-image"
        main_input_name = "prompt"
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
