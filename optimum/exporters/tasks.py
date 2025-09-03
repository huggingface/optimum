# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model export tasks manager."""

import importlib
import os
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from huggingface_hub import HfApi
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.errors import OfflineModeIsEnabled
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, http_user_agent

from ..utils.import_utils import is_diffusers_available
from ..utils.logging import get_logger


if TYPE_CHECKING:
    from .base import ExporterConfig

    if is_torch_available():
        from transformers import PreTrainedModel
    elif is_tf_available():
        from transformers import TFPreTrainedModel

logger = get_logger(__name__)

if not is_torch_available() and not is_tf_available():
    logger.warning(
        "The export tasks are only supported for PyTorch or TensorFlow. You will not be able to export models"
        " without one of these libraries installed."
    )

if is_torch_available():
    import torch


if is_diffusers_available():
    from diffusers import DiffusionPipeline
    from diffusers.pipelines.auto_pipeline import (
        AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
        AUTO_INPAINT_PIPELINES_MAPPING,
        AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    )

ExportConfigConstructor = Callable[[PretrainedConfig], "ExporterConfig"]
TaskNameToExportConfigDict = Dict[str, ExportConfigConstructor]


def make_backend_config_constructor_for_task(config_cls: Type, task: str) -> ExportConfigConstructor:
    if "-with-past" in task:
        if not getattr(config_cls, "SUPPORTS_PAST", False):
            raise ValueError(f"{config_cls} does not support tasks with past.")
        constructor = partial(config_cls, use_past=True, task=task.replace("-with-past", ""))
    else:
        constructor = partial(config_cls, task=task)
    return constructor


def get_diffusers_tasks_to_model_mapping():
    """task -> model type -> model class mapping"""

    tasks_to_model_mapping = {}

    for task_name, model_mapping in (
        ("text-to-image", AUTO_TEXT2IMAGE_PIPELINES_MAPPING),
        ("image-to-image", AUTO_IMAGE2IMAGE_PIPELINES_MAPPING),
        ("inpainting", AUTO_INPAINT_PIPELINES_MAPPING),
    ):
        tasks_to_model_mapping[task_name] = {}

        for model_type, model_class in model_mapping.items():
            tasks_to_model_mapping[task_name][model_type] = model_class.__name__

    return tasks_to_model_mapping


def get_transformers_tasks_to_model_mapping(tasks_to_model_loader, framework="pt"):
    """task -> model type -> model class mapping"""
    if framework == "pt":
        auto_modeling_module = importlib.import_module("transformers.models.auto.modeling_auto")
    elif framework == "tf":
        auto_modeling_module = importlib.import_module("transformers.models.auto.modeling_tf_auto")

    tasks_to_model_mapping = {}
    for task_name, model_loaders in tasks_to_model_loader.items():
        if isinstance(model_loaders, str):
            model_loaders = (model_loaders,)

        tasks_to_model_mapping[task_name] = {}
        for model_loader in model_loaders:
            model_loader_class = getattr(auto_modeling_module, model_loader, None)
            if model_loader_class is not None:
                # we can just update the model_type to model_class mapping since
                # we can only have one task->model_type->model_class mapping either way
                # e.g. we merge automatic-speech-recognition's SpeechSeq2Seq and CTC models without worrying
                tasks_to_model_mapping[task_name].update(model_loader_class._model_mapping._model_mapping)

    return tasks_to_model_mapping


class TasksManager:
    """
    Handles the `task name -> model class` and `architecture -> configuration` mappings.
    """

    # Torch model loaders
    _TRANSFORMERS_TASKS_TO_MODEL_LOADERS = {}
    _DIFFUSERS_TASKS_TO_MODEL_LOADERS = {}
    _TIMM_TASKS_TO_MODEL_LOADERS = {}
    _LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP = {}

    # Torch model mappings
    _TRANSFORMERS_TASKS_TO_MODEL_MAPPINGS = {}
    _DIFFUSERS_TASKS_TO_MODEL_MAPPINGS = {}

    # TF model loaders
    _TRANSFORMERS_TASKS_TO_TF_MODEL_LOADERS = {}
    _LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP = {}

    # TF model mappings
    _TRANSFORMERS_TASKS_TO_MODEL_MAPPINGS = {}

    if is_torch_available():
        # Refer to https://huggingface.co/datasets/huggingface/transformers-metadata/blob/main/pipeline_tags.json
        # In case the same task (pipeline tag) may map to several loading classes, we use a tuple and the
        # auto-class _model_mapping to determine the right one.

        # TODO: having several tasks pointing to the same auto-model class is bug prone to auto-detect the
        # task in a Hub repo that has no pipeline_tag, and no transformersInfo.pipeline_tag, as we then rely on
        # on transformersInfo["auto_model"] and this dictionary.
        _TRANSFORMERS_TASKS_TO_MODEL_LOADERS = {
            "audio-classification": "AutoModelForAudioClassification",
            "audio-frame-classification": "AutoModelForAudioFrameClassification",
            "audio-xvector": "AutoModelForAudioXVector",
            "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", "AutoModelForCTC"),
            "depth-estimation": "AutoModelForDepthEstimation",
            "feature-extraction": "AutoModel",
            "fill-mask": "AutoModelForMaskedLM",
            "image-classification": "AutoModelForImageClassification",
            "image-segmentation": (
                "AutoModelForImageSegmentation",
                "AutoModelForSemanticSegmentation",
                "AutoModelForInstanceSegmentation",
                "AutoModelForUniversalSegmentation",
            ),
            "image-to-image": "AutoModelForImageToImage",
            "image-to-text": ("AutoModelForVision2Seq", "AutoModel"),
            "mask-generation": "AutoModel",
            "masked-im": "AutoModelForMaskedImageModeling",
            "multiple-choice": "AutoModelForMultipleChoice",
            "object-detection": "AutoModelForObjectDetection",
            "question-answering": "AutoModelForQuestionAnswering",
            "reinforcement-learning": "AutoModel",
            "semantic-segmentation": "AutoModelForSemanticSegmentation",
            "text-to-audio": ("AutoModelForTextToSpectrogram", "AutoModelForTextToWaveform"),
            "text-generation": "AutoModelForCausalLM",
            "text2text-generation": "AutoModelForSeq2SeqLM",
            "text-classification": "AutoModelForSequenceClassification",
            "token-classification": "AutoModelForTokenClassification",
            "visual-question-answering": "AutoModelForVisualQuestionAnswering",
            "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
            "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
        }

        _TRANSFORMERS_TASKS_TO_MODEL_MAPPINGS = get_transformers_tasks_to_model_mapping(
            _TRANSFORMERS_TASKS_TO_MODEL_LOADERS, framework="pt"
        )

        _TIMM_TASKS_TO_MODEL_LOADERS = {
            "image-classification": "create_model",
        }

        _SENTENCE_TRANSFORMERS_TASKS_TO_MODEL_LOADERS = {
            "feature-extraction": "SentenceTransformer",
            "sentence-similarity": "SentenceTransformer",
        }

        if is_diffusers_available():
            _DIFFUSERS_TASKS_TO_MODEL_LOADERS = {
                "image-to-image": "AutoPipelineForImage2Image",
                "inpainting": "AutoPipelineForInpainting",
                "text-to-image": "AutoPipelineForText2Image",
            }

            _DIFFUSERS_TASKS_TO_MODEL_MAPPINGS = get_diffusers_tasks_to_model_mapping()

        _LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP = {
            "diffusers": _DIFFUSERS_TASKS_TO_MODEL_LOADERS,
            "sentence_transformers": _SENTENCE_TRANSFORMERS_TASKS_TO_MODEL_LOADERS,
            "timm": _TIMM_TASKS_TO_MODEL_LOADERS,
            "transformers": _TRANSFORMERS_TASKS_TO_MODEL_LOADERS,
        }

    if is_tf_available():
        _TRANSFORMERS_TASKS_TO_TF_MODEL_LOADERS = {
            "document-question-answering": "TFAutoModelForDocumentQuestionAnswering",
            "feature-extraction": "TFAutoModel",
            "fill-mask": "TFAutoModelForMaskedLM",
            "text-generation": "TFAutoModelForCausalLM",
            "image-classification": "TFAutoModelForImageClassification",
            "text2text-generation": "TFAutoModelForSeq2SeqLM",
            "text-classification": "TFAutoModelForSequenceClassification",
            "token-classification": "TFAutoModelForTokenClassification",
            "multiple-choice": "TFAutoModelForMultipleChoice",
            "question-answering": "TFAutoModelForQuestionAnswering",
            "image-segmentation": "TFAutoModelForImageSegmentation",
            "masked-im": "TFAutoModelForMaskedImageModeling",
            "semantic-segmentation": "TFAutoModelForSemanticSegmentation",
            "automatic-speech-recognition": "TFAutoModelForSpeechSeq2Seq",
            "audio-classification": "TFAutoModelForAudioClassification",
            "image-to-text": "TFAutoModelForVision2Seq",
            "zero-shot-image-classification": "TFAutoModelForZeroShotImageClassification",
            "zero-shot-object-detection": "TFAutoModelForZeroShotObjectDetection",
        }

        _LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP = {
            "transformers": _TRANSFORMERS_TASKS_TO_TF_MODEL_LOADERS,
        }

        _TRANSFORMERS_TASKS_TO_TF_MODEL_MAPPINGS = get_transformers_tasks_to_model_mapping(
            _TRANSFORMERS_TASKS_TO_TF_MODEL_LOADERS, framework="tf"
        )

    _SYNONYM_TASK_MAP = {
        "audio-ctc": "automatic-speech-recognition",
        "causal-lm": "text-generation",
        "causal-lm-with-past": "text-generation-with-past",
        "default": "feature-extraction",
        "default-with-past": "feature-extraction-with-past",
        "masked-lm": "fill-mask",
        "mask-generation": "feature-extraction",
        "sentence-similarity": "feature-extraction",
        "seq2seq-lm": "text2text-generation",
        "seq2seq-lm-with-past": "text2text-generation-with-past",
        "sequence-classification": "text-classification",
        "speech2seq-lm": "automatic-speech-recognition",
        "speech2seq-lm-with-past": "automatic-speech-recognition-with-past",
        "summarization": "text2text-generation",
        "text-to-speech": "text-to-audio",
        "translation": "text2text-generation",
        "vision2seq-lm": "image-to-text",
        "zero-shot-classification": "text-classification",
        "image-feature-extraction": "feature-extraction",
        "pretraining": "feature-extraction",
        # for backward compatibility and testing (where
        # model task and model type are still the same)
        "stable-diffusion": "text-to-image",
        "stable-diffusion-xl": "text-to-image",
        "latent-consistency": "text-to-image",
    }

    _CUSTOM_CLASSES = {
        ("pt", "colpali", "feature-extraction"): ("transformers", "ColPaliForRetrieval"),
        ("pt", "patchtsmixer", "time-series-forecasting"): ("transformers", "PatchTSMixerForPrediction"),
        ("pt", "patchtst", "time-series-forecasting"): ("transformers", "PatchTSTForPrediction"),
        ("pt", "pix2struct", "image-to-text"): ("transformers", "Pix2StructForConditionalGeneration"),
        ("pt", "pix2struct", "visual-question-answering"): ("transformers", "Pix2StructForConditionalGeneration"),
        ("pt", "visual_bert", "question-answering"): ("transformers", "VisualBertForQuestionAnswering"),
        # VisionEncoderDecoderModel is not registered in AutoModelForDocumentQuestionAnswering
        ("pt", "vision-encoder-decoder", "document-question-answering"): ("transformers", "VisionEncoderDecoderModel"),
        ("pt", "vitpose", "keypoint-detection"): ("transformers", "VitPoseForPoseEstimation"),
    }

    _ENCODER_DECODER_TASKS = (
        "automatic-speech-recognition",
        "document-question-answering",
        "feature-extraction-with-past",
        "image-to-text",
        "text2text-generation",
        "visual-question-answering",
    )

    _MODEL_TYPE_FOR_DEFAULT_CONFIG = {
        "timm": "default-timm-config",
    }

    _SUPPORTED_MODEL_TYPE = {}
    _TIMM_SUPPORTED_MODEL_TYPE = {}
    _DIFFUSERS_SUPPORTED_MODEL_TYPE = {}
    _SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE = {}

    _LIBRARY_TO_SUPPORTED_MODEL_TYPES = {
        "diffusers": _DIFFUSERS_SUPPORTED_MODEL_TYPE,
        "sentence_transformers": _SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE,
        "timm": _TIMM_SUPPORTED_MODEL_TYPE,
        "transformers": _SUPPORTED_MODEL_TYPE,
    }
    _UNSUPPORTED_CLI_MODEL_TYPE = {
        # diffusers model part
        "clip-text",
        "clip-text-with-projection",
        "flux-transformer-2d",
        "sd3-transformer-2d",
        "siglip-text",
        "siglip-text-with-projection",
        "t5-encoder",
        "unet-2d-condition",
        "vae-decoder",
        "vae-encoder",
    }
    _SUPPORTED_CLI_MODEL_TYPE = (
        set(_SUPPORTED_MODEL_TYPE.keys())
        | set(_DIFFUSERS_SUPPORTED_MODEL_TYPE.keys())
        | set(_TIMM_SUPPORTED_MODEL_TYPE.keys())
        | set(_SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE.keys())
    ) - _UNSUPPORTED_CLI_MODEL_TYPE

    @classmethod
    def create_register(
        cls, backend: str, overwrite_existing: bool = False
    ) -> Callable[[str, Tuple[str, ...]], Callable[[Type], Type]]:
        """
        Creates a register function for the specified backend.

        Args:
            backend (`str`):
                The name of the backend that the register function will handle.
            overwrite_existing (`bool`, defaults to `False`):
                Whether or not the register function is allowed to overwrite an already existing config.

        Returns:
            `Callable[[str, Tuple[str, ...]], Callable[[Type], Type]]`: A decorator taking the model type and a the
            supported tasks.

        Example:
            ```python
            >>> register_for_new_backend = create_register("new-backend")

            >>> @register_for_new_backend("bert", "text-classification", "token-classification")
            >>> class BertNewBackendConfig(NewBackendConfig):
            >>>     pass
            ```
        """

        def wrapper(
            model_type: str, *supported_tasks: str, library_name: str = "transformers"
        ) -> Callable[[Type], Type]:
            def decorator(config_cls: Type) -> Type:
                supported_model_type_for_library = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[
                    library_name
                ]  # This is a pointer.

                mapping = supported_model_type_for_library.get(model_type, {})
                mapping_backend = mapping.get(backend, {})
                for task in supported_tasks:
                    normalized_task = task.replace("-with-past", "")
                    if normalized_task not in cls.get_all_tasks():
                        known_tasks = ", ".join(cls.get_all_tasks())
                        raise ValueError(
                            f'The TasksManager does not know the task called "{normalized_task}", known tasks: {known_tasks}.'
                        )
                    if not overwrite_existing and task in mapping_backend:
                        continue
                    mapping_backend[task] = make_backend_config_constructor_for_task(config_cls, task)
                mapping[backend] = mapping_backend
                supported_model_type_for_library[model_type] = mapping
                return config_cls

            return decorator

        return wrapper

    @staticmethod
    def get_supported_tasks_for_model_type(
        model_type: str, exporter: str, model_name: Optional[str] = None, library_name: Optional[str] = None
    ) -> TaskNameToExportConfigDict:
        """
        Retrieves the `task -> exporter backend config constructors` map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported tasks for.
            exporter (`str`):
                The name of the exporter.
            model_name (`Optional[str]`, defaults to `None`):
                The name attribute of the model object, only used for the exception message.
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".

        Returns:
            `TaskNameToExportConfigDict`: The dictionary mapping each task to a corresponding `ExporterConfig`
            constructor.
        """
        if library_name is None:
            logger.warning(
                'Not passing the argument `library_name` to `get_supported_tasks_for_model_type` is deprecated and the support will be removed in a future version of Optimum. Please specify a `library_name`. Defaulting to `"transformers`.'
            )

            # We are screwed if different dictionaries have the same keys.
            supported_model_type_for_library = {
                **TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE,
                **TasksManager._TIMM_SUPPORTED_MODEL_TYPE,
                **TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE,
                **TasksManager._SUPPORTED_MODEL_TYPE,
            }
            library_name = "transformers"
        else:
            supported_model_type_for_library = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]

        model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type

        default_model_type = None
        if library_name in TasksManager._MODEL_TYPE_FOR_DEFAULT_CONFIG:
            default_model_type = TasksManager._MODEL_TYPE_FOR_DEFAULT_CONFIG[library_name]

        if model_type not in supported_model_type_for_library:
            if default_model_type is not None:
                model_type = default_model_type
            else:
                raise KeyError(
                    f"{model_type_and_model_name} is not supported yet for {library_name}. "
                    f"Only {list(supported_model_type_for_library.keys())} are supported for the library {library_name}. "
                    f"If you want to support {model_type} please propose a PR or open up an issue."
                )
        if exporter not in supported_model_type_for_library[model_type]:
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet with the {exporter} backend. "
                f"Only {list(supported_model_type_for_library[model_type].keys())} are supported. "
                f"If you want to support {exporter} please propose a PR or open up an issue."
            )

        return supported_model_type_for_library[model_type][exporter]

    @staticmethod
    def get_supported_model_type_for_task(task: str, exporter: str) -> List[str]:
        """
        Returns the list of supported architectures by the exporter for a given task. Transformers-specific.
        """

        supported_model_types = [
            model_type
            for model_type in TasksManager._SUPPORTED_MODEL_TYPE
            if task in TasksManager._SUPPORTED_MODEL_TYPE[model_type][exporter]
        ]

        return supported_model_types

    @staticmethod
    def synonyms_for_task(task: str) -> Set[str]:
        synonyms = [k for k, v in TasksManager._SYNONYM_TASK_MAP.items() if v == task]
        synonyms += [k for k, v in TasksManager._SYNONYM_TASK_MAP.items() if v == TasksManager.map_from_synonym(task)]
        synonyms = set(synonyms)
        try:
            synonyms.remove(task)
        except KeyError:
            pass
        return synonyms

    @staticmethod
    def map_from_synonym(task: str) -> str:
        if task in TasksManager._SYNONYM_TASK_MAP:
            task = TasksManager._SYNONYM_TASK_MAP[task]
        return task

    @staticmethod
    def _validate_framework_choice(framework: str):
        """
        Validates if the framework requested for the export is both correct and available, otherwise throws an
        exception.
        """
        if framework not in ["pt", "tf"]:
            raise ValueError(f"Only two frameworks are supported for export: pt or tf, but {framework} was provided.")
        elif framework == "pt" and not is_torch_available():
            raise RuntimeError("Cannot export model using PyTorch because no PyTorch package was found.")
        elif framework == "tf" and not is_tf_available():
            raise RuntimeError("Cannot export model using TensorFlow because no TensorFlow package was found.")

    @staticmethod
    def get_model_class_for_task(
        task: str,
        framework: str = "pt",
        model_type: Optional[str] = None,
        model_class_name: Optional[str] = None,
        library: str = "transformers",
    ) -> Type:
        """
        Attempts to retrieve an AutoModel class from a task name.

        Args:
            task (`str`):
                The task required.
            framework (`str`, defaults to `"pt"`):
                The framework to use for the export.
            model_type (`Optional[str]`, defaults to `None`):
                The model type to retrieve the model class for. Some architectures need a custom class to be loaded,
                and can not be loaded from auto class.
            model_class_name (`Optional[str]`, defaults to `None`):
                A model class name, allowing to override the default class that would be detected for the task. This
                parameter is useful for example for "automatic-speech-recognition", that may map to
                AutoModelForSpeechSeq2Seq or to AutoModelForCTC.
            library (`str`, defaults to `transformers`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".

        Returns:
            The AutoModel class corresponding to the task.
        """
        task = task.replace("-with-past", "")
        task = TasksManager.map_from_synonym(task)

        TasksManager._validate_framework_choice(framework)

        if (framework, model_type, task) in TasksManager._CUSTOM_CLASSES:
            library, class_name = TasksManager._CUSTOM_CLASSES[(framework, model_type, task)]
            loaded_library = importlib.import_module(library)

            return getattr(loaded_library, class_name)
        else:
            if framework == "pt":
                tasks_to_model_loader = TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[library]
            else:
                tasks_to_model_loader = TasksManager._LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP[library]

            loaded_library = importlib.import_module(library)

            if model_class_name is None:
                if task not in tasks_to_model_loader:
                    raise KeyError(
                        f"Unknown task: {task}. Possible values are: "
                        + ", ".join([f"`{key}` for {tasks_to_model_loader[key]}" for key in tasks_to_model_loader])
                    )

                if isinstance(tasks_to_model_loader[task], str):
                    model_class_name = tasks_to_model_loader[task]
                else:
                    # automatic-speech-recognition case, which may map to several auto class
                    if library == "transformers":
                        if model_type is None:
                            logger.warning(
                                f"No model type passed for the task {task}, that may be mapped to several loading"
                                f" classes ({tasks_to_model_loader[task]}). Defaulting to {tasks_to_model_loader[task][0]}"
                                " to load the model."
                            )
                            model_class_name = tasks_to_model_loader[task][0]
                        else:
                            for autoclass_name in tasks_to_model_loader[task]:
                                module = getattr(loaded_library, autoclass_name)
                                if model_type in module._model_mapping._model_mapping:
                                    model_class_name = autoclass_name
                                    break

                            if model_class_name is None:
                                raise ValueError(
                                    f"Unrecognized configuration classes {tasks_to_model_loader[task]} do not match"
                                    f" with the model type {model_type} and task {task}."
                                )
                    else:
                        raise NotImplementedError(
                            "For library other than transformers, the _TASKS_TO_MODEL_LOADER mapping should be one to one."
                        )

            return getattr(loaded_library, model_class_name)

    @staticmethod
    def get_model_files(
        model_name_or_path: Union[str, Path],
        subfolder: str = "",
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        request_exception = None
        full_model_path = Path(model_name_or_path, subfolder)

        hf_api = HfApi(user_agent=http_user_agent(), token=token)

        if full_model_path.is_dir():
            all_files = [
                os.path.relpath(os.path.join(dirpath, file), full_model_path)
                for dirpath, _, filenames in os.walk(full_model_path)
                for file in filenames
            ]
        else:
            try:
                if not isinstance(model_name_or_path, str):
                    model_name_or_path = str(model_name_or_path)
                all_files = hf_api.list_repo_files(
                    model_name_or_path,
                    repo_type="model",
                    revision=revision,
                    token=token,
                )
                if subfolder != "":
                    all_files = [file[len(subfolder) + 1 :] for file in all_files if file.startswith(subfolder)]
            except (RequestsConnectionError, OfflineModeIsEnabled) as e:
                snapshot_path = hf_api.snapshot_download(
                    repo_id=model_name_or_path,
                    cache_dir=cache_dir,
                    revision=revision,
                    token=token,
                )
                full_model_path = Path(snapshot_path, subfolder)
                if full_model_path.is_dir():
                    all_files = [
                        os.path.relpath(os.path.join(dirpath, file), full_model_path)
                        for dirpath, _, filenames in os.walk(full_model_path)
                        for file in filenames
                    ]
                else:
                    request_exception = e

        return all_files, request_exception

    @staticmethod
    def determine_framework(
        model_name_or_path: Union[str, Path],
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
    ) -> str:
        """
        Determines the framework to use for the export.

        The priority is in the following order:
            1. User input via `framework`.
            2. If local checkpoint is provided, use the same framework as the checkpoint.
            3. If model repo, try to infer the framework from the cache if available, else from the Hub.
            4. If could not infer, use available framework in environment, with priority given to PyTorch.

        Args:
            model_name_or_path (`Union[str, Path]`):
                Can be either the model id of a model repo on the Hugging Face Hub, or a path to a local directory
                containing a model.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`,  defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).

        Returns:
            `str`: The framework to use for the export.

        """

        all_files, request_exception = TasksManager.get_model_files(
            model_name_or_path, subfolder=subfolder, cache_dir=cache_dir, token=token, revision=revision
        )

        pt_weight_name = Path(WEIGHTS_NAME).stem
        pt_weight_extension = Path(WEIGHTS_NAME).suffix
        safe_weight_name = Path(SAFE_WEIGHTS_NAME).stem
        safe_weight_extension = Path(SAFE_WEIGHTS_NAME).suffix
        is_pt_weight_file = [
            (file.startswith(pt_weight_name) and file.endswith(pt_weight_extension))
            or (file.startswith(safe_weight_name) and file.endswith(safe_weight_extension))
            for file in all_files
        ]

        weight_name = Path(TF2_WEIGHTS_NAME).stem
        weight_extension = Path(TF2_WEIGHTS_NAME).suffix
        is_tf_weight_file = [file.startswith(weight_name) and file.endswith(weight_extension) for file in all_files]

        if any(is_pt_weight_file):
            framework = "pt"
        elif any(is_tf_weight_file):
            framework = "tf"
        elif "model_index.json" in all_files and any(
            file.endswith((pt_weight_extension, safe_weight_extension)) for file in all_files
        ):
            # diffusers case
            framework = "pt"
        elif "config_sentence_transformers.json" in all_files:
            # Sentence Transformers libary relies on PyTorch.
            framework = "pt"
        else:
            if request_exception is not None:
                raise RequestsConnectionError(
                    f"The framework could not be automatically inferred. If using the command-line, please provide the argument --framework (pt,tf) Detailed error: {request_exception}"
                )
            else:
                raise FileNotFoundError(
                    "Cannot determine framework from given checkpoint location."
                    f" There should be a {Path(WEIGHTS_NAME).stem}*{Path(WEIGHTS_NAME).suffix} for PyTorch"
                    f" or {Path(TF2_WEIGHTS_NAME).stem}*{Path(TF2_WEIGHTS_NAME).suffix} for TensorFlow."
                )

        if is_torch_available():
            framework = framework or "pt"
        elif is_tf_available():
            framework = framework or "tf"
        else:
            raise EnvironmentError("Neither PyTorch nor TensorFlow found in environment. Cannot export model.")

        logger.info(f"Framework not specified. Using {framework} to export the model.")

        return framework

    @classmethod
    def _infer_task_from_model_or_model_class(
        cls,
        model: Optional[Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"]] = None,
        model_class: Optional[Type[Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"]]] = None,
    ) -> str:
        if model is not None and model_class is not None:
            raise ValueError("Either a model or a model class must be provided, but both were given here.")
        if model is None and model_class is None:
            raise ValueError("Either a model or a model class must be provided, but none were given here.")

        target_class_name = model.__class__.__name__ if model is not None else model_class.__name__
        target_class_module = model.__class__.__module__ if model is not None else model_class.__module__

        # using TASKS_TO_MODEL_LOADERS to infer the task name
        tasks_to_model_loaders = None

        if target_class_name.startswith("AutoModel"):
            tasks_to_model_loaders = cls._TRANSFORMERS_TASKS_TO_MODEL_LOADERS
        elif target_class_name.startswith("TFAutoModel"):
            tasks_to_model_loaders = cls._TRANSFORMERS_TASKS_TO_TF_MODEL_LOADERS
        elif target_class_name.startswith("AutoPipeline"):
            tasks_to_model_loaders = cls._DIFFUSERS_TASKS_TO_MODEL_LOADERS

        if tasks_to_model_loaders is not None:
            for task_name, model_loaders in tasks_to_model_loaders.items():
                if isinstance(model_loaders, str):
                    model_loaders = (model_loaders,)
                for model_loader_class_name in model_loaders:
                    if target_class_name == model_loader_class_name:
                        return task_name

        # using TASKS_TO_MODEL_MAPPINGS to infer the task name
        tasks_to_model_mapping = None

        if target_class_module.startswith("transformers"):
            if target_class_name.startswith("TF"):
                tasks_to_model_mapping = cls._TRANSFORMERS_TASKS_TO_TF_MODEL_MAPPINGS
            else:
                tasks_to_model_mapping = cls._TRANSFORMERS_TASKS_TO_MODEL_MAPPINGS
        elif target_class_module.startswith("diffusers"):
            tasks_to_model_mapping = cls._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS

        if tasks_to_model_mapping is not None:
            for task_name, model_mapping in tasks_to_model_mapping.items():
                for model_type, model_class_name in model_mapping.items():
                    if target_class_name == model_class_name:
                        return task_name

        raise ValueError(
            "The task name could not be automatically inferred. If using the command-line, please provide the argument --task task-name. Example: `--task text-classification`."
        )

    @classmethod
    def _infer_task_from_model_name_or_path(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        library_name: Optional[str] = None,
    ) -> str:
        inferred_task_name = None

        is_local = os.path.isdir(os.path.join(model_name_or_path, subfolder))

        if is_local:
            # TODO: maybe implement that.
            raise RuntimeError(
                f"Cannot infer the task from a local directory yet, please specify the task manually ({', '.join(TasksManager.get_all_tasks())})."
            )
        else:
            if subfolder != "":
                raise RuntimeError(
                    "Cannot infer the task from a model repo with a subfolder yet, please specify the task manually."
                )
            try:
                model_info = HfApi(user_agent=http_user_agent(), token=token).model_info(
                    model_name_or_path, revision=revision, token=token
                )
            except (RequestsConnectionError, OfflineModeIsEnabled):
                raise RuntimeError(
                    f"Hugging Face Hub is not reachable and we cannot infer the task from a cached model. Make sure you are not offline, or otherwise please specify the `task` (or `--task` in command-line) argument ({', '.join(TasksManager.get_all_tasks())})."
                )
            if library_name is None:
                library_name = cls.infer_library_from_model(
                    model_name_or_path,
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                )

            if library_name == "timm":
                inferred_task_name = "image-classification"
            elif library_name == "diffusers":
                pipeline_tag = pipeline_tag = model_info.pipeline_tag
                model_config = model_info.config
                if pipeline_tag is not None:
                    inferred_task_name = cls.map_from_synonym(pipeline_tag)
                elif model_config is not None:
                    if model_config is not None and model_config.get("diffusers", None) is not None:
                        diffusers_class_name = model_config["diffusers"]["_class_name"]
                        for task_name, model_mapping in cls._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS.items():
                            for model_type, model_class_name in model_mapping.items():
                                if diffusers_class_name == model_class_name:
                                    inferred_task_name = task_name
                                    break
                            if inferred_task_name is not None:
                                break
            elif library_name == "sentence_transformers":
                inferred_task_name = "feature-extraction"
            elif library_name == "transformers":
                pipeline_tag = model_info.pipeline_tag
                transformers_info = model_info.transformersInfo
                if pipeline_tag is not None:
                    inferred_task_name = cls.map_from_synonym(model_info.pipeline_tag)
                elif transformers_info is not None:
                    transformers_pipeline_tag = transformers_info.get("pipeline_tag", None)
                    transformers_auto_model = transformers_info.get("auto_model", None)
                    if transformers_pipeline_tag is not None:
                        pipeline_tag = transformers_info["pipeline_tag"]
                        inferred_task_name = cls.map_from_synonym(pipeline_tag)
                    elif transformers_auto_model is not None:
                        transformers_auto_model = transformers_auto_model.replace("TF", "")
                        for task_name, model_loaders in cls._TRANSFORMERS_TASKS_TO_MODEL_LOADERS.items():
                            if isinstance(model_loaders, str):
                                model_loaders = (model_loaders,)
                            for model_loader_class_name in model_loaders:
                                if transformers_auto_model == model_loader_class_name:
                                    inferred_task_name = task_name
                                    break
                            if inferred_task_name is not None:
                                break

        if inferred_task_name is None:
            raise KeyError(f"Could not find the proper task name for the model {model_name_or_path}.")

        return inferred_task_name

    @classmethod
    def infer_task_from_model(
        cls,
        model: Union[str, "PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline", Type],
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        library_name: Optional[str] = None,
    ) -> str:
        """
        Infers the task from the model repo, model instance, or model class.

        Args:
            model (`Union[str, PreTrainedModel, TFPreTrainedModel, DiffusionPipeline, Type]`):
                The model to infer the task from. This can either be the name of a repo on the HuggingFace Hub, an
                instance of a model, or a model class.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`,  defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers". See `TasksManager.infer_library_from_model` for the priority should
                none be provided.
        Returns:
            `str`: The task name automatically detected from the HF hub repo, model instance, or model class.
        """
        inferred_task_name = None

        if isinstance(model, str):
            inferred_task_name = cls._infer_task_from_model_name_or_path(
                model_name_or_path=model,
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                library_name=library_name,
            )
        elif type(model) is type:
            inferred_task_name = cls._infer_task_from_model_or_model_class(model_class=model)
        else:
            inferred_task_name = cls._infer_task_from_model_or_model_class(model=model)

        if inferred_task_name is None:
            raise ValueError(
                "The task name could not be automatically inferred. If using the command-line, please provide the argument --task task-name. Example: `--task text-classification`."
            )

        return inferred_task_name

    @classmethod
    def _infer_library_from_model_or_model_class(
        cls,
        model: Optional[Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"]] = None,
        model_class: Optional[Type[Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"]]] = None,
    ):
        inferred_library_name = None
        if model is not None and model_class is not None:
            raise ValueError("Either a model or a model class must be provided, but both were given here.")
        if model is None and model_class is None:
            raise ValueError("Either a model or a model class must be provided, but none were given here.")

        target_class_module = model.__class__.__module__ if model is not None else model_class.__module__

        if target_class_module.startswith("sentence_transformers"):
            inferred_library_name = "sentence_transformers"
        elif target_class_module.startswith("transformers"):
            inferred_library_name = "transformers"
        elif target_class_module.startswith("diffusers"):
            inferred_library_name = "diffusers"
        elif target_class_module.startswith("timm"):
            inferred_library_name = "timm"

        if inferred_library_name is None:
            raise ValueError(
                "The library name could not be automatically inferred. If using the command-line, please provide the argument --library {transformers,diffusers,timm,sentence_transformers}. Example: `--library diffusers`."
            )

        return inferred_library_name

    @classmethod
    def _infer_library_from_model_name_or_path(
        cls,
        model_name_or_path: Union[str, Path],
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
    ):
        """
        Infers the library from the model name or path.

        Args:
            model_name_or_path (`str`):
                The model to infer the task from. This can either be the name of a repo on the HuggingFace Hub, or a path
                to a local directory containing the model.
            subfolder (`str`, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*, defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).

        Returns:
            `str`: The library name automatically detected from the model repo.
        """

        inferred_library_name = None

        all_files, _ = TasksManager.get_model_files(
            model_name_or_path,
            subfolder=subfolder,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )

        if "model_index.json" in all_files:
            inferred_library_name = "diffusers"
        elif (
            any(file_path.startswith("sentence_") for file_path in all_files)
            or "config_sentence_transformers.json" in all_files
        ):
            inferred_library_name = "sentence_transformers"
        elif "config.json" in all_files:
            kwargs = {
                "subfolder": subfolder,
                "revision": revision,
                "cache_dir": cache_dir,
                "token": token,
            }
            # We do not use PretrainedConfig.from_pretrained which has unwanted warnings about model type.
            config_dict, kwargs = PretrainedConfig.get_config_dict(model_name_or_path, **kwargs)
            model_config = PretrainedConfig.from_dict(config_dict, **kwargs)

            if hasattr(model_config, "pretrained_cfg") or hasattr(model_config, "architecture"):
                inferred_library_name = "timm"
            elif hasattr(model_config, "_diffusers_version"):
                inferred_library_name = "diffusers"
            else:
                inferred_library_name = "transformers"

        if inferred_library_name is None:
            raise ValueError(
                "The library name could not be automatically inferred. If using the command-line, please provide the argument --library {transformers,diffusers,timm,sentence_transformers}. Example: `--library diffusers`."
            )

        return inferred_library_name

    @classmethod
    def infer_library_from_model(
        cls,
        model: Union[str, "PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline", Type],
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
    ):
        """
        Infers the library from the model repo, model instance, or model class.

        Args:
            model (`Union[str, PreTrainedModel, TFPreTrainedModel, DiffusionPipeline, Type]`):
                The model to infer the task from. This can either be the name of a repo on the HuggingFace Hub, an
                instance of a model, or a model class.
            subfolder (`str`, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*, defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).

        Returns:
            `str`: The library name automatically detected from the model repo, model instance, or model class.
        """

        if isinstance(model, str):
            library_name = cls._infer_library_from_model_name_or_path(
                model_name_or_path=model,
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        elif type(model) is type:
            library_name = cls._infer_library_from_model_or_model_class(model_class=model)
        else:
            library_name = cls._infer_library_from_model_or_model_class(model=model)

        return library_name

    @classmethod
    def standardize_model_attributes(
        cls,
        model: Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"],
        library_name: Optional[str] = None,
    ):
        """
        Updates the model for export. This function is suitable to make required changes to the models from different
        libraries to follow transformers style.

        Args:
            model (`Union[PreTrainedModel, TFPreTrainedModel, DiffusionPipeline]`):
                The instance of the model.

        """

        if library_name is None:
            library_name = TasksManager.infer_library_from_model(model)

        if library_name == "diffusers":
            inferred_model_type = None

            for task_name, model_mapping in cls._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS.items():
                for model_type, model_class_name in model_mapping.items():
                    if model.__class__.__name__ == model_class_name:
                        inferred_model_type = model_type
                        break
                if inferred_model_type is not None:
                    break

            # `model_type` is a class attribute in Transformers, let's avoid modifying it.
            model.config.export_model_type = inferred_model_type

        elif library_name == "timm":
            # Retrieve model config and set it like in transformers
            model.config = PretrainedConfig.from_dict(model.pretrained_cfg)
            # `model_type` is a class attribute in Transformers, let's avoid modifying it.
            model.config.export_model_type = model.pretrained_cfg["architecture"]

        elif library_name == "sentence_transformers":
            if "Transformer" in model[0].__class__.__name__:
                model.config = model[0].auto_model.config
                model.config.export_model_type = "transformer"
            elif "CLIP" in model[0].__class__.__name__:
                model.config = model[0].model.config
                model.config.export_model_type = "clip"
            else:
                raise ValueError(
                    f"The export of a sentence_transformers model with the first module being {model[0].__class__.__name__} is currently not supported in Optimum. Please open an issue or submit a PR to add the support."
                )

    @staticmethod
    def get_all_tasks():
        """
        Retrieves all the possible tasks.

        Returns:
            `List`: all the possible tasks.
        """
        tasks = []
        if is_torch_available():
            framework = "pt"
            mapping = TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP
        else:
            framework = "tf"
            mapping = TasksManager._LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP

        tasks = []
        for d in mapping.values():
            tasks += list(d.keys())

        for custom_class in TasksManager._CUSTOM_CLASSES:
            if custom_class[0] == framework:
                tasks.append(custom_class[2])

        tasks = list(set(tasks))

        return tasks

    @staticmethod
    def get_model_from_task(
        task: str,
        model_name_or_path: Union[str, Path],
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        framework: Optional[str] = None,
        torch_dtype: Optional["torch.dtype"] = None,
        device: Optional[Union["torch.device", str]] = None,
        library_name: Optional[str] = None,
        **model_kwargs,
    ) -> Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"]:
        """
        Retrieves a model from its name and the task to be enabled.

        Args:
            task (`str`):
                The task required.
            model_name_or_path (`Union[str, Path]`):
                Can be either the model id of a model repo on the Hugging Face Hub, or a path to a local directory
                containing a model.
            subfolder (`str`, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
            framework (`Optional[str]`, *optional*):
                The framework to use for the export. See `TasksManager.determine_framework` for the priority should
                none be provided.
            torch_dtype (`Optional[torch.dtype]`, defaults to `None`):
                Data type to load the model on. PyTorch-only argument.
            device (`Optional[torch.device]`, defaults to `None`):
                Device to initialize the model on. PyTorch-only argument. For PyTorch, defaults to "cpu".
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers". See `TasksManager.infer_library_from_model` for the priority should
                none be provided.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to pass to the model `.from_pretrained()` method.
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers". See `TasksManager.infer_library_from_model` for the priority should
                none be provided.

        Returns:
            The instance of the model.

        """

        if framework is None:
            framework = TasksManager.determine_framework(
                model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=cache_dir, token=token
            )

        if library_name is None:
            library_name = TasksManager.infer_library_from_model(
                model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=cache_dir, token=token
            )

        original_task = task
        if task == "auto":
            task = TasksManager.infer_task_from_model(
                model_name_or_path,
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                library_name=library_name,
            )

        model_type = None
        model_class_name = None
        kwargs = {"subfolder": subfolder, "revision": revision, "cache_dir": cache_dir, **model_kwargs}

        if library_name == "transformers":
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
            model_type = config.model_type
            if original_task == "automatic-speech-recognition" or task == "automatic-speech-recognition":
                if original_task == "auto" and config.architectures is not None:
                    model_class_name = config.architectures[0]
            elif original_task == "reinforcement-learning" or task == "reinforcement-learning":
                if config.architectures is not None:
                    model_class_name = config.architectures[0]

        if library_name == "diffusers":
            config = DiffusionPipeline.load_config(model_name_or_path, **kwargs)
            class_name = config.get("_class_name", None)
            loaded_library = importlib.import_module(library_name)
            model_class = getattr(loaded_library, class_name)
        else:
            model_class = TasksManager.get_model_class_for_task(
                task, framework, model_type=model_type, model_class_name=model_class_name, library=library_name
            )

        if library_name == "timm":
            model = model_class(f"hf_hub:{model_name_or_path}", pretrained=True, exportable=True)
            model = model.to(torch_dtype).to(device)
        elif library_name == "sentence_transformers":
            cache_folder = model_kwargs.pop("cache_folder", None)
            use_auth_token = model_kwargs.pop("use_auth_token", None)
            token = model_kwargs.pop("token", None)
            trust_remote_code = model_kwargs.pop("trust_remote_code", False)
            model_kwargs["torch_dtype"] = torch_dtype

            if use_auth_token is not None:
                warnings.warn(
                    "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                    FutureWarning,
                )
                if token is not None:
                    raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
                token = use_auth_token

            model = model_class(
                model_name_or_path,
                device=device,
                cache_folder=cache_folder,
                token=token,
                revision=revision,
                trust_remote_code=trust_remote_code,
                model_kwargs=model_kwargs,
            )
        else:
            try:
                if framework == "pt":
                    kwargs["torch_dtype"] = torch_dtype

                    if isinstance(device, str):
                        device = torch.device(device)
                    elif device is None:
                        device = torch.device("cpu")

                    # TODO : fix EulerDiscreteScheduler loading to enable for SD models
                    if version.parse(torch.__version__) >= version.parse("2.0") and library_name != "diffusers":
                        with device:
                            # Initialize directly in the requested device, to save allocation time. Especially useful for large
                            # models to initialize on cuda device.
                            model = model_class.from_pretrained(model_name_or_path, **kwargs)
                    else:
                        model = model_class.from_pretrained(model_name_or_path, **kwargs).to(device)
                else:
                    model = model_class.from_pretrained(model_name_or_path, **kwargs)
            except OSError:
                if framework == "pt":
                    logger.info("Loading TensorFlow model in PyTorch before exporting.")
                    kwargs["from_tf"] = True
                    model = model_class.from_pretrained(model_name_or_path, **kwargs)
                else:
                    logger.info("Loading PyTorch model in TensorFlow before exporting.")
                    kwargs["from_pt"] = True
                    model = model_class.from_pretrained(model_name_or_path, **kwargs)

        TasksManager.standardize_model_attributes(model, library_name=library_name)

        return model

    @staticmethod
    def get_exporter_config_constructor(
        exporter: str,
        model: Optional[Union["PreTrainedModel", "TFPreTrainedModel"]] = None,
        task: str = "feature-extraction",
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        exporter_config_kwargs: Optional[Dict[str, Any]] = None,
        library_name: Optional[str] = None,
    ) -> ExportConfigConstructor:
        """
        Gets the `ExportConfigConstructor` for a model (or alternatively for a model type) and task combination.

        Args:
            exporter (`str`):
                The exporter to use.
            model (`Optional[Union[PreTrainedModel, TFPreTrainedModel]]`, defaults to `None`):
                The instance of the model.
            task (`str`, defaults to `"feature-extraction"`):
                The task to retrieve the config for.
            model_type (`Optional[str]`, defaults to `None`):
                The model type to retrieve the config for.
            model_name (`Optional[str]`, defaults to `None`):
                The name attribute of the model object, only used for the exception message.
            exporter_config_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
                Arguments that will be passed to the exporter config class when building the config constructor.
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".

        Returns:
            `ExportConfigConstructor`: The `ExporterConfig` constructor for the requested backend.
        """
        if library_name is None:
            logger.warning(
                "Passing the argument `library_name` to `get_supported_tasks_for_model_type` is required, but got library_name=None. Defaulting to `transformers`. An error will be raised in a future version of Optimum if `library_name` is not provided."
            )

            # We are screwed if different dictionaries have the same keys.
            supported_model_type_for_library = {
                **TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE,
                **TasksManager._TIMM_SUPPORTED_MODEL_TYPE,
                **TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE,
                **TasksManager._SUPPORTED_MODEL_TYPE,
            }
            library_name = "transformers"
        else:
            supported_model_type_for_library = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]

        if model is None and model_type is None:
            raise ValueError("Either a model_type or model should be provided to retrieve the export config.")

        if model_type is None:
            if hasattr(model.config, "export_model_type"):
                # We can specifiy a custom `export_model_type` attribute in the config. Useful for timm, sentence_transformers
                model_type = model.config.export_model_type
            else:
                model_type = getattr(model.config, "model_type", None)

            if model_type is None:
                raise ValueError("Model type cannot be inferred. Please provide the model_type for the model!")

            model_name = getattr(model, "name", model_name)

        model_tasks = TasksManager.get_supported_tasks_for_model_type(
            model_type, exporter, model_name=model_name, library_name=library_name
        )

        if task not in model_tasks:
            synonyms = TasksManager.synonyms_for_task(task)
            for synonym in synonyms:
                if synonym in model_tasks:
                    task = synonym
                    break
            if task not in model_tasks:
                raise ValueError(
                    f"{model_type} doesn't support task {task} for the {exporter} backend."
                    f" Supported tasks are: {', '.join(model_tasks.keys())}."
                )

        if model_type not in supported_model_type_for_library:
            model_type = TasksManager._MODEL_TYPE_FOR_DEFAULT_CONFIG[library_name]

        exporter_config_constructor = supported_model_type_for_library[model_type][exporter][task]
        if exporter_config_kwargs is not None:
            exporter_config_constructor = partial(exporter_config_constructor, **exporter_config_kwargs)

        return exporter_config_constructor
