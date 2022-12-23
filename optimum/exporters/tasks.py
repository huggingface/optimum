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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union

from transformers import PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging

import huggingface_hub


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel

    from .base import ExportConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if not is_torch_available() and not is_tf_available():
    logger.warning(
        "The export tasks are only supported for PyTorch or TensorFlow. You will not be able to export models"
        " without one of these libraries installed."
    )

ExportConfigConstructor = Callable[[PretrainedConfig], "ExportConfig"]
TaskNameToExportConfigDict = Dict[str, ExportConfigConstructor]


def supported_tasks_mapping(*supported_tasks: str, **exporters: str) -> Dict[str, TaskNameToExportConfigDict]:
    """
    Generates the mapping between supported tasks and their corresponding `ExportConfig` for a given model, for
    every backend.

    Args:
        supported_tasks (`Tuple[str]`):
            The names of the supported tasks.
        exporters (`Dict[str, str]`):
            The export backend name -> config class name mapping. For instance:
            ```python
            >>> kwargs = {
            >>>     "onnx": "BertOnnxConfig",
            >>>     "tflite": "BertTFLiteConfig",
            >>>     ...
            >>> }
            ```
    Returns:
        `Dict[str, TaskNameToExportConfigDict]`: The dictionary mapping a task to an `ExportConfig` constructor.
    """
    mapping = {}
    for backend, config_cls_name in exporters.items():
        config_cls = getattr(importlib.import_module(f"optimum.exporters.{backend}.model_configs"), config_cls_name)
        mapping[backend] = {}
        for task in supported_tasks:
            if "-with-past" in task:
                mapping[backend][task] = partial(config_cls.with_past, task=task.replace("-with-past", ""))
            else:
                mapping[backend][task] = partial(config_cls, task=task)

    return mapping


class TasksManager:
    """
    Handles the `task name -> model class` and `architecture -> configuration` mappings.
    """

    _TASKS_TO_AUTOMODELS = {}
    _TASKS_TO_TF_AUTOMODELS = {}
    if is_torch_available():
        _TASKS_TO_AUTOMODELS = {
            "default": "AutoModel",
            "masked-lm": "AutoModelForMaskedLM",
            "causal-lm": "AutoModelForCausalLM",
            "seq2seq-lm": "AutoModelForSeq2SeqLM",
            "sequence-classification": "AutoModelForSequenceClassification",
            "token-classification": "AutoModelForTokenClassification",
            "multiple-choice": "AutoModelForMultipleChoice",
            "object-detection": "AutoModelForObjectDetection",
            "question-answering": "AutoModelForQuestionAnswering",
            "image-classification": "AutoModelForImageClassification",
            "image-segmentation": "AutoModelForImageSegmentation",
            "masked-im": "AutoModelForMaskedImageModeling",
            "semantic-segmentation": "AutoModelForSemanticSegmentation",
            "speech2seq-lm": "AutoModelForSpeechSeq2Seq",
            "stable-diffusion": "StableDiffusionPipeline",
        }
    if is_tf_available():
        _TASKS_TO_TF_AUTOMODELS = {
            "default": "TFAutoModel",
            "masked-lm": "TFAutoModelForMaskedLM",
            "causal-lm": "TFAutoModelForCausalLM",
            "seq2seq-lm": "TFAutoModelForSeq2SeqLM",
            "sequence-classification": "TFAutoModelForSequenceClassification",
            "token-classification": "TFAutoModelForTokenClassification",
            "multiple-choice": "TFAutoModelForMultipleChoice",
            "question-answering": "TFAutoModelForQuestionAnswering",
            "semantic-segmentation": "TFAutoModelForSemanticSegmentation",
        }

    _TASKS_TO_LIBRARY = {
        "default": "transformers",
        "masked-lm": "transformers",
        "causal-lm": "transformers",
        "seq2seq-lm": "transformers",
        "sequence-classification": "transformers",
        "token-classification": "transformers",
        "multiple-choice": "transformers",
        "object-detection": "transformers",
        "question-answering": "transformers",
        "image-classification": "transformers",
        "image-segmentation": "transformers",
        "masked-im": "transformers",
        "semantic-segmentation": "transformers",
        "speech2seq-lm": "transformers",
        "stable-diffusion": "diffusers",
    }

    # Set of model topologies we support associated to the tasks supported by each topology and the factory
    _SUPPORTED_MODEL_TYPE = {
        "albert": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="AlbertOnnxConfig",
        ),
        "bart": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "sequence-classification",
            "question-answering",
            onnx="BartOnnxConfig",
        ),
        # BEiT cannot be used with the masked image modeling autoclass, so this task is excluded here
        "beit": supported_tasks_mapping("default", "image-classification", onnx="BeitOnnxConfig"),
        "bert": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for BERT
            # "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="BertOnnxConfig",
        ),
        "big-bird": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for big-bird
            # "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="BigBirdOnnxConfig",
        ),
        "bigbird-pegasus": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "sequence-classification",
            "question-answering",
            onnx="BigBirdPegasusOnnxConfig",
        ),
        "blenderbot": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx="BlenderbotOnnxConfig",
        ),
        "blenderbot-small": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx="BlenderbotSmallOnnxConfig",
        ),
        "bloom": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            "token-classification",
            onnx="BloomOnnxConfig",
        ),
        "camembert": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for camembert
            # "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="CamembertOnnxConfig",
        ),
        "clip": supported_tasks_mapping(
            "default",
            onnx="CLIPOnnxConfig",
        ),
        "clip-text-model": supported_tasks_mapping(
            "default",
            onnx="CLIPTextOnnxConfig",
        ),
        "codegen": supported_tasks_mapping(
            "default",
            # "default-with-past",
            "causal-lm",
            # "causal-lm-with-past",
            onnx="CodeGenOnnxConfig",
        ),
        "convbert": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="ConvBertOnnxConfig",
        ),
        "convnext": supported_tasks_mapping(
            "default",
            "image-classification",
            onnx="ConvNextOnnxConfig",
        ),
        "data2vec-text": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="Data2VecTextOnnxConfig",
        ),
        "data2vec-vision": supported_tasks_mapping(
            "default",
            "image-classification",
            # ONNX doesn't support `adaptive_avg_pool2d` yet
            # "semantic-segmentation",
            onnx="Data2VecVisionOnnxConfig",
        ),
        "deberta": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "token-classification",
            "question-answering",
            onnx="DebertaOnnxConfig",
        ),
        "deberta-v2": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="DebertaV2OnnxConfig",
        ),
        "deit": supported_tasks_mapping("default", "image-classification", "masked-im", onnx="DeiTOnnxConfig"),
        "detr": supported_tasks_mapping(
            "default",
            "object-detection",
            "image-segmentation",
            onnx="DetrOnnxConfig",
        ),
        "distilbert": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="DistilBertOnnxConfig",
        ),
        "electra": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for electra
            # "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="ElectraOnnxConfig",
        ),
        "flaubert": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="FlaubertOnnxConfig",
        ),
        "gpt2": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            "token-classification",
            onnx="GPT2OnnxConfig",
        ),
        "gptj": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "question-answering",
            "sequence-classification",
            onnx="GPTJOnnxConfig",
        ),
        "gpt-neo": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            onnx="GPTNeoOnnxConfig",
        ),
        "groupvit": supported_tasks_mapping(
            "default",
            onnx="GroupViTOnnxConfig",
        ),
        "ibert": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="IBertOnnxConfig",
        ),
        "layoutlm": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "token-classification",
            onnx="LayoutLMOnnxConfig",
        ),
        # "layoutlmv2": supported_tasks_mapping(
        #     "default",
        #     "question-answering",
        #     "sequence-classification",
        #     "token-classification",
        #     onnx="LayoutLMv2OnnxConfig",
        # ),
        "layoutlmv3": supported_tasks_mapping(
            "default",
            "question-answering",
            "sequence-classification",
            "token-classification",
            onnx="LayoutLMv3OnnxConfig",
        ),
        "levit": supported_tasks_mapping("default", "image-classification", onnx="LevitOnnxConfig"),
        "longt5": supported_tasks_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx="LongT5OnnxConfig",
        ),
        # "longformer": supported_tasks_mapping(
        #     "default",
        #     "masked-lm",
        #     "multiple-choice",
        #     "question-answering",
        #     "sequence-classification",
        #     "token-classification",
        #     onnx_config_cls="models.longformer.LongformerOnnxConfig",
        # ),
        "marian": supported_tasks_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "causal-lm",
            "causal-lm-with-past",
            onnx="MarianOnnxConfig",
        ),
        "mbart": supported_tasks_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "sequence-classification",
            "question-answering",
            onnx="MBartOnnxConfig",
        ),
        "mobilebert": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="MobileBertOnnxConfig",
        ),
        "mobilevit": supported_tasks_mapping(
            "default",
            "image-classification",
            onnx="MobileViTOnnxConfig",
        ),
        "mobilenet-v1": supported_tasks_mapping(
            "default",
            "image-classification",
            onnx="MobileNetV1OnnxConfig",
        ),
        "mobilenet-v2": supported_tasks_mapping(
            "default",
            "image-classification",
            onnx="MobileNetV2OnnxConfig",
        ),
        "mt5": supported_tasks_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx="MT5OnnxConfig",
        ),
        "m2m-100": supported_tasks_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx="M2M100OnnxConfig",
        ),
        # TODO: owlvit is actually not yet supported in exporters
        # "owlvit": supported_tasks_mapping(
        #     "default",
        #     "zero-shot-object-detection",
        #     onnx="OwlViTOnnxConfig",
        # ),
        "perceiver": supported_tasks_mapping(
            "masked-lm",
            "image-classification",
            "sequence-classification",
            onnx="PerceiverOnnxConfig",
        ),
        "resnet": supported_tasks_mapping(
            "default",
            "image-classification",
            onnx="ResNetOnnxConfig",
        ),
        "roberta": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for roberta
            # "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="RobertaOnnxConfig",
        ),
        "roformer": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for roformer
            # "causal-lm",
            "sequence-classification",
            "token-classification",
            "multiple-choice",
            "question-answering",
            "token-classification",
            onnx="RoFormerOnnxConfig",
        ),
        "segformer": supported_tasks_mapping(
            "default",
            "image-classification",
            "semantic-segmentation",
            onnx="SegformerOnnxConfig",
        ),
        "squeezebert": supported_tasks_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="SqueezeBertOnnxConfig",
        ),
        "t5": supported_tasks_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx="T5OnnxConfig",
        ),
        "unet": supported_tasks_mapping(
            "semantic-segmentation",
            onnx="UNetOnnxConfig",
        ),
        "vae": supported_tasks_mapping(
            "semantic-segmentation",
            onnx="VaeOnnxConfig",
        ),
        "vit": supported_tasks_mapping("default", "image-classification", "masked-im", onnx="ViTOnnxConfig"),
        "whisper": supported_tasks_mapping(
            "default",
            "default-with-past",
            "speech2seq-lm",
            "speech2seq-lm-with-past",
            onnx="WhisperOnnxConfig",
        ),
        "xlm": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for xlm
            # "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="XLMOnnxConfig",
        ),
        "xlm-roberta": supported_tasks_mapping(
            "default",
            "masked-lm",
            # the logic for causal-lm is not supported for xlm-roberta
            # "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="XLMRobertaOnnxConfig",
        ),
        "yolos": supported_tasks_mapping(
            "default",
            "object-detection",
            onnx="YolosOnnxConfig",
        ),
        "swin": supported_tasks_mapping(
            "default",
            "image-classification",
            "masked-im",
            onnx="SwinOnnxConfig",
        ),
    }
    _UNSUPPORTED_CLI_MODEL_TYPE = {"unet", "vae", "clip-text-model"}
    _SUPPORTED_CLI_MODEL_TYPE = set(_SUPPORTED_MODEL_TYPE.keys()) - _UNSUPPORTED_CLI_MODEL_TYPE

    @staticmethod
    def get_supported_tasks_for_model_type(
        model_type: str, exporter: str, model_name: Optional[str] = None
    ) -> TaskNameToExportConfigDict:
        """
        Retrieves the `task -> exporter backend config constructors` map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported tasks for.
            exporter (`str`):
                The name of the exporter.
            model_name (`Optional[str]`, *optional*):
                The name attribute of the model object, only used for the exception message.

        Returns:
            `TaskNameToExportConfigDict`: The dictionary mapping each task to a corresponding `ExportConfig`
            constructor.
        """
        model_type = model_type.lower()
        model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type
        if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet. "
                f"Only {TasksManager._SUPPORTED_CLI_MODEL_TYPE} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        elif exporter not in TasksManager._SUPPORTED_MODEL_TYPE[model_type]:
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet with the {exporter} backend. "
                f"Only {list(TasksManager._SUPPORTED_MODEL_TYPE[model_type].keys())} are supported. "
                f"If you want to support {exporter} please propose a PR or open up an issue."
            )
        else:
            return TasksManager._SUPPORTED_MODEL_TYPE[model_type][exporter]

    @staticmethod
    def format_task(task: str) -> str:
        return task.replace("-with-past", "")

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
    def get_model_class_for_task(task: str, framework: str = "pt") -> Type:
        """
        Attempts to retrieve an AutoModel class from a task name.

        Args:
            task (`str`):
                The task required.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The AutoModel class corresponding to the task.
        """
        task = TasksManager.format_task(task)
        TasksManager._validate_framework_choice(framework)
        if framework == "pt":
            tasks_to_automodel = TasksManager._TASKS_TO_AUTOMODELS
        else:
            tasks_to_automodel = TasksManager._TASKS_TO_TF_AUTOMODELS
        if task not in tasks_to_automodel:
            raise KeyError(
                f"Unknown task: {task}. Possible values are: "
                + ", ".join([f"`{key}` for {tasks_to_automodel[key]}" for key in tasks_to_automodel])
            )

        module = importlib.import_module(TasksManager._TASKS_TO_LIBRARY[task])
        return getattr(module, tasks_to_automodel[task])

    @staticmethod
    def determine_framework(
        model_name_or_path: Union[str, Path], subfolder: str = "", framework: Optional[str] = None
    ) -> str:
        """
        Determines the framework to use for the export.

        The priority is in the following order:
            1. User input via `framework`.
            2. If local checkpoint is provided, use the same framework as the checkpoint.
            3. If model repo, try to infer the framework from the Hub.
            4. If could not infer, use available framework in environment, with priority given to PyTorch.

        Args:
            model_name_or_path (`Union[str, Path]`):
                Can be either the model id of a model repo on the Hugging Face Hub, or a path to a local directory
                containing a model.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            framework (`Optional[str]`, *optional*):
                The framework to use for the export. See above for priority if none provided.

        Returns:
            `str`: The framework to use for the export.

        """
        if framework is not None:
            return framework

        framework_map = {"pt": "PyTorch", "tf": "TensorFlow"}

        full_model_path = Path(model_name_or_path) / subfolder
        if full_model_path.is_dir():
            if (full_model_path / WEIGHTS_NAME).is_file():
                framework = "pt"
            elif (full_model_path / TF2_WEIGHTS_NAME).is_file():
                framework = "tf"
            else:
                raise FileNotFoundError(
                    "Cannot determine framework from given checkpoint location."
                    f" There should be a {WEIGHTS_NAME} for PyTorch"
                    f" or {TF2_WEIGHTS_NAME} for TensorFlow."
                )
            logger.info(f"Local {framework_map[framework]} model found.")
        else:
            if not isinstance(model_name_or_path, str):
                model_name_or_path = str(model_name_or_path)
            try:
                url = huggingface_hub.hf_hub_url(model_name_or_path, WEIGHTS_NAME, subfolder=subfolder)
                huggingface_hub.get_hf_file_metadata(url)
                framework = "pt"
            except Exception:
                pass

            if framework is None:
                try:
                    url = huggingface_hub.hf_hub_url(model_name_or_path, TF2_WEIGHTS_NAME, subfolder=subfolder)
                    huggingface_hub.get_hf_file_metadata(url)
                    framework = "tf"
                except Exception:
                    pass

        if is_torch_available():
            framework = framework or "pt"
        elif is_tf_available():
            framework = framework or "tf"
        else:
            raise EnvironmentError("Neither PyTorch nor TensorFlow found in environment. Cannot export model.")

        logger.info(f"Framework not specified. Using {framework} to export to ONNX.")

        return framework

    @staticmethod
    def infer_task_from_model(model_name_or_path: str, subfolder: str = "", revision: Optional[str] = None) -> str:
        """
        Infers the task from the model repo.

        Args:
            model_name_or_path (`str`):
                The model repo or local path (not supported for now).
            subfolder (`str`, *optional*, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.

        Returns:
            `str`: The task name automatically detected from the model repo.
        """

        tasks_to_automodels = {}
        class_name_prefix = ""
        if is_torch_available():
            tasks_to_automodels = TasksManager._TASKS_TO_AUTOMODELS
        else:
            tasks_to_automodels = TasksManager._TASKS_TO_TF_AUTOMODELS
            class_name_prefix = "TF"

        inferred_task_name = None
        is_local = os.path.isdir(os.path.join(model_name_or_path, subfolder))

        if is_local:
            # TODO: maybe implement that.
            raise RuntimeError("Cannot infer the task from a local directory yet, please specify the task manually.")
        else:
            if subfolder != "":
                raise RuntimeError(
                    "Cannot infer the task from a model repo with a subfolder yet, please specify the task manually."
                )
            model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
            if model_info.library_name == "diffusers":
                # TODO : getattr(model_info, "model_index") defining auto_model_class_name currently set to None
                if "stable-diffusion" in model_info.tags:
                    inferred_task_name = "stable-diffusion"
            else:
                transformers_info = model_info.transformersInfo
                if transformers_info is None or transformers_info.get("auto_model") is None:
                    raise RuntimeError(f"Could not infer the task from the model repo {model_name_or_path}")
                auto_model_class_name = transformers_info["auto_model"]
                if not auto_model_class_name.startswith("TF"):
                    auto_model_class_name = f"{class_name_prefix}{auto_model_class_name}"
                for task_name, class_name_for_task in tasks_to_automodels.items():
                    if class_name_for_task == auto_model_class_name:
                        inferred_task_name = task_name
                        break
        if inferred_task_name is None:
            raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")
        logger.info(f"Automatic task detection to {inferred_task_name}.")
        return inferred_task_name

    @staticmethod
    def get_all_tasks():
        """
        Retrieves all the possible tasks.

        Returns:
            `List`: all the possible tasks.
        """
        tasks = []
        if is_torch_available():
            tasks = list(TasksManager._TASKS_TO_AUTOMODELS.keys())
        else:
            tasks = list(TasksManager._TASKS_TO_TF_AUTOMODELS)
        return tasks

    @staticmethod
    def get_model_from_task(
        task: str,
        model_name_or_path: Union[str, Path],
        subfolder: str = "",
        revision: Optional[str] = None,
        framework: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **model_kwargs
    ) -> Union["PreTrainedModel", "TFPreTrainedModel"]:
        """
        Retrieves a model from its name and the task to be enabled.

        Args:
            task (`str`):
                The task required.
            model_name_or_path (`Union[str, Path]`):
                Can be either the model id of a model repo on the Hugging Face Hub, or a path to a local directory
                containing a model.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            framework (`Optional[str]`, *optional*):
                The framework to use for the export. See `TasksManager.determine_framework` for the priority should
                none be provided.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to pass to the model `.from_pretrained()` method.

        Returns:
            The instance of the model.

        """
        framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)
        if task == "auto":
            task = TasksManager.infer_task_from_model(model_name_or_path, subfolder=subfolder, revision=revision)
        model_class = TasksManager.get_model_class_for_task(task, framework)
        kwargs = {"subfolder": subfolder, "revision": revision, "cache_dir": cache_dir, **model_kwargs}
        try:
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
        return model

    @staticmethod
    def get_exporter_config_constructor(
        exporter: str,
        model: Union["PreTrainedModel", "TFPreTrainedModel"] = None,
        task: str = "default",
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> ExportConfigConstructor:
        """
        Gets the `ExportConfigConstructor` for a model (or alternatively for a model type) and task combination.

        Args:
            exporter (`str`):
                The exporter to use.
            model (`Optional[Union[PreTrainedModel, TFPreTrainedModel]]`, defaults to `None`):
                The instance of the model.
            task (`str`, defaults to `"default"`):
                The task to retrieve the config for.
            model_type (`Optional[str]`, defaults to `None`):
                The model type to retrieve the config for.
            model_name (`Optional[str]`, defaults to `None`):
                The name attribute of the model object, only used for the exception message.

        Returns:
            `ExportConfigConstructor`: The `ExportConfig` constructor for the requested backend.
        """
        if model is None:
            if model_type is None or model_name is None:
                raise ValueError("Either a model_type or model should be provided to retrieve the export config.")

        if model_type is None:
            model_type = getattr(model.config, "model_type", model_type)

            if model_type is None:
                raise ValueError("Model type cannot be inferred. Please provide the model_type for the model!")

            model_type = model_type.replace("_", "-")
            model_name = getattr(model, "name", model_name)

        model_tasks = TasksManager.get_supported_tasks_for_model_type(model_type, exporter, model_name=model_name)
        if task not in model_tasks:
            raise ValueError(
                f"{model_type} doesn't support task {task} for the {exporter} backend."
                f" Supported values are: {model_tasks}"
            )
        return TasksManager._SUPPORTED_MODEL_TYPE[model_type][exporter][task]
