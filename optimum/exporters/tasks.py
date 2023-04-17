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
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import huggingface_hub
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging

from ..utils.import_utils import is_onnx_available


if TYPE_CHECKING:
    import torch

    from .base import ExportConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if not is_torch_available() and not is_tf_available():
    logger.warning(
        "The export tasks are only supported for PyTorch or TensorFlow. You will not be able to export models"
        " without one of these libraries installed."
    )

if is_torch_available():
    from transformers import PreTrainedModel

if is_tf_available():
    from transformers import TFPreTrainedModel

ExportConfigConstructor = Callable[[PretrainedConfig], "ExportConfig"]
TaskNameToExportConfigDict = Dict[str, ExportConfigConstructor]


def is_backend_available(backend):
    backend_availablilty = {
        "onnx": is_onnx_available(),
        "tflite": is_tf_available(),
    }
    return backend_availablilty[backend]


def make_backend_config_constructor_for_task(config_cls: Type, task: str) -> ExportConfigConstructor:
    if "-with-past" in task:
        if not hasattr(config_cls, "with_past"):
            raise ValueError(f"{config_cls} does not support tasks with past.")
        constructor = partial(config_cls.with_past, task=task.replace("-with-past", ""))
    else:
        constructor = partial(config_cls, task=task)
    return constructor


def supported_tasks_mapping(
    *supported_tasks: Union[str, Tuple[str, Tuple[str, ...]]], **exporters: str
) -> Dict[str, TaskNameToExportConfigDict]:
    """
    Generates the mapping between supported tasks and their corresponding `ExportConfig` for a given model, for
    every backend.

    Args:
        supported_tasks (`Tuple[Union[str, Tuple[str, Tuple[str, ...]]]`):
            The names of the supported tasks.
            If some task is supported by only a subset of all the backends, it can be specified as follows:
                ```python
                >>> ("multiple-choice", ("onnx",))
                ```

            The line above means that the multiple-choice task will be supported only by the ONNX backend.

        exporters (`Dict[str, str]`):
            The export backend name -> config class name mapping. For instance:
            ```python
            >>> exporters = {  # doctest: +SKIP
            ...     "onnx": "BertOnnxConfig",
            ...     "tflite": "BertTFLiteConfig",
            ...     ...
            ... }
            ```

    Returns:
        `Dict[str, TaskNameToExportConfigDict]`: The dictionary mapping a task to an `ExportConfig` constructor.
    """
    mapping = {}
    for backend, config_cls_name in exporters.items():
        if is_backend_available(backend):
            config_cls = getattr(
                importlib.import_module(f"optimum.exporters.{backend}.model_configs"), config_cls_name
            )
            mapping[backend] = {}
            for task in supported_tasks:
                if isinstance(task, tuple):
                    task, supported_backends_for_task = task
                    if backend not in supported_backends_for_task:
                        continue
                config_constructor = make_backend_config_constructor_for_task(config_cls, task)
                mapping[backend][task] = config_constructor
    return mapping


def get_automodels_to_tasks(tasks_to_automodel: Dict[str, Union[str, Tuple[str]]]) -> Dict[str, str]:
    """
    Reverses tasks_to_automodel while flattening the case where the same task maps to several
    auto classes (e.g. automatic-speech-recognition).
    """
    automodels_to_tasks = {}
    for task, automodels in tasks_to_automodel.items():
        if isinstance(automodels, str):
            automodels_to_tasks[automodels] = task
        else:
            automodels_to_tasks.update({automodel_name: task for automodel_name in automodels})

    return automodels_to_tasks


class TasksManager:
    """
    Handles the `task name -> model class` and `architecture -> configuration` mappings.
    """

    _TASKS_TO_AUTOMODELS = {}
    _TASKS_TO_TF_AUTOMODELS = {}
    if is_torch_available():
        # Refer to https://huggingface.co/datasets/huggingface/transformers-metadata/blob/main/pipeline_tags.json
        # In case the same task (pipeline tag) may map to several loading classes, we use a tuple and the
        # auto-class _model_mapping to determine the right one.
        _TASKS_TO_AUTOMODELS = {
            "conversational": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
            "feature-extraction": "AutoModel",
            "fill-mask": "AutoModelForMaskedLM",
            "text-generation": "AutoModelForCausalLM",
            "text2text-generation": "AutoModelForSeq2SeqLM",
            "text-classification": "AutoModelForSequenceClassification",
            "token-classification": "AutoModelForTokenClassification",
            "multiple-choice": "AutoModelForMultipleChoice",
            "object-detection": "AutoModelForObjectDetection",
            "question-answering": "AutoModelForQuestionAnswering",
            "image-classification": "AutoModelForImageClassification",
            "image-segmentation": "AutoModelForImageSegmentation",
            "masked-im": "AutoModelForMaskedImageModeling",
            "semantic-segmentation": "AutoModelForSemanticSegmentation",
            "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", "AutoModelForCTC"),
            "audio-classification": "AutoModelForAudioClassification",
            "audio-frame-classification": "AutoModelForAudioFrameClassification",
            "audio-xvector": "AutoModelForAudioXVector",
            "image-to-text": "AutoModelForVision2Seq",
            "stable-diffusion": "StableDiffusionPipeline",
            "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
            "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
        }
    if is_tf_available():
        _TASKS_TO_TF_AUTOMODELS = {
            "conversational": ("TFAutoModelForCausalLM", "TFAutoModelForSeq2SeqLM"),
            "feature-extraction": "TFAutoModel",
            "fill-mask": "TFAutoModelForMaskedLM",
            "text-generation": "TFAutoModelForCausalLM",
            "image-classification": "TFAutoModelForImageClassification",
            "text2text-generation": "TFAutoModelForSeq2SeqLM",
            "text-classification": "TFAutoModelForSequenceClassification",
            "token-classification": "TFAutoModelForTokenClassification",
            "multiple-choice": "TFAutoModelForMultipleChoice",
            "object-detection": "TFAutoModelForObjectDetection",
            "question-answering": "TFAutoModelForQuestionAnswering",
            "image-segmentation": "TFAutoModelForImageSegmentation",
            "masked-im": "TFAutoModelForMaskedImageModeling",
            "semantic-segmentation": "TFAutoModelForSemanticSegmentation",
            "automatic-speech-recognition": "TFAutoModelForSpeechSeq2Seq",
            "audio-classification": "TFAutoModelForAudioClassification",
            "audio-frame-classification": "TFAutoModelForAudioFrameClassification",
            "audio-xvector": "TFAutoModelForAudioXVector",
            "image-to-text": "TFAutoModelForVision2Seq",
            "zero-shot-image-classification": "TFAutoModelForZeroShotImageClassification",
            "zero-shot-object-detection": "TFAutoModelForZeroShotObjectDetection",
        }

    _SYNONYM_TASK_MAP = {
        "sequence-classification": "text-classification",
        "causal-lm": "text-generation",
        "causal-lm-with-past": "text-generation-with-past",
        "seq2seq-lm": "text2text-generation",
        "seq2seq-lm-with-past": "text2text-generation-with-past",
        "speech2seq-lm": "automatic-speech-recognition",
        "speech2seq-lm-with-past": "automatic-speech-recognition-with-past",
        "masked-lm": "fill-mask",
        "vision2seq-lm": "image-to-text",
        "default": "feature-extraction",
        "default-with-past": "feature-extraction-with-past",
        "audio-ctc": "automatic-speech-recognition",
        "translation": "text2text-generation",
        "summarization": "text2text-generation",
        "zero-shot-classification": "text-classification",
    }

    # Reverse dictionaries str -> str, where several automodels may map to the same task
    _AUTOMODELS_TO_TASKS = get_automodels_to_tasks(_TASKS_TO_AUTOMODELS)
    _TF_AUTOMODELS_TO_TASKS = get_automodels_to_tasks(_TASKS_TO_TF_AUTOMODELS)

    _CUSTOM_CLASSES = {
        ("pt", "pix2struct", "image-to-text"): ("transformers", "Pix2StructForConditionalGeneration"),
        ("pt", "visual-bert", "question-answering"): ("transformers", "VisualBertForQuestionAnswering"),
    }

    _TASKS_TO_LIBRARY = {
        "conversational": "transformers",
        "feature-extraction": "transformers",
        "fill-mask": "transformers",
        "text-generation": "transformers",
        "text2text-generation": "transformers",
        "text-classification": "transformers",
        "token-classification": "transformers",
        "translation": "transformers",
        "multiple-choice": "transformers",
        "object-detection": "transformers",
        "question-answering": "transformers",
        "image-classification": "transformers",
        "image-segmentation": "transformers",
        "masked-im": "transformers",
        "semantic-segmentation": "transformers",
        "automatic-speech-recognition": "transformers",
        "audio-classification": "transformers",
        "audio-frame-classification": "transformers",
        "audio-xvector": "transformers",
        "image-to-text": "transformers",
        "stable-diffusion": "diffusers",
        "summarization": "transformers",
        "zero-shot-classification": "transformers",
        "zero-shot-image-classification": "transformers",
        "zero-shot-object-detection": "transformers",
    }

    # TODO: some models here support text-generation export but are not supported in ORTModelForCausalLM
    # Set of model topologies we support associated to the tasks supported by each topology and the factory
    _SUPPORTED_MODEL_TYPE = {
        "audio-spectrogram-transformer": supported_tasks_mapping(
            "feature-extraction",
            "audio-classification",
            onnx="ASTOnnxConfig",
        ),
        "albert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="AlbertOnnxConfig",
            tflite="AlbertTFLiteConfig",
        ),
        "bart": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            "text-classification",
            "question-answering",
            onnx="BartOnnxConfig",
        ),
        # BEiT cannot be used with the masked image modeling autoclass, so this task is excluded here
        "beit": supported_tasks_mapping("feature-extraction", "image-classification", onnx="BeitOnnxConfig"),
        "bert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            # the logic for text-generation is not supported for BERT
            # "text-generation",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="BertOnnxConfig",
            tflite="BertTFLiteConfig",
        ),
        # For big-bird and bigbird-pegasus being unsupported, refer to model_configs.py
        # "big-bird": supported_tasks_mapping(
        #     "feature-extraction",
        #     "fill-mask",
        #     # the logic for text-generation is not supported for big-bird
        #     # "text-generation",
        #     "text-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx="BigBirdOnnxConfig",
        #     # TODO: check model_config.py to know why it cannot be enabled yet.
        #     # tflite="BigBirdTFLiteConfig",
        # ),
        # "bigbird-pegasus": supported_tasks_mapping(
        #     "feature-extraction",
        #     "feature-extraction-with-past",
        #     "text-generation",
        #     "text-generation-with-past",
        #     "text2text-generation",
        #     "text2text-generation-with-past",
        #     "text-classification",
        #     "question-answering",
        #     onnx="BigBirdPegasusOnnxConfig",
        # ),
        "blenderbot": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            onnx="BlenderbotOnnxConfig",
        ),
        "blenderbot-small": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            onnx="BlenderbotSmallOnnxConfig",
        ),
        "bloom": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text-classification",
            "token-classification",
            onnx="BloomOnnxConfig",
        ),
        "camembert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            # the logic for text-generation is not supported for camembert
            # "text-generation",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="CamembertOnnxConfig",
            tflite="CamembertTFLiteConfig",
        ),
        "clip": supported_tasks_mapping(
            "feature-extraction",
            "zero-shot-image-classification",
            onnx="CLIPOnnxConfig",
        ),
        "clip-text-model": supported_tasks_mapping(
            "feature-extraction",
            onnx="CLIPTextOnnxConfig",
        ),
        "codegen": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            onnx="CodeGenOnnxConfig",
        ),
        "convbert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="ConvBertOnnxConfig",
            tflite="ConvBertTFLiteConfig",
        ),
        "convnext": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            onnx="ConvNextOnnxConfig",
        ),
        "data2vec-text": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="Data2VecTextOnnxConfig",
        ),
        "data2vec-vision": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            # ONNX doesn't support `adaptive_avg_pool2d` yet
            # "semantic-segmentation",
            onnx="Data2VecVisionOnnxConfig",
        ),
        "data2vec-audio": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            "audio-frame-classification",
            "audio-xvector",
            onnx="Data2VecAudioOnnxConfig",
        ),
        "deberta": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "token-classification",
            "question-answering",
            onnx="DebertaOnnxConfig",
            tflite="DebertaTFLiteConfig",
        ),
        "deberta-v2": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            ("multiple-choice", ("onnx",)),
            "token-classification",
            "question-answering",
            onnx="DebertaV2OnnxConfig",
            tflite="DebertaV2TFLiteConfig",
        ),
        "deit": supported_tasks_mapping(
            "feature-extraction", "image-classification", "masked-im", onnx="DeiTOnnxConfig"
        ),
        "detr": supported_tasks_mapping(
            "feature-extraction",
            "object-detection",
            "image-segmentation",
            onnx="DetrOnnxConfig",
        ),
        "distilbert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="DistilBertOnnxConfig",
            tflite="DistilBertTFLiteConfig",
        ),
        "donut-swin": supported_tasks_mapping(
            "feature-extraction",
            onnx="DonutSwinOnnxConfig",
        ),
        "electra": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            # the logic for text-generation is not supported for electra
            # "text-generation",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="ElectraOnnxConfig",
            tflite="ElectraTFLiteConfig",
        ),
        "flaubert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="FlaubertOnnxConfig",
            tflite="FlaubertTFLiteConfig",
        ),
        "gpt2": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text-classification",
            "token-classification",
            onnx="GPT2OnnxConfig",
        ),
        "gptj": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "question-answering",
            "text-classification",
            onnx="GPTJOnnxConfig",
        ),
        "gpt-neo": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text-classification",
            onnx="GPTNeoOnnxConfig",
        ),
        "gpt-neox": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            onnx="GPTNeoXOnnxConfig",
        ),
        "groupvit": supported_tasks_mapping(
            "feature-extraction",
            onnx="GroupViTOnnxConfig",
        ),
        "hubert": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            onnx="HubertOnnxConfig",
        ),
        "ibert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="IBertOnnxConfig",
        ),
        "imagegpt": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            onnx="ImageGPTOnnxConfig",
        ),
        "layoutlm": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "token-classification",
            onnx="LayoutLMOnnxConfig",
        ),
        # "layoutlmv2": supported_tasks_mapping(
        #     "feature-extraction",
        #     "question-answering",
        #     "text-classification",
        #     "token-classification",
        #     onnx="LayoutLMv2OnnxConfig",
        # ),
        "layoutlmv3": supported_tasks_mapping(
            "feature-extraction",
            "question-answering",
            "text-classification",
            "token-classification",
            onnx="LayoutLMv3OnnxConfig",
        ),
        "levit": supported_tasks_mapping("feature-extraction", "image-classification", onnx="LevitOnnxConfig"),
        "longt5": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            onnx="LongT5OnnxConfig",
        ),
        # "longformer": supported_tasks_mapping(
        #     "feature-extraction",
        #     "fill-mask",
        #     "multiple-choice",
        #     "question-answering",
        #     "text-classification",
        #     "token-classification",
        #     onnx_config_cls="models.longformer.LongformerOnnxConfig",
        # ),
        "marian": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            "text-generation",
            "text-generation-with-past",
            onnx="MarianOnnxConfig",
        ),
        "mbart": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            "text-classification",
            "question-answering",
            onnx="MBartOnnxConfig",
        ),
        # TODO: enable once the missing operator is supported.
        # "mctct": supported_tasks_mapping(
        #     "feature-extraction",
        #     "automatic-speech-recognition",
        #     onnx="MCTCTOnnxConfig",
        # ),
        "mobilebert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="MobileBertOnnxConfig",
            tflite="MobileBertTFLiteConfig",
        ),
        "mobilevit": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            onnx="MobileViTOnnxConfig",
        ),
        "mobilenet-v1": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            onnx="MobileNetV1OnnxConfig",
        ),
        "mobilenet-v2": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            onnx="MobileNetV2OnnxConfig",
        ),
        "mpnet": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="MPNetOnnxConfig",
            tflite="MPNetTFLiteConfig",
        ),
        "mt5": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            onnx="MT5OnnxConfig",
        ),
        "m2m-100": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            onnx="M2M100OnnxConfig",
        ),
        "nystromformer": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "multiple-choice",
            "question-answering",
            "text-classification",
            "token-classification",
            onnx="NystromformerOnnxConfig",
        ),
        # TODO: owlvit cannot be exported yet, check model_config.py to know why.
        # "owlvit": supported_tasks_mapping(
        #     "feature-extraction",
        #     "zero-shot-object-detection",
        #     onnx="OwlViTOnnxConfig",
        # ),
        "opt": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "question-answering",
            "text-classification",
            onnx="OPTOnnxConfig",
        ),
        "llama": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text-classification",
            onnx="LlamaOnnxConfig",
        ),
        "pegasus": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            onnx="PegasusOnnxConfig",
        ),
        "perceiver": supported_tasks_mapping(
            "fill-mask",
            "image-classification",
            "text-classification",
            onnx="PerceiverOnnxConfig",
        ),
        "poolformer": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            onnx="PoolFormerOnnxConfig",
        ),
        "regnet": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            onnx="RegNetOnnxConfig",
        ),
        "resnet": supported_tasks_mapping(
            "feature-extraction", "image-classification", onnx="ResNetOnnxConfig", tflite="ResNetTFLiteConfig"
        ),
        "roberta": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            # the logic for text-generation is not supported for roberta
            # "text-generation",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="RobertaOnnxConfig",
            tflite="RobertaTFLiteConfig",
        ),
        "roformer": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            # the logic for text-generation is not supported for roformer
            # "text-generation",
            "text-classification",
            "token-classification",
            "multiple-choice",
            "question-answering",
            "token-classification",
            onnx="RoFormerOnnxConfig",
            tflite="RoFormerTFLiteConfig",
        ),
        "segformer": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            "semantic-segmentation",
            onnx="SegformerOnnxConfig",
        ),
        "sew": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            onnx="SEWOnnxConfig",
        ),
        "sew-d": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            onnx="SEWDOnnxConfig",
        ),
        "speech-to-text": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "automatic-speech-recognition",
            "automatic-speech-recognition-with-past",
            onnx="Speech2TextOnnxConfig",
        ),
        "splinter": supported_tasks_mapping(
            "feature-extraction",
            "question-answering",
            onnx="SplinterOnnxConfig",
        ),
        "squeezebert": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="SqueezeBertOnnxConfig",
        ),
        "swin": supported_tasks_mapping(
            "feature-extraction",
            "image-classification",
            "masked-im",
            onnx="SwinOnnxConfig",
        ),
        "t5": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            onnx="T5OnnxConfig",
        ),
        "trocr": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "image-to-text",
            "image-to-text-with-past",
            onnx="TrOCROnnxConfig",
        ),
        "unet": supported_tasks_mapping(
            "semantic-segmentation",
            onnx="UNetOnnxConfig",
        ),
        "unispeech": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            onnx="UniSpeechOnnxConfig",
        ),
        "unispeech-sat": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            "audio-frame-classification",
            "audio-xvector",
            onnx="UniSpeechSATOnnxConfig",
        ),
        "vae-encoder": supported_tasks_mapping(
            "semantic-segmentation",
            onnx="VaeEncoderOnnxConfig",
        ),
        "vae-decoder": supported_tasks_mapping(
            "semantic-segmentation",
            onnx="VaeDecoderOnnxConfig",
        ),
        "vision-encoder-decoder": supported_tasks_mapping(
            "image-to-text",
            "image-to-text-with-past",
            onnx="VisionEncoderDecoderOnnxConfig",
        ),
        "vit": supported_tasks_mapping(
            "feature-extraction", "image-classification", "masked-im", onnx="ViTOnnxConfig"
        ),
        "wavlm": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            "audio-frame-classification",
            "audio-xvector",
            onnx="WavLMOnnxConfig",
        ),
        "wav2vec2": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            "audio-frame-classification",
            "audio-xvector",
            onnx="Wav2Vec2OnnxConfig",
        ),
        "wav2vec2-conformer": supported_tasks_mapping(
            "feature-extraction",
            "automatic-speech-recognition",
            "audio-classification",
            "audio-frame-classification",
            "audio-xvector",
            onnx="Wav2Vec2ConformerOnnxConfig",
        ),
        "whisper": supported_tasks_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "automatic-speech-recognition",
            "automatic-speech-recognition-with-past",
            onnx="WhisperOnnxConfig",
        ),
        "xlm": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            # the logic for text-generation is not supported for xlm
            # "text-generation",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="XLMOnnxConfig",
            tflite="XLMTFLiteConfig",
        ),
        "xlm-roberta": supported_tasks_mapping(
            "feature-extraction",
            "fill-mask",
            # the logic for text-generation is not supported for xlm-roberta
            # "text-generation",
            "text-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="XLMRobertaOnnxConfig",
            tflite="XLMRobertaTFLiteConfig",
        ),
        "yolos": supported_tasks_mapping(
            "feature-extraction",
            "object-detection",
            onnx="YolosOnnxConfig",
        ),
    }
    _UNSUPPORTED_CLI_MODEL_TYPE = {"unet", "vae-encoder", "vae-decoder", "clip-text-model", "trocr"}
    _SUPPORTED_CLI_MODEL_TYPE = set(_SUPPORTED_MODEL_TYPE.keys()) - _UNSUPPORTED_CLI_MODEL_TYPE

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

        def wrapper(model_type: str, *supported_tasks: str) -> Callable[[Type], Type]:
            def decorator(config_cls: Type) -> Type:
                mapping = cls._SUPPORTED_MODEL_TYPE.get(model_type, {})
                mapping_backend = mapping.get(backend, {})
                for task in supported_tasks:
                    if task not in cls._TASKS_TO_LIBRARY:
                        known_tasks = ", ".join(cls._TASKS_TO_LIBRARY.keys())
                        raise ValueError(
                            f'The TasksManager does not know the task called "{task}", known tasks: {known_tasks}.'
                        )
                    if not overwrite_existing and task in mapping_backend:
                        continue
                    mapping_backend[task] = make_backend_config_constructor_for_task(config_cls, task)
                mapping[backend] = mapping_backend
                cls._SUPPORTED_MODEL_TYPE[model_type] = mapping
                return config_cls

            return decorator

        return wrapper

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
            model_name (`Optional[str]`, defaults to `None`):
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
    def get_supported_model_type_for_task(task: str, exporter: str) -> List[str]:
        """
        Returns the list of supported architectures by the exporter for a given task.
        """
        return [
            model_type.replace("-", "_")
            for model_type in TasksManager._SUPPORTED_MODEL_TYPE
            if task in TasksManager._SUPPORTED_MODEL_TYPE[model_type][exporter]
        ]

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
        task: str, framework: str = "pt", model_type: Optional[str] = None, model_class_name: Optional[str] = None
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
                tasks_to_automodel = TasksManager._TASKS_TO_AUTOMODELS
            else:
                tasks_to_automodel = TasksManager._TASKS_TO_TF_AUTOMODELS

            library = TasksManager._TASKS_TO_LIBRARY[task]
            loaded_library = importlib.import_module(library)

            if model_class_name is None:
                if task not in tasks_to_automodel:
                    raise KeyError(
                        f"Unknown task: {task}. Possible values are: "
                        + ", ".join([f"`{key}` for {tasks_to_automodel[key]}" for key in tasks_to_automodel])
                    )

                if isinstance(tasks_to_automodel[task], str):
                    model_class_name = tasks_to_automodel[task]
                else:
                    # automatic-speech-recognition case, which may map to several auto class
                    if library == "transformers":
                        if model_type is None:
                            logger.warning(
                                f"No model type passed for the task {task}, that may be mapped to several loading"
                                f" classes ({tasks_to_automodel[task]}). Defaulting to {tasks_to_automodel[task][0]}"
                                " to load the model."
                            )
                            model_class_name = tasks_to_automodel[task][0]
                        else:
                            for autoclass_name in tasks_to_automodel[task]:
                                module = getattr(loaded_library, autoclass_name)
                                # TODO: we must really get rid of this - and _ mess
                                if (
                                    model_type in module._model_mapping._model_mapping
                                    or model_type.replace("-", "_") in module._model_mapping._model_mapping
                                ):
                                    model_class_name = autoclass_name
                                    break

                            if model_class_name is None:
                                raise ValueError(
                                    f"Unrecognized configuration classes {tasks_to_automodel[task]} do not match"
                                    f" with the model type {model_type} and task {task}."
                                )
                    else:
                        raise NotImplementedError(
                            "For library other than transformers, the _TASKS_TO_AUTOMODELS mapping should be one to one."
                        )

            return getattr(loaded_library, model_class_name)

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

        full_model_path = Path(model_name_or_path) / subfolder
        if full_model_path.is_dir():
            all_files = [
                os.path.relpath(os.path.join(dirpath, file), full_model_path)
                for dirpath, _, filenames in os.walk(full_model_path)
                for file in filenames
            ]
        else:
            if not isinstance(model_name_or_path, str):
                model_name_or_path = str(model_name_or_path)
            all_files = huggingface_hub.list_repo_files(model_name_or_path, repo_type="model")
            if subfolder != "":
                all_files = [file[len(subfolder) + 1 :] for file in all_files if file.startswith(subfolder)]

        weight_name = Path(WEIGHTS_NAME).stem
        weight_extension = Path(WEIGHTS_NAME).suffix
        is_pt_weight_file = [file.startswith(weight_name) and file.endswith(weight_extension) for file in all_files]

        weight_name = Path(TF2_WEIGHTS_NAME).stem
        weight_extension = Path(TF2_WEIGHTS_NAME).suffix
        is_tf_weight_file = [file.startswith(weight_name) and file.endswith(weight_extension) for file in all_files]

        if any(is_pt_weight_file):
            framework = "pt"
        elif any(is_tf_weight_file):
            framework = "tf"
        elif "model_index.json" in all_files and any(file.endswith(Path(WEIGHTS_NAME).suffix) for file in all_files):
            # stable diffusion case
            framework = "pt"
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

        logger.info(f"Framework not specified. Using {framework} to export to ONNX.")

        return framework

    @classmethod
    def _infer_task_from_model_or_model_class(
        cls, model: Optional[Union["PreTrainedModel", "TFPreTrainedModel"]] = None, model_class: Optional[Type] = None
    ) -> str:
        if model is not None and model_class is not None:
            raise ValueError("Either a model or a model class must be provided, but both were given here.")
        if model is None and model_class is None:
            raise ValueError("Either a model or a model class must be provided, but none were given here.")
        target_name = model.__class__.__name__ if model is not None else model_class.__name__
        task_name = None
        iterable = (cls._AUTOMODELS_TO_TASKS.items(), cls._TF_AUTOMODELS_TO_TASKS.items())
        pt_auto_module = importlib.import_module("transformers.models.auto.modeling_auto")
        tf_auto_module = importlib.import_module("transformers.models.auto.modeling_tf_auto")
        for auto_cls_name, task in itertools.chain.from_iterable(iterable):
            if any(
                (
                    target_name.startswith("Auto"),
                    target_name.startswith("TFAuto"),
                    target_name == "StableDiffusionPipeline",
                )
            ):
                if target_name == auto_cls_name:
                    task_name = task
                    break

                continue

            module = tf_auto_module if auto_cls_name.startswith("TF") else pt_auto_module
            # getattr(module, auto_cls_name)._model_mapping is a _LazyMapping, it also has an attribute called
            # "_model_mapping" that is what we want here: class names and not actual classes.
            auto_cls = getattr(module, auto_cls_name, None)
            # This is the case for StableDiffusionPipeline for instance.
            if auto_cls is None:
                continue
            model_mapping = auto_cls._model_mapping._model_mapping
            if target_name in model_mapping.values():
                task_name = task
                break
        if task_name is None:
            raise ValueError(f"Could not infer the task name for {target_name}.")
        return task_name

    @classmethod
    def _infer_task_from_model_name_or_path(
        cls, model_name_or_path: str, subfolder: str = "", revision: Optional[str] = None
    ) -> str:
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
                pipeline_tag = getattr(model_info, "pipeline_tag", None)
                # conversational is not a supported task per se, just an alias that may map to
                # text-generaton or text2text-generation
                if pipeline_tag is not None and pipeline_tag != "conversational":
                    inferred_task_name = TasksManager.map_from_synonym(model_info.pipeline_tag)
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
        return inferred_task_name

    @classmethod
    def infer_task_from_model(
        cls,
        model: Union[str, "PreTrainedModel", "TFPreTrainedModel", Type],
        subfolder: str = "",
        revision: Optional[str] = None,
    ) -> str:
        """
        Infers the task from the model repo.

        Args:
            model (`str`):
                The model to infer the task from. This can either be the name of a repo on the HuggingFace Hub, an
                instance of a model, or a model class.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.

        Returns:
            `str`: The task name automatically detected from the model repo.
        """
        is_torch_pretrained_model = is_torch_available() and isinstance(model, PreTrainedModel)
        is_tf_pretrained_model = is_tf_available() and isinstance(model, TFPreTrainedModel)
        task = None
        if isinstance(model, str):
            task = cls._infer_task_from_model_name_or_path(model, subfolder=subfolder, revision=revision)
        elif is_torch_pretrained_model or is_tf_pretrained_model:
            task = cls._infer_task_from_model_or_model_class(model=model)
        elif inspect.isclass(model):
            task = cls._infer_task_from_model_or_model_class(model_class=model)

        if task is None:
            raise ValueError(f"Could not infer the task from {model}.")

        return task

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
        torch_dtype: Optional["torch.dtype"] = None,
        **model_kwargs,
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
            torch_dtype (`Optional[torch.dtype]`, defaults to `None`):
                Data type to load the model on. PyTorch-only argument.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to pass to the model `.from_pretrained()` method.

        Returns:
            The instance of the model.

        """
        framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)

        original_task = task
        if task == "auto":
            task = TasksManager.infer_task_from_model(model_name_or_path, subfolder=subfolder, revision=revision)

        model_type = None
        model_class_name = None
        if TasksManager._TASKS_TO_LIBRARY[task.replace("-with-past", "")] == "transformers":
            # TODO: if automatic-speech-recognition is passed as task, it may map to several
            # different auto class (AutoModelForSpeechSeq2Seq or AutoModelForCTC),
            # depending on the model type
            if original_task in ["auto", "automatic-speech-recognition"]:
                config = AutoConfig.from_pretrained(model_name_or_path)
                model_type = config.model_type.replace("_", "-")
                if original_task == "auto" and config.architectures is not None:
                    model_class_name = config.architectures[0]

        model_class = TasksManager.get_model_class_for_task(
            task, framework, model_type=model_type, model_class_name=model_class_name
        )

        kwargs = {"subfolder": subfolder, "revision": revision, "cache_dir": cache_dir, **model_kwargs}
        try:
            if framework == "pt":
                kwargs["torch_dtype"] = torch_dtype
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
        model: Optional[Union["PreTrainedModel", "TFPreTrainedModel"]] = None,
        task: str = "feature-extraction",
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        exporter_config_kwargs: Optional[Dict[str, Any]] = None,
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
            exporter_config_kwargs(`Optional[Dict[str, Any]]`, defaults to `None`):
                Arguments that will be passed to the exporter config class when building the config constructor.

        Returns:
            `ExportConfigConstructor`: The `ExportConfig` constructor for the requested backend.
        """
        if model is None and model_type is None:
            raise ValueError("Either a model_type or model should be provided to retrieve the export config.")

        if model_type is None:
            model_type = getattr(model.config, "model_type", model_type)

            if model_type is None:
                raise ValueError("Model type cannot be inferred. Please provide the model_type for the model!")

            model_type = model_type.replace("_", "-")
            model_name = getattr(model, "name", model_name)

        model_tasks = TasksManager.get_supported_tasks_for_model_type(model_type, exporter, model_name=model_name)

        task = TasksManager.map_from_synonym(task)
        if task not in model_tasks:
            raise ValueError(
                f"{model_type} doesn't support task {task} for the {exporter} backend."
                f" Supported tasks are: {', '.join(model_tasks.keys())}."
            )

        exporter_config_constructor = TasksManager._SUPPORTED_MODEL_TYPE[model_type][exporter][task]
        if exporter_config_kwargs is not None:
            exporter_config_constructor = partial(exporter_config_constructor, **exporter_config_kwargs)

        return exporter_config_constructor
