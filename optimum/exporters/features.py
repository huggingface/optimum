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
"""Model export features manager."""

import importlib
import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union

from transformers import PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging

if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel
    from .base import ExportConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_available():
    from transformers.models.auto import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedImageModeling,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSemanticSegmentation,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )
if is_tf_available():
    from transformers.models.auto import (
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForMultipleChoice,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSemanticSegmentation,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
    )
if not is_torch_available() and not is_tf_available():
    logger.warning(
        "The export features are only supported for PyTorch or TensorFlow. You will not be able to export models"
        " without one of these libraries installed."
    )

ExportConfigConstructor = Callable[[PretrainedConfig], "ExportConfig"]
FeatureNameToExportConfigDict = Dict[str, ExportConfigConstructor]


def supported_features_mapping(
    *supported_features: str, **exporters: str
) -> Dict[str, FeatureNameToExportConfigDict]:
    """
    Generate the mapping between supported the features and their corresponding `ExportConfig` for a given model, for
    every backend.

    Args:
        supported_features (`Tuple[str]`):
        The names of the supported features.
        exporters (`Dict[str, str]`):
            The export backend name -> config class name mapping. For instance, for instance:
            ```
            kwargs = {
                "onnx": "BertOnnxConfig",
                "tflite": "BertTFLiteConfig",
                ...
            }
            ```
    Returns:
        `Dict[str, FeatureNameToExportConfigDict]`: The dictionary mapping a feature to an `ExportConfig` constructor.
    """
    mapping = {}
    for backend, config_cls_name in exporters.items():
        config_cls = getattr(importlib.import_module(f"optimum.exporters.{backend}.model_configs"), config_cls_name)
        mapping[backend] = {}
        for feature in supported_features:
            if "-with-past" in feature:
                task = feature.replace("-with-past", "")
                mapping[backend][feature] = partial(config_cls.with_past, task=task)
            else:
                mapping[backend][feature] = partial(config_cls, task=feature)

    return mapping


class FeaturesManager:
    _TASKS_TO_AUTOMODELS = {}
    _TASKS_TO_TF_AUTOMODELS = {}
    if is_torch_available():
        _TASKS_TO_AUTOMODELS = {
            "default": AutoModel,
            "masked-lm": AutoModelForMaskedLM,
            "causal-lm": AutoModelForCausalLM,
            "seq2seq-lm": AutoModelForSeq2SeqLM,
            "sequence-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "multiple-choice": AutoModelForMultipleChoice,
            "object-detection": AutoModelForObjectDetection,
            "question-answering": AutoModelForQuestionAnswering,
            "image-classification": AutoModelForImageClassification,
            "image-segmentation": AutoModelForImageSegmentation,
            "masked-im": AutoModelForMaskedImageModeling,
            "semantic-segmentation": AutoModelForSemanticSegmentation,
        }
    if is_tf_available():
        _TASKS_TO_TF_AUTOMODELS = {
            "default": TFAutoModel,
            "masked-lm": TFAutoModelForMaskedLM,
            "causal-lm": TFAutoModelForCausalLM,
            "seq2seq-lm": TFAutoModelForSeq2SeqLM,
            "sequence-classification": TFAutoModelForSequenceClassification,
            "token-classification": TFAutoModelForTokenClassification,
            "multiple-choice": TFAutoModelForMultipleChoice,
            "question-answering": TFAutoModelForQuestionAnswering,
            "semantic-segmentation": TFAutoModelForSemanticSegmentation,
        }

    # Set of model topologies we support associated to the features supported by each topology and the factory
    _SUPPORTED_MODEL_TYPE = {
        # "albert": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.albert.AlbertOnnxConfig",
        # ),
        # "bart": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     "sequence-classification",
        #     "question-answering",
        #     onnx_config_cls="models.bart.BartOnnxConfig",
        # ),
        # BEiT cannot be used with the masked image modeling autoclass, so this feature is excluded here
        # "beit": supported_features_mapping(
        #     "default", "image-classification", onnx_config_cls="models.beit.BeitOnnxConfig"
        # ),
        "bert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="BertOnnxConfig",
        ),
        # "big-bird": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "causal-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.big_bird.BigBirdOnnxConfig",
        # ),
        # "bigbird-pegasus": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     "sequence-classification",
        #     "question-answering",
        #     onnx_config_cls="models.bigbird_pegasus.BigBirdPegasusOnnxConfig",
        # ),
        # "blenderbot": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     onnx_config_cls="models.blenderbot.BlenderbotOnnxConfig",
        # ),
        # "blenderbot-small": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     onnx_config_cls="models.blenderbot_small.BlenderbotSmallOnnxConfig",
        # ),
        # "bloom": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     "sequence-classification",
        #     "token-classification",
        #     onnx_config_cls="models.bloom.BloomOnnxConfig",
        # ),
        # "camembert": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "causal-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.camembert.CamembertOnnxConfig",
        # ),
        # "clip": supported_features_mapping(
        #     "default",
        #     onnx_config_cls="models.clip.CLIPOnnxConfig",
        # ),
        # "codegen": supported_features_mapping(
        #     "default",
        #     "causal-lm",
        #     onnx_config_cls="models.codegen.CodeGenOnnxConfig",
        # ),
        # "convbert": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.convbert.ConvBertOnnxConfig",
        # ),
        # "convnext": supported_features_mapping(
        #     "default",
        #     "image-classification",
        #     onnx_config_cls="models.convnext.ConvNextOnnxConfig",
        # ),
        # "data2vec-text": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.data2vec.Data2VecTextOnnxConfig",
        # ),
        # "data2vec-vision": supported_features_mapping(
        #     "default",
        #     "image-classification",
        #     # ONNX doesn't support `adaptive_avg_pool2d` yet
        #     # "semantic-segmentation",
        #     onnx_config_cls="models.data2vec.Data2VecVisionOnnxConfig",
        # ),
        "deberta": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "token-classification",
            "question-answering",
            onnx="DebertaOnnxConfig",
        ),
        "deberta-v2": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="DebertaV2OnnxConfig",
        ),
        # "deit": supported_features_mapping(
        #     "default", "image-classification", "masked-im", onnx_config_cls="models.deit.DeiTOnnxConfig"
        # ),
        # "detr": supported_features_mapping(
        #     "default",
        #     "object-detection",
        #     "image-segmentation",
        #     onnx_config_cls="models.detr.DetrOnnxConfig",
        # ),
        "distilbert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="DistilBertOnnxConfig",
        ),
        # "electra": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "causal-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.electra.ElectraOnnxConfig",
        # ),
        "flaubert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="FlaubertOnnxConfig",
        ),
        "gpt2": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            "token-classification",
            onnx="GPT2OnnxConfig",
        ),
        # "gptj": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     "question-answering",
        #     "sequence-classification",
        #     onnx_config_cls="models.gptj.GPTJOnnxConfig",
        # ),
        "gpt-neo": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            onnx="GPTNeoOnnxConfig",
        ),
        # "groupvit": supported_features_mapping(
        #     "default",
        #     onnx_config_cls="models.groupvit.GroupViTOnnxConfig",
        # ),
        # "ibert": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.ibert.IBertOnnxConfig",
        # ),
        # "layoutlm": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "sequence-classification",
        #     "token-classification",
        #     onnx_config_cls="models.layoutlm.LayoutLMOnnxConfig",
        # ),
        # "layoutlmv3": supported_features_mapping(
        #     "default",
        #     "question-answering",
        #     "sequence-classification",
        #     "token-classification",
        #     onnx_config_cls="models.layoutlmv3.LayoutLMv3OnnxConfig",
        # ),
        # "levit": supported_features_mapping(
        #     "default", "image-classification", onnx_config_cls="models.levit.LevitOnnxConfig"
        # ),
        # "longt5": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     onnx_config_cls="models.longt5.LongT5OnnxConfig",
        # ),
        # "longformer": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "multiple-choice",
        #     "question-answering",
        #     "sequence-classification",
        #     "token-classification",
        #     onnx_config_cls="models.longformer.LongformerOnnxConfig",
        # ),
        # "marian": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     onnx_config_cls="models.marian.MarianOnnxConfig",
        # ),
        # "mbart": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "causal-lm",
        #     "causal-lm-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     "sequence-classification",
        #     "question-answering",
        #     onnx_config_cls="models.mbart.MBartOnnxConfig",
        # ),
        # "mobilebert": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.mobilebert.MobileBertOnnxConfig",
        # ),
        # "mobilevit": supported_features_mapping(
        #     "default",
        #     "image-classification",
        #     onnx_config_cls="models.mobilevit.MobileViTOnnxConfig",
        # ),
        # "mt5": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     onnx_config_cls="models.mt5.MT5OnnxConfig",
        # ),
        # "m2m-100": supported_features_mapping(
        #     "default",
        #     "default-with-past",
        #     "seq2seq-lm",
        #     "seq2seq-lm-with-past",
        #     onnx_config_cls="models.m2m_100.M2M100OnnxConfig",
        # ),
        # "owlvit": supported_features_mapping(
        #     "default",
        #     onnx_config_cls="models.owlvit.OwlViTOnnxConfig",
        # ),
        # "perceiver": supported_features_mapping(
        #     "image-classification",
        #     "masked-lm",
        #     "sequence-classification",
        #     onnx_config_cls="models.perceiver.PerceiverOnnxConfig",
        # ),
        # "resnet": supported_features_mapping(
        #     "default",
        #     "image-classification",
        #     onnx_config_cls="models.resnet.ResNetOnnxConfig",
        # ),
        "roberta": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx="RobertaOnnxConfig",
        ),
        # "roformer": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "causal-lm",
        #     "sequence-classification",
        #     "token-classification",
        #     "multiple-choice",
        #     "question-answering",
        #     "token-classification",
        #     onnx_config_cls="models.roformer.RoFormerOnnxConfig",
        # ),
        # "segformer": supported_features_mapping(
        #     "default",
        #     "image-classification",
        #     "semantic-segmentation",
        #     onnx_config_cls="models.segformer.SegformerOnnxConfig",
        # ),
        # "squeezebert": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.squeezebert.SqueezeBertOnnxConfig",
        # ),
        "t5": supported_features_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx="T5OnnxConfig",
        ),
        # "vit": supported_features_mapping(
        #     "default", "image-classification", "masked-im", onnx_config_cls="models.vit.ViTOnnxConfig"
        # ),
        # "xlm": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "causal-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.xlm.XLMOnnxConfig",
        # ),
        # "xlm-roberta": supported_features_mapping(
        #     "default",
        #     "masked-lm",
        #     "causal-lm",
        #     "sequence-classification",
        #     "multiple-choice",
        #     "token-classification",
        #     "question-answering",
        #     onnx_config_cls="models.xlm_roberta.XLMRobertaOnnxConfig",
        # ),
        # "yolos": supported_features_mapping(
        #     "default",
        #     "object-detection",
        #     onnx_config_cls="models.yolos.YolosOnnxConfig",
        # ),
    }

    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, exporter: str, model_name: Optional[str] = None
    ) -> FeatureNameToExportConfigDict:
        """
        Tries to retrieve the feature -> exporter backend config constructors map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported features for.
            exporter (`str`):
                The name of the exporter.
            model_name (`str`, *optional*):
                The name attribute of the model object, only used for the exception message.

        Returns:
            `FeatureNameToExportConfigDict`: The dictionary mapping each feature to a corresponding `ExportConfig`
            constructor.
        """
        model_type = model_type.lower()
        model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type
        if model_type not in FeaturesManager._SUPPORTED_MODEL_TYPE:
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet. "
                f"Only {list(FeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        elif exporter not in FeaturesManager._SUPPORTED_MODEL_TYPE[model_type]:
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet with the {exporter} backend. "
                f"Only {list(FeaturesManager._SUPPORTED_MODEL_TYPE[model_type].keys())} are supported. "
                f"If you want to support {exporter} please propose a PR or open up an issue."
            )
        else:
            return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][exporter]

    @staticmethod
    def feature_to_task(feature: str) -> str:
        return feature.replace("-with-past", "")

    @staticmethod
    def _validate_framework_choice(framework: str):
        """
        Validates if the framework requested for the export is both correct and available, otherwise throws an
        exception.
        """
        if framework not in ["pt", "tf"]:
            raise ValueError(
                f"Only two frameworks are supported for export: pt or tf, but {framework} was provided."
            )
        elif framework == "pt" and not is_torch_available():
            raise RuntimeError("Cannot export model to ONNX using PyTorch because no PyTorch package was found.")
        elif framework == "tf" and not is_tf_available():
            raise RuntimeError("Cannot export model to ONNX using TensorFlow because no TensorFlow package was found.")

    @staticmethod
    def get_model_class_for_feature(feature: str, framework: str = "pt") -> Type:
        """
        Attempts to retrieve an AutoModel class from a feature name.

        Args:
            feature (`str`):
                The feature required.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The AutoModel class corresponding to the feature.
        """
        task = FeaturesManager.feature_to_task(feature)
        FeaturesManager._validate_framework_choice(framework)
        if framework == "pt":
            task_to_automodel = FeaturesManager._TASKS_TO_AUTOMODELS
        else:
            task_to_automodel = FeaturesManager._TASKS_TO_TF_AUTOMODELS
        if task not in task_to_automodel:
            raise KeyError(
                f"Unknown task: {feature}. Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}"
            )
        return task_to_automodel[task]

    # TODO: refactor this!
    @staticmethod
    def determine_framework(model: str, framework: str = None) -> str:
        """
        Determines the framework to use for the export.

        The priority is in the following order:
            1. User input via `framework`.
            2. If local checkpoint is provided, use the same framework as the checkpoint.
            3. Available framework in environment, with priority given to PyTorch

        Args:
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `None`):
                The framework to use for the export. See above for priority if none provided.

        Returns:
            `str`: The framework to use for the export.

        """
        if framework is not None:
            return framework

        framework_map = {"pt": "PyTorch", "tf": "TensorFlow"}
        exporter_map = {"pt": "torch", "tf": "tf2onnx"}

        if os.path.isdir(model):
            if os.path.isfile(os.path.join(model, WEIGHTS_NAME)):
                framework = "pt"
            elif os.path.isfile(os.path.join(model, TF2_WEIGHTS_NAME)):
                framework = "tf"
            else:
                raise FileNotFoundError(
                    "Cannot determine framework from given checkpoint location."
                    f" There should be a {WEIGHTS_NAME} for PyTorch"
                    f" or {TF2_WEIGHTS_NAME} for TensorFlow."
                )
            logger.info(f"Local {framework_map[framework]} model found.")
        else:
            if is_torch_available():
                framework = "pt"
            elif is_tf_available():
                framework = "tf"
            else:
                raise EnvironmentError("Neither PyTorch nor TensorFlow found in environment. Cannot export to ONNX.")

        logger.info(f"Framework not requested. Using {exporter_map[framework]} to export to ONNX.")

        return framework

    @staticmethod
    def get_model_from_feature(
        feature: str, model: str, framework: str = None, cache_dir: str = None
    ) -> Union["PreTrainedModel", "TFPreTrainedModel"]:
        """
        Attempts to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature (`str`):
                The feature required.
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `None`):
                The framework to use for the export. See `FeaturesManager.determine_framework` for the priority should
                none be provided.

        Returns:
            The instance of the model.

        """
        framework = FeaturesManager.determine_framework(model, framework)
        model_class = FeaturesManager.get_model_class_for_feature(feature, framework)
        try:
            model = model_class.from_pretrained(model, cache_dir=cache_dir)
        except OSError:
            if framework == "pt":
                logger.info("Loading TensorFlow model in PyTorch before exporting.")
                model = model_class.from_pretrained(model, from_tf=True, cache_dir=cache_dir)
            else:
                logger.info("Loading PyTorch model in TensorFlow before exporting.")
                model = model_class.from_pretrained(model, from_pt=True, cache_dir=cache_dir)
        return model

    @staticmethod
    def check_supported_model_or_raise(
        model: Union["PreTrainedModel", "TFPreTrainedModel"], exporter: str, feature: str = "default"
    ) -> Tuple[str, ExportConfigConstructor]:
        """
        Check whether or not the model has the requested features.

        Args:
            model (`Union[PreTrainedModel, TFPreTrainedModel]`):
                The model to export.
            exporter (`str`):
                The exporter to use.
            feature (`str`):
                The name of the feature to check if it is available.

        Returns:
             `Tuple[str, ExportConfigConstructor]`: The model type as well as the `ExportConfig`
             constructor for the requested backend.

        """
        model_type = model.config.model_type.replace("_", "-")
        model_name = getattr(model, "name", "")
        model_features = FeaturesManager.get_supported_features_for_model_type(model_type, exporter, model_name=model_name)
        if feature not in model_features:
            raise ValueError(
                f"{model.config.model_type} doesn't support feature {feature} for the {exporter} backend."
                f" Supported values are: {model_features}"
            )

        return model.config.model_type, FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][exporter][feature]

    def get_config(model_type: str, exporter: str, feature: str) -> ExportConfigConstructor:
        """
        Gets the OnnxConfig for a model_type and feature combination.

        Args:
            model_type (`str`):
                The model type to retrieve the config for.
            exporter (`str`):
                The exporter to use.
            feature (`str`):
                The feature to retrieve the config for.

        Returns:
            `ExportConfigConstructor`: The `ExportConfig` constructor for the requested backend.
        """
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][exporter][feature]
