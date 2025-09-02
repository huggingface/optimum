# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Pipelines running different backends."""

from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from optimum.utils.import_utils import (
    is_ipex_available,
    is_onnxruntime_available,
    is_openvino_available,
    is_optimum_intel_available,
    is_optimum_onnx_available,
)
from optimum.utils.logging import get_logger


logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import (
        BaseImageProcessor,
        FeatureExtractionMixin,
        Pipeline,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        ProcessorMixin,
        TFPreTrainedModel,
    )


# The docstring is simply a copy of transformers.pipelines.pipeline's doc with minor modifications
# to reflect the fact that this pipeline loads Accelerated models using optimum.
def pipeline(
    task: Optional[str] = None,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, "PretrainedConfig"]] = None,
    tokenizer: Optional[Union[str, "PreTrainedTokenizer", "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, "FeatureExtractionMixin "]] = None,
    image_processor: Optional[Union[str, "BaseImageProcessor"]] = None,
    processor: Optional[Union[str, "ProcessorMixin"]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None,
    torch_dtype: Optional[Union[str, "torch.dtype"]] = "auto",
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    pipeline_class: Optional[Any] = None,
    accelerator: Optional[str] = None,
    **kwargs: Any,
) -> "Pipeline":
    """Utility factory method to build a [`Pipeline`] with an Optimum accelerated model, similar to `transformers.pipeline`.
    A pipeline consists of:
        - One or more components for pre-processing model inputs, such as a [tokenizer](tokenizer),
        [image_processor](image_processor), [feature_extractor](feature_extractor), or [processor](processors).
        - A [model](model) that generates predictions from the inputs.
        - Optional post-processing steps to refine the model's output, which can also be handled by processors.
    <Tip>
    While there are such optional arguments as `tokenizer`, `feature_extractor`, `image_processor`, and `processor`,
    they shouldn't be specified all at once. If these components are not provided, `pipeline` will try to load
    required ones automatically. In case you want to provide these components explicitly, please refer to a
    specific pipeline in order to get more details regarding what components are required.
    </Tip>
    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:
            - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
            - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
            - `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
            - `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
            - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
            - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
            - `"image-classification"`: will return a [`ImageClassificationPipeline`].
            - `"image-feature-extraction"`: will return an [`ImageFeatureExtractionPipeline`].
            - `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
            - `"image-text-to-text"`: will return a [`ImageTextToTextPipeline`].
            - `"image-to-image"`: will return a [`ImageToImagePipeline`].
            - `"image-to-text"`: will return a [`ImageToTextPipeline`].
            - `"mask-generation"`: will return a [`MaskGenerationPipeline`].
            - `"object-detection"`: will return a [`ObjectDetectionPipeline`].
            - `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
            - `"summarization"`: will return a [`SummarizationPipeline`].
            - `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
            - `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a
              [`TextClassificationPipeline`].
            - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
            - `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
            - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
            - `"translation"`: will return a [`TranslationPipeline`].
            - `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
            - `"video-classification"`: will return a [`VideoClassificationPipeline`].
            - `"visual-question-answering"`: will return a [`VisualQuestionAnsweringPipeline`].
            - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
            - `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
            - `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
            - `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].
        model (`str` or [`ORTModel` or `OVModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a ONNX Runtime model inheriting from [`ORTModel` or `OVModel`].
            If not provided, the default for the `task` will be loaded.
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].
            If not provided, the default configuration file for the requested model will be used. That means that if
            `model` is given, its default configuration will be used. However, if `model` is not supplied, this
            `task`'s default model's config is used instead.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].
            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from [`PreTrainedFeatureExtractor`].
            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.
            If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
            is a string). However, if `config` is also not given or not a string, then the default feature extractor
            for the given `task` will be loaded.
        image_processor (`str` or [`BaseImageProcessor`], *optional*):
            The image processor that will be used by the pipeline to preprocess images for the model. This can be a
            model identifier or an actual image processor inheriting from [`BaseImageProcessor`].
            Image processors are used for Vision models and multi-modal models that require image inputs. Multi-modal
            models will also require a tokenizer to be passed.
            If not provided, the default image processor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default image processor for `config` is loaded (if it is
            a string).
        processor (`str` or [`ProcessorMixin`], *optional*):
            The processor that will be used by the pipeline to preprocess data for the model. This can be a model
            identifier or an actual processor inheriting from [`ProcessorMixin`].
            Processors are used for multi-modal models that require multi-modal inputs, for example, a model that
            requires both text and image inputs.
            If not provided, the default processor for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default processor for `config` is loaded (if it is a string).
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        revision (`str`, *optional*, defaults to `"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        device (`int` or `str` or `torch.device`):
            Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this
            pipeline will be allocated.
        device_map (`str` or `dict[str, Union[int, str, torch.device]`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set
            `device_map="auto"` to compute the most optimized `device_map` automatically (see
            [here](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling#accelerate.cpu_offload)
            for more information).
            <Tip warning={true}>
            Do not use `device_map` AND `device` at the same time as they will conflict
            </Tip>
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,
            tokenization or even pipeline files. This option should only be set to `True` for repositories you trust
            and in which you have read the code, as it will execute code present on the Hub on your local machine.
        model_kwargs (`dict[str, Any]`, *optional*):
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.
        pipeline_class (`type`, *optional*):
            Can be used to force using a custom pipeline class. If not provided, the default pipeline class for the
            specified task will be used.
        accelerator (`str`, *optional*):
            The accelerator to use, either `"ort"` for ONNX Runtime, `"ov"` for OpenVINO, or `"ipex"` for Intel
            Extension for PyTorch. If no accelerator is specified, will default to the one currently installed/available.
        kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).
    Returns:
        [`Pipeline`]: A suitable pipeline for the task.
    Examples:
    ```python
    >>> from optimum.pipelines import pipeline
    >>> # Sentiment analysis pipeline with default model, using OpenVINO
    >>> analyzer = pipeline("sentiment-analysis", accelerator="ov")
    >>> # Question answering pipeline, specifying the checkpoint identifier, with IPEX
    >>> oracle = pipeline(
    ...     "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased", accelerator="ipex"
    ... )
    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer, with ONNX Runtime
    >>> model = ORTModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```
    """

    if accelerator is None:
        # probably needs to check for couple of stuff here, like target device, type(model) etc.
        if is_optimum_intel_available() and is_openvino_available():
            logger.info(
                "`accelerator` not specified. Using OpenVINO (`ov`) as the accelerator since `optimum-intel[openvino]` is installed."
            )
            accelerator = "ov"
        elif is_optimum_onnx_available() and is_onnxruntime_available():
            logger.info(
                "`accelerator` not specified. Using ONNX Runtime (`ort`) as the accelerator since `optimum-onnx[onnxruntime]` is installed."
            )
            accelerator = "ort"
        elif is_optimum_intel_available() and is_ipex_available():
            logger.info(
                "`accelerator` not specified. Using IPEX (`ipex`) as the accelerator since `optimum-intel[ipex]` is installed."
            )
            accelerator = "ipex"
        else:
            raise ImportError(
                "You need to install either `optimum-onnx[onnxruntime]` to use ONNX Runtime as an accelerator, "
                "or `optimum-intel[openvino]` to use OpenVINO as an accelerator, "
                "or `optimum-intel[ipex]` to use IPEX as an accelerator."
            )

    if accelerator == "ort":
        from optimum.onnxruntime import pipeline as ort_pipeline

        return ort_pipeline(
            task=task,
            model=model,
            config=config,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            framework=framework,
            revision=revision,
            use_fast=use_fast,
            token=token,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            pipeline_class=pipeline_class,
            **kwargs,
        )
    elif accelerator in ["ov", "ipex"]:
        from optimum.intel import pipeline as intel_pipeline

        return intel_pipeline(
            task=task,
            model=model,
            config=config,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            framework=framework,
            revision=revision,
            use_fast=use_fast,
            token=token,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            pipeline_class=pipeline_class,
            accelerator=accelerator,
            **kwargs,
        )
    else:
        raise ValueError(f"Accelerator {accelerator} not recognized. Please use 'ort', 'ov' or 'ipex'.")
