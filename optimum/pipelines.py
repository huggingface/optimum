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
"""Pipelines running different backends."""

from typing import Any, Dict, Optional, Union

from transformers import (
    AutomaticSpeechRecognitionPipeline,
    FeatureExtractionPipeline,
    ImageClassificationPipeline,
    ImageSegmentationPipeline,
    Pipeline,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    ZeroShotClassificationPipeline,
)
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import SUPPORTED_TASKS as TRANSFORMERS_SUPPORTED_TASKS

from .bettertransformer import BetterTransformer
from .utils import is_onnxruntime_available
from .utils.file_utils import find_files_matching_pattern


SUPPORTED_TASKS = {}

if is_onnxruntime_available():
    from .onnxruntime import (
        ORTModelForCausalLM,
        ORTModelForFeatureExtraction,
        ORTModelForImageClassification,
        ORTModelForQuestionAnswering,
        ORTModelForSemanticSegmentation,
        ORTModelForSeq2SeqLM,
        ORTModelForSequenceClassification,
        ORTModelForSpeechSeq2Seq,
        ORTModelForTokenClassification,
    )
    from .onnxruntime.modeling_ort import ORTModel

    SUPPORTED_TASKS = {
        "feature-extraction": {
            "impl": FeatureExtractionPipeline,
            "class": (ORTModelForFeatureExtraction,),
            "default": "distilbert-base-cased",
            "type": "text",  # feature extraction is only supported for text at the moment
        },
        "image-classification": {
            "impl": ImageClassificationPipeline,
            "class": (ORTModelForImageClassification,),
            "default": "google/vit-base-patch16-224",
            "type": "image",
        },
        "image-segmentation": {
            "impl": ImageSegmentationPipeline,
            "class": (ORTModelForSemanticSegmentation,) if is_onnxruntime_available() else (),
            "default": "nvidia/segformer-b0-finetuned-ade-512-512",
            "type": "image",
        },
        "question-answering": {
            "impl": QuestionAnsweringPipeline,
            "class": (ORTModelForQuestionAnswering,),
            "default": "distilbert-base-cased-distilled-squad",
            "type": "text",
        },
        "text-classification": {
            "impl": TextClassificationPipeline,
            "class": (ORTModelForSequenceClassification,),
            "default": "distilbert-base-uncased-finetuned-sst-2-english",
            "type": "text",
        },
        "text-generation": {
            "impl": TextGenerationPipeline,
            "class": (ORTModelForCausalLM,),
            "default": "distilgpt2",
            "type": "text",
        },
        "token-classification": {
            "impl": TokenClassificationPipeline,
            "class": (ORTModelForTokenClassification,),
            "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "type": "text",
        },
        "zero-shot-classification": {
            "impl": ZeroShotClassificationPipeline,
            "class": (ORTModelForSequenceClassification,),
            "default": "facebook/bart-large-mnli",
            "type": "text",
        },
        "summarization": {
            "impl": SummarizationPipeline,
            "class": (ORTModelForSeq2SeqLM,),
            "default": "t5-base",
            "type": "text",
        },
        "translation": {
            "impl": TranslationPipeline,
            "class": (ORTModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
        "text2text-generation": {
            "impl": Text2TextGenerationPipeline,
            "class": (ORTModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
        "automatic-speech-recognition": {
            "impl": AutomaticSpeechRecognitionPipeline,
            "class": (ORTModelForSpeechSeq2Seq,),
            "default": "openai/whisper-tiny.en",
            "type": "multimodal",
        },
    }

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
    elif values["type"] == "image":
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")


def load_bettertransformer(
    model,
    targeted_task,
    load_tokenizer=None,
    tokenizer=None,
    feature_extractor=None,
    load_feature_extractor=None,
    SUPPORTED_TASKS=None,
    subfolder: str = "",
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: str = "main",
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
):
    if model_kwargs is None:
        model_kwargs = {}

    if model is None:
        model_id = TRANSFORMERS_SUPPORTED_TASKS[targeted_task]["default"]
        model = TRANSFORMERS_SUPPORTED_TASKS[targeted_task]["pt"][0].from_pretrained(model_id, **model_kwargs)
    elif isinstance(model, str):
        model_id = model
        model = TRANSFORMERS_SUPPORTED_TASKS[targeted_task]["pt"][0].from_pretrained(model, **model_kwargs)
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
        )

    model = BetterTransformer.transform(model, **kwargs)

    return model, model_id, tokenizer, feature_extractor


def load_ort_pipeline(
    model,
    targeted_task,
    load_tokenizer,
    tokenizer,
    feature_extractor,
    load_feature_extractor,
    SUPPORTED_TASKS,
    subfolder: str = "",
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: str = "main",
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
):
    if model_kwargs is None:
        model_kwargs = {}

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id, from_transformers=True)
    elif isinstance(model, str):
        from .onnxruntime.modeling_seq2seq import ENCODER_ONNX_FILE_PATTERN, ORTModelForConditionalGeneration

        model_id = model
        ort_model_class = SUPPORTED_TASKS[targeted_task]["class"][0]

        if issubclass(ort_model_class, ORTModelForConditionalGeneration):
            pattern = ENCODER_ONNX_FILE_PATTERN
        else:
            pattern = ".+?.onnx"

        onnx_files = find_files_matching_pattern(
            model,
            pattern,
            glob_pattern="**/*.onnx",
            subfolder=subfolder,
            use_auth_token=use_auth_token,
            revision=revision,
        )
        from_transformers = len(onnx_files) == 0
        model = ort_model_class.from_pretrained(model, from_transformers=from_transformers, **model_kwargs)
    elif isinstance(model, ORTModel):
        if tokenizer is None and load_tokenizer:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                    tokenizer = preprocessor
                    break
            if tokenizer is None:
                raise ValueError(
                    "Could not automatically find a tokenizer for the ORTModel, you must pass a tokenizer explictly"
                )
        if feature_extractor is None and load_feature_extractor:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, PreTrainedFeatureExtractor):
                    feature_extractor = preprocessor
                    break
            if feature_extractor is None:
                raise ValueError(
                    "Could not automatically find a feature extractor for the ORTModel, you must pass a "
                    "feature_extractor explictly"
                )
        model_id = None
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
        )
    return model, model_id, tokenizer, feature_extractor


MAPPING_LOADING_FUNC = {
    "ort": load_ort_pipeline,
    "bettertransformer": load_bettertransformer,
}


def pipeline(
    task: str = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    accelerator: Optional[str] = "ort",
    *model_kwargs,
    **kwargs,
) -> Pipeline:
    targeted_task = "translation" if task.startswith("translation") else task

    if accelerator == "ort":
        if targeted_task not in list(SUPPORTED_TASKS.keys()):
            raise ValueError(
                f"Task {targeted_task} is not supported. Supported tasks are { list(SUPPORTED_TASKS.keys())}"
            )

    if accelerator not in MAPPING_LOADING_FUNC:
        raise ValueError(
            f'Accelerator {accelerator} is not supported. Supported accelerators are "ort" and "bettertransformer".'
        )

    # copied from transformers.pipelines.__init__.py l.609
    if targeted_task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False
    else:
        load_tokenizer = True
    if targeted_task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False
    else:
        load_feature_extractor = True

    model, model_id, tokenizer, feature_extractor = MAPPING_LOADING_FUNC[accelerator](
        model,
        targeted_task,
        load_tokenizer,
        tokenizer,
        feature_extractor,
        load_feature_extractor,
        SUPPORTED_TASKS,
        *model_kwargs,
        **kwargs,
    )

    if tokenizer is None and load_tokenizer:
        tokenizer = get_preprocessor(model_id)
    if feature_extractor is None and load_feature_extractor:
        feature_extractor = get_preprocessor(model_id)

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        use_fast=use_fast,
        use_auth_token=use_auth_token,
        **kwargs,
    )
