from sre_constants import ANY
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from transformers import (
    AutoTokenizer,
    FeatureExtractionPipeline,
    Pipeline,
    PreTrainedTokenizer,
    QuestionAnsweringPipeline,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    ZeroShotClassificationPipeline,
)
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.pipelines import TASK_ALIASES

from optimum.utils import is_onnxruntime_available


SUPPORTED_TASKS = {}

if is_onnxruntime_available():
    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForQuestionAnswering,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
        ORTModel,
    )

    SUPPORTED_TASKS = {
        "feature-extraction": {
            "impl": FeatureExtractionPipeline,
            "class": (ORTModelForFeatureExtraction,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-cased",
        },
        "text-classification": {
            "impl": TextClassificationPipeline,
            "class": (ORTModelForSequenceClassification,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        "token-classification": {
            "impl": TokenClassificationPipeline,
            "class": (ORTModelForTokenClassification,) if is_onnxruntime_available() else (),
            "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
        },
        "question-answering": {
            "impl": QuestionAnsweringPipeline,
            "class": (ORTModelForQuestionAnswering,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-cased-distilled-squad",
        },
        "zero-shot-classification": {
            "impl": ZeroShotClassificationPipeline,
            "class": (ORTModelForSequenceClassification,) if is_onnxruntime_available() else (),
            "default": "facebook/bart-large-mnli",
        },
    }


def optimum_pipeline(
    task: str = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    do_optimization: Optional[bool] = False,
    do_quantization: Optional[bool] = False,
    **kwargs,
) -> Pipeline:

    if task not in list(SUPPORTED_TASKS.keys()):
        raise ValueError(f"Task {task} is not supported. Supported tasks are { list(SUPPORTED_TASKS.keys())}")

    if model is None:
        model_id = SUPPORTED_TASKS[task]["default"]
        model = SUPPORTED_TASKS[task]["class"][0].from_transformers(model_id)
    elif isinstance(model, str):
        model_id = model
        model = SUPPORTED_TASKS[task]["class"][0].from_transformers(model)
    elif isinstance(model, ORTModel):
        if tokenizer is None:
            raise ValueError("If you pass a model as a ORTModel, you must pass a tokenizer as well")
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if do_optimization:
        model = model.optimize()
    if do_quantization:
        model = model.quantize()

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_fast=use_fast,
        **kwargs,
    )
