from typing import Any, Optional, Union

from transformers import (
    AutoTokenizer,
    FeatureExtractionPipeline,
    Pipeline,
    PreTrainedTokenizer,
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

from optimum.utils import is_onnxruntime_available


SUPPORTED_TASKS = {}

if is_onnxruntime_available():
    from optimum.onnxruntime import (
        ORTModelForCausalLM,
        ORTModelForFeatureExtraction,
        ORTModelForQuestionAnswering,
        ORTModelForSeq2SeqLM,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )
    from optimum.onnxruntime.modeling_ort import ORTModel

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
        "text-generation": {
            "impl": TextGenerationPipeline,
            "class": (ORTModelForCausalLM,) if is_onnxruntime_available() else (),
            "default": "distilgpt2",
        },
        "summarization": {
            "impl": SummarizationPipeline,
            "class": (ORTModelForSeq2SeqLM,) if is_onnxruntime_available() else (),
            "default": "t5-base",
        },
        "translation": {
            "impl": TranslationPipeline,
            "class": (ORTModelForSeq2SeqLM,) if is_onnxruntime_available() else (),
            "default": "t5-base",
        },
        "text2text-generation": {
            "impl": Text2TextGenerationPipeline,
            "class": (ORTModelForSeq2SeqLM,) if is_onnxruntime_available() else (),
            "default": "t5-small",
        },
    }


def pipeline(
    task: str = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    accelerator: Optional[str] = "ort",
    **kwargs,
) -> Pipeline:

    targeted_task = "translation" if task.startswith("translation") else task

    if targeted_task not in list(SUPPORTED_TASKS.keys()):
        raise ValueError(f"Task {targeted_task} is not supported. Supported tasks are { list(SUPPORTED_TASKS.keys())}")

    if accelerator != "ort":
        raise ValueError(f"Accelerator {accelerator} is not supported. Supported accelerators are ort")

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id, from_transformers=True)
    elif isinstance(model, str):
        model_id = model
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model, from_transformers=True)
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

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_fast=use_fast,
        **kwargs,
    )
