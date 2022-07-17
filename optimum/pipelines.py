from typing import Any, Optional, Union

from transformers import (
    FeatureExtractionPipeline,
    ImageClassificationPipeline,
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
from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import NO_FEATURE_EXTRACTOR_TASKS, NO_TOKENIZER_TASKS

from .utils import is_onnxruntime_available


SUPPORTED_TASKS = {}

if is_onnxruntime_available():
    from .onnxruntime import (
        ORTModelForCausalLM,
        ORTModelForFeatureExtraction,
        ORTModelForImageClassification,
        ORTModelForQuestionAnswering,
        ORTModelForSeq2SeqLM,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )
    from .onnxruntime.modeling_ort import ORTModel

    SUPPORTED_TASKS = {
        "feature-extraction": {
            "impl": FeatureExtractionPipeline,
            "class": (ORTModelForFeatureExtraction,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-cased",
        },
        "image-classification": {
            "impl": ImageClassificationPipeline,
            "class": (ORTModelForImageClassification,) if is_onnxruntime_available() else (),
            "default": "google/vit-base-patch16-224",
        },
        "question-answering": {
            "impl": QuestionAnsweringPipeline,
            "class": (ORTModelForQuestionAnswering,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-cased-distilled-squad",
        },
        "text-classification": {
            "impl": TextClassificationPipeline,
            "class": (ORTModelForSequenceClassification,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        "text-generation": {
            "impl": TextGenerationPipeline,
            "class": (ORTModelForCausalLM,) if is_onnxruntime_available() else (),
            "default": "distilgpt2",
        },
        "token-classification": {
            "impl": TokenClassificationPipeline,
            "class": (ORTModelForTokenClassification,) if is_onnxruntime_available() else (),
            "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
        },
        "zero-shot-classification": {
            "impl": ZeroShotClassificationPipeline,
            "class": (ORTModelForSequenceClassification,) if is_onnxruntime_available() else (),
            "default": "facebook/bart-large-mnli",
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

    # copied from transformers.pipelines.__init__.py l.609
    if task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False
    else:
        load_tokenizer = True
    if task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False
    else:
        load_feature_extractor = True

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id, from_transformers=True)
    elif isinstance(model, str):
        model_id = model
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model, from_transformers=True)
    elif isinstance(model, ORTModel):
        if tokenizer is None and load_tokenizer:
            raise ValueError("If you pass a model as a ORTModel, you must pass a tokenizer as well")
        if feature_extractor is None and load_feature_extractor:
            raise ValueError("If you pass a model as a ORTModel, you must pass a feature extractor as well")
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
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
