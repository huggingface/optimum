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
from transformers.onnx.utils import get_preprocessor

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
            "type": "text",  # feature extraction is only supported for text at the moment
        },
        "image-classification": {
            "impl": ImageClassificationPipeline,
            "class": (ORTModelForImageClassification,) if is_onnxruntime_available() else (),
            "default": "google/vit-base-patch16-224",
            "type": "image",
        },
        "question-answering": {
            "impl": QuestionAnsweringPipeline,
            "class": (ORTModelForQuestionAnswering,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-cased-distilled-squad",
            "type": "text",
        },
        "text-classification": {
            "impl": TextClassificationPipeline,
            "class": (ORTModelForSequenceClassification,) if is_onnxruntime_available() else (),
            "default": "distilbert-base-uncased-finetuned-sst-2-english",
            "type": "text",
        },
        "text-generation": {
            "impl": TextGenerationPipeline,
            "class": (ORTModelForCausalLM,) if is_onnxruntime_available() else (),
            "default": "distilgpt2",
            "type": "text",
        },
        "token-classification": {
            "impl": TokenClassificationPipeline,
            "class": (ORTModelForTokenClassification,) if is_onnxruntime_available() else (),
            "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "type": "text",
        },
        "zero-shot-classification": {
            "impl": ZeroShotClassificationPipeline,
            "class": (ORTModelForSequenceClassification,) if is_onnxruntime_available() else (),
            "default": "facebook/bart-large-mnli",
            "type": "text",
        },
        "summarization": {
            "impl": SummarizationPipeline,
            "class": (ORTModelForSeq2SeqLM,) if is_onnxruntime_available() else (),
            "default": "t5-base",
            "type": "text",
        },
        "translation": {
            "impl": TranslationPipeline,
            "class": (ORTModelForSeq2SeqLM,) if is_onnxruntime_available() else (),
            "default": "t5-small",
            "type": "text",
        },
        "text2text-generation": {
            "impl": Text2TextGenerationPipeline,
            "class": (ORTModelForSeq2SeqLM,) if is_onnxruntime_available() else (),
            "default": "t5-small",
            "type": "text",
        },
    }

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
    elif values["type"] == "image":
        NO_TOKENIZER_TASKS.add(task)
    else:
        raise ValueError(f"Supported types are 'text' and 'image', got {values['type']}")


def load_bettertransformer(
    model,
    targeted_task,
    load_tokenizer=None,
    tokenizer=None,
    feature_extractor=None,
    load_feature_extractor=None,
    SUPPORTED_TASKS=None,
    **kwargs
):
    from transformers.pipelines import SUPPORTED_TASKS as TRANSFORMERS_SUPPORTED_TASKS

    from optimum.bettertransformer import BetterTransformer

    if model is None:
        model_id = TRANSFORMERS_SUPPORTED_TASKS[targeted_task]["default"]
        model = TRANSFORMERS_SUPPORTED_TASKS[targeted_task]["pt"][0].from_pretrained(model_id, **kwargs)
    elif isinstance(model, str):
        model_id = model
        model = TRANSFORMERS_SUPPORTED_TASKS[targeted_task]["pt"][0].from_pretrained(model, **kwargs)
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
        )

    model = BetterTransformer.transform(model, **kwargs)

    return model, model_id


def load_ort_pipeline(
    model,
    targeted_task,
    load_tokenizer,
    tokenizer,
    feature_extractor,
    load_feature_extractor,
    SUPPORTED_TASKS,
    **kwargs
):
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
        model_id = None
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
        )
    return model, model_id


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
    **kwargs,
) -> Pipeline:
    targeted_task = "translation" if task.startswith("translation") else task

    if targeted_task not in list(SUPPORTED_TASKS.keys()):
        raise ValueError(f"Task {targeted_task} is not supported. Supported tasks are { list(SUPPORTED_TASKS.keys())}")

    if accelerator not in ["ort", "bettertransformer"]:
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

    model, model_id = MAPPING_LOADING_FUNC[accelerator](
        model,
        targeted_task,
        load_tokenizer,
        tokenizer,
        feature_extractor,
        load_feature_extractor,
        SUPPORTED_TASKS,
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
