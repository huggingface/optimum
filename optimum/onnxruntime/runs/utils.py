from ..modeling_decoder import ORTModelForCausalLM
from ..modeling_ort import (
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForQuestionAnswering,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)


task_ortmodel_map = {
    "text-generation": ORTModelForCausalLM,
    "feature-extraction": ORTModelForFeatureExtraction,
    "image-classification": ORTModelForImageClassification,
    "question-answering": ORTModelForQuestionAnswering,
    "text-classification": ORTModelForSequenceClassification,
    "token-classification": ORTModelForTokenClassification,
}
