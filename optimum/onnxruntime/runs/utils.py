from optimum.onnxruntime.modeling_ort import (
    ORTModelForCausalLM,
    ORTModelForFeatureExtraction,
    ORTModelForQuestionAnswering,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)


task_ortmodel_map = {
    "feature-extraction": ORTModelForFeatureExtraction,
    "question-answering": ORTModelForQuestionAnswering,
    "text-classification": ORTModelForSequenceClassification,
    "token-classification": ORTModelForTokenClassification,
    "causal-lm": ORTModelForCausalLM,
}
