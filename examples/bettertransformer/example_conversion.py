import torch
from transformers import AutoModelForSequenceClassification

from optimum.bettertransformer import BetterTransformer


model_hf = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
model = BetterTransformer.transform(model_hf)

inputs_ids = torch.LongTensor([[1, 1, 1, 1, 1, 1]])
attention_mask = torch.Tensor([[1, 1, 1, 0, 0, 0]])

print(model)
print(model_hf)

print(model_hf(inputs_ids))
print(model(inputs_ids))
