import logging
import argparse
import wget
import os
import zipfile
import numpy as np
import random
import time
import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification, TrainingArguments
from transformers import Trainer, get_linear_schedule_with_warmup, default_data_collator, AdamW

import onnxruntime
from torch_ort import ORTModule

# Load PyTorch models from transformers
model_name = "bert-base-cased"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
dataset_name = "sst2"
task = "sst2"
dataset = load_dataset("glue", dataset_name)
metric = load_metric("glue", dataset_name)

max_seq_length = min(128, tokenizer.model_max_length)
padding = "max_length"

def preprocess_function(examples):
    args = (examples["sentence"],)
    return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

max_train_samples = 1000
max_valid_samples = 100
max_test_samples = 100
train_dataset = encoded_dataset["train"].select(range(max_train_samples))
valid_dataset = encoded_dataset["validation"].select(range(max_valid_samples))
test_dataset = encoded_dataset["test"].remove_columns(["label"]).select(range(max_test_samples))

# Wrap the model with ORTModule
model = ORTModule(model)
if torch.cuda.is_available():
    model.cuda()
    
# Setup the training with trainer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

# ----args
training_args = TrainingArguments(
    output_dir="./models", 
    num_train_epochs=2, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16, 
    warmup_steps=500,
    weight_decay=0.01, 
    logging_dir="./logs",
)

# Trainer
max_train_samples = 1000
train_dataset = encoded_dataset["train"].select(range(max_train_samples))
valid_dataset = encoded_dataset["validation"]
test_dataset = encoded_dataset["test"]

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Training / Fune-tuning
def train_func(model):
    trainer.model_wrapped = model
    trainer.model = model    
    train_result = trainer.train()
    metrics = train_result.metrics
    # trainer.save_model() 
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

train_func(model)