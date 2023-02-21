# coding=utf-8
# Copyright 2022 the HuggingFace Inc. team.
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

import gc
import random
import subprocess
import sys
import tempfile
import unittest
from itertools import chain
from typing import List, Optional
from unittest.mock import Mock, patch

import nltk
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    default_data_collator,
    is_torch_available,
)
from transformers.testing_utils import (
    require_deepspeed,
    require_torch,
    slow,
)
from transformers.training_args import OptimizerNames


if is_torch_available():
    pass

import onnxruntime
from parameterized import parameterized

from optimum.onnxruntime import ORTSeq2SeqTrainer, ORTSeq2SeqTrainingArguments, ORTTrainer, ORTTrainingArguments
from optimum.onnxruntime.training_args import ORTOptimizerNames


nltk.download("punkt")

_ENCODERS_TO_TEST = {
    ("distilbert", "distilbert-base-cased"),
}

_DECODERS_TO_TEST = {
    ("gpt2", "gpt2"),
}

_SEQ2SEQ_MODELS_TO_TEST = {
    ("t5", "t5-small"),
}

_ENCODER_TASKS_DATASETS_CONFIGS = {
    "sequence-classification": {
        "dataset": ["glue", "sst2"],
        "metric": ["glue", "sst2"],
        "data_collator": default_data_collator,
        "data_collator_class": DataCollatorWithPadding,
    },
    "token-classification": {
        "dataset": ["conll2003"],
        "metric": ["seqeval"],
        "data_collator_class": DataCollatorForTokenClassification,
    },
}

_DECODER_TASKS_DATASETS_CONFIGS = {
    "causal-lm": {
        "dataset": ["wikitext", "wikitext-2-raw-v1"],
        "metric": ["accuracy"],
        "data_collator": default_data_collator,
    },
    "causal-lm-with-past": {
        "dataset": ["wikitext", "wikitext-2-raw-v1"],
        "metric": ["accuracy"],
        "data_collator": default_data_collator,
    },
}

_SEQ2SEQ_TASKS_DATASETS_CONFIGS = {
    "seq2seq-lm": {
        "dataset": ["xsum"],
        "metric": ["rouge"],
        "data_collator_class": DataCollatorForSeq2Seq,
    },
    "seq2seq-lm-with-past": {
        "dataset": ["xsum"],
        "metric": ["rouge"],
        "data_collator_class": DataCollatorForSeq2Seq,
    },
}


def _get_models_to_test(model_list, task_list, both_inf_backend=False, excluded: Optional[List[str]] = None):
    models_to_test = []

    for name, model_name in model_list:
        for feature, data_metric_config in task_list.items():
            if excluded and (name in excluded or feature in excluded):
                continue
            if both_inf_backend:
                models_to_test.append(
                    (f"{name}_{feature}", model_name, feature, data_metric_config, True)
                )  # inference_with_ort=True
                models_to_test.append(
                    (f"{name}_{feature}", model_name, feature, data_metric_config, False)
                )  # inference_with_ort=False
            else:
                models_to_test.append((f"{name}_{feature}", model_name, feature, data_metric_config))

    return sorted(models_to_test)


def _get_data_collator(data_metric_config, tokenizer=None, model=None, training_args=None, label_pad_token_id=None):
    if "data_collator" in data_metric_config.keys():
        data_collator = data_metric_config["data_collator"]
    elif "data_collator_class" in data_metric_config.keys():
        data_collator_class = data_metric_config["data_collator_class"]
        if training_args is not None:
            pad_to_multiple_of = 8 if training_args.fp16 else None
        if data_collator_class is DataCollatorForSeq2Seq:
            data_collator = data_collator_class(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        else:
            data_collator = data_collator_class(tokenizer, pad_to_multiple_of=8)
    else:
        raise KeyError("You need to pass either `data_collator` or `data_collator_class` to create the data collator.")

    return data_collator


def get_ort_training_args(feature, **kwargs):
    if feature in _ENCODER_TASKS_DATASETS_CONFIGS or feature in _DECODER_TASKS_DATASETS_CONFIGS:
        training_args = ORTTrainingArguments(**kwargs)
    elif feature in _SEQ2SEQ_TASKS_DATASETS_CONFIGS:
        training_args = ORTSeq2SeqTrainingArguments(**kwargs)
    return training_args


def get_ort_trainer(
    model_name,
    feature,
    data_metric_config,
    training_args,
    max_seq_length=None,
    max_train_samples=None,
    max_valid_samples=None,
    max_test_samples=None,
    **kwargs,
):
    training_kwargs = load_and_prepare(feature)(
        model_name,
        data_metric_config,
        max_seq_length,
        training_args=training_args,
        max_train_samples=max_train_samples,
        max_valid_samples=max_valid_samples,
        max_test_samples=max_test_samples,
        **kwargs,
    )
    test_dataset = training_kwargs.pop("test_dataset", None)

    if getattr(training_args, "predict_with_generate", False) is not True:
        training_kwargs.pop("compute_metrics", None)

    if feature in _ENCODER_TASKS_DATASETS_CONFIGS or feature in _DECODER_TASKS_DATASETS_CONFIGS:
        trainer = ORTTrainer(feature=feature, args=training_args, **training_kwargs)
    elif feature in _SEQ2SEQ_TASKS_DATASETS_CONFIGS:
        trainer = ORTSeq2SeqTrainer(feature=feature, args=training_args, **training_kwargs)
    else:
        raise

    return trainer, test_dataset


def load_and_prepare(feature):
    preprocess_mapping = {
        "sequence-classification": load_and_prepare_glue,
        "token-classification": load_and_prepare_ner,
        "causal-lm": load_and_prepare_clm,
        "causal-lm-with-past": load_and_prepare_clm,
        "seq2seq-lm": load_and_prepare_xsum,
        "seq2seq-lm-with-past": load_and_prepare_xsum,
    }
    return preprocess_mapping[feature]


def load_and_prepare_glue(model_name, data_metric_config, max_seq_length, padding="max_length", **kwargs):
    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare dataset
    dataset = load_dataset(*data_metric_config["dataset"])
    metric = load(*data_metric_config["metric"])

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    data_collator = _get_data_collator(data_metric_config)

    def preprocess_function(examples):
        args = (examples["sentence"],)
        return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = encoded_dataset["train"]
    valid_dataset = encoded_dataset["validation"]
    test_dataset = encoded_dataset["test"].remove_columns(["label"])

    max_train_samples = kwargs.get("max_train_samples", None)
    max_valid_samples = kwargs.get("max_valid_samples", None)
    max_test_samples = kwargs.get("max_test_samples", None)

    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    if max_valid_samples:
        valid_dataset = valid_dataset.select(range(max_valid_samples))
    if max_test_samples:
        test_dataset = test_dataset.select(range(max_test_samples))

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "compute_metrics": compute_metrics,
    }


def load_and_prepare_ner(model_name, data_metric_config, max_seq_length, padding="max_length", **kwargs):
    # Load dataset and metric
    dataset = load_dataset(*data_metric_config["dataset"])
    metric = load(*data_metric_config["metric"])
    label_all_tokens = True
    task = "ner"
    label_list = dataset["train"].features[f"{task}_tags"].feature.names

    # Prepare model
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
    if model_name.split("-")[0] in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Prepare dataset
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    data_collator = _get_data_collator(data_metric_config, tokenizer)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    train_dataset = tokenized_datasets["train"]
    valid_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"].remove_columns(["labels"])

    max_train_samples = kwargs.get("max_train_samples", None)
    max_valid_samples = kwargs.get("max_valid_samples", None)
    max_test_samples = kwargs.get("max_test_samples", None)

    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    if max_valid_samples:
        valid_dataset = valid_dataset.select(range(max_valid_samples))
    if max_test_samples:
        test_dataset = test_dataset.select(range(max_test_samples))

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return {
        "model": model,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "compute_metrics": compute_metrics,
    }


def load_and_prepare_clm(model_name, data_metric_config, max_seq_length, padding="max_length", **kwargs):
    # Load dataset and metric
    dataset = load_dataset(*data_metric_config["dataset"])
    metric = load(*data_metric_config["metric"])

    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare dataset
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    data_collator = _get_data_collator(data_metric_config)
    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name], padding=padding, max_length=max_seq_length)
        return output

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=column_names)
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    train_dataset = lm_dataset["train"]
    valid_dataset = lm_dataset["validation"]
    test_dataset = lm_dataset["test"].remove_columns(["labels"])

    max_train_samples = kwargs.get("max_train_samples", None)
    max_valid_samples = kwargs.get("max_valid_samples", None)
    max_test_samples = kwargs.get("max_test_samples", None)
    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    if max_valid_samples:
        valid_dataset = valid_dataset.select(range(max_valid_samples))
    if max_test_samples:
        test_dataset = test_dataset.select(range(max_test_samples))

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "compute_metrics": compute_metrics,
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
    }


def load_and_prepare_xsum(model_name, data_metric_config, _, **kwargs):
    # Prepare model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset and metric
    dataset = load_dataset(*data_metric_config["dataset"])
    metric = load(*data_metric_config["metric"])

    if model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""

    max_input_length = kwargs.get("max_input_length", 128)
    max_target_length = kwargs.get("max_input_length", 64)

    training_args = kwargs.get("training_args", None)
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = _get_data_collator(
        data_metric_config, tokenizer, model, training_args=training_args, label_pad_token_id=label_pad_token_id
    )

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = encoded_dataset["train"]
    valid_dataset = encoded_dataset["validation"]
    test_dataset = encoded_dataset["test"]

    max_train_samples = kwargs.get("max_train_samples", None)
    max_valid_samples = kwargs.get("max_valid_samples", None)
    max_test_samples = kwargs.get("max_test_samples", None)

    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    if max_valid_samples:
        valid_dataset = valid_dataset.select(range(max_valid_samples))
    if max_test_samples:
        test_dataset = test_dataset.select(range(max_test_samples))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    return {
        "model": model,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "compute_metrics": compute_metrics,
    }


class ORTTrainerIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        args = ORTTrainingArguments("..")
        self.n_epochs = min(args.num_train_epochs, 1)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.max_seq_length = 64
        self.max_train_samples = 50
        self.max_valid_samples = 20
        self.max_test_samples = 10

        self.warmup_steps = 10
        self.weight_decay = 0.01

    @parameterized.expand(
        _get_models_to_test(_ENCODERS_TO_TEST, _ENCODER_TASKS_DATASETS_CONFIGS, both_inf_backend=True)
        # + _get_models_to_test(_DECODERS_TO_TEST, _DECODER_TASKS_DATASETS_CONFIGS, both_inf_backend=True)  # Skip test for OOM bug
        + _get_models_to_test(_SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS, both_inf_backend=True),
        skip_on_empty=True,
    )
    def test_trainer_fp32(self, test_name, model_name, feature, data_metric_config, inference_with_ort):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = get_ort_training_args(
                feature=feature,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
            )

            trainer, test_dataset = get_ort_trainer(
                model_name,
                feature,
                data_metric_config,
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            trainer.save_model()
            trainer.evaluate(inference_with_ort=inference_with_ort)
            trainer.predict(test_dataset, inference_with_ort=inference_with_ort)
            gc.collect()

    @parameterized.expand(
        _get_models_to_test(_ENCODERS_TO_TEST, _ENCODER_TASKS_DATASETS_CONFIGS, both_inf_backend=True)
        # + _get_models_to_test(_DECODERS_TO_TEST, _DECODER_TASKS_DATASETS_CONFIGS, both_inf_backend=True)  # Skip test for OOM bug
        + _get_models_to_test(_SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS, both_inf_backend=True),
        skip_on_empty=True,
    )
    def test_trainer_fp32_with_label_smoothing(
        self, test_name, model_name, feature, data_metric_config, inference_with_ort
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = get_ort_training_args(
                feature=feature,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                label_smoothing_factor=0.1,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
            )

            trainer, test_dataset = get_ort_trainer(
                model_name,
                feature,
                data_metric_config,
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            trainer.save_model()
            trainer.evaluate(inference_with_ort=inference_with_ort)
            trainer.predict(test_dataset, inference_with_ort=inference_with_ort)
            gc.collect()

    @slow
    @parameterized.expand(
        _get_models_to_test(_ENCODERS_TO_TEST, _ENCODER_TASKS_DATASETS_CONFIGS)
        # + _get_models_to_test(_DECODERS_TO_TEST, _DECODER_TASKS_DATASETS_CONFIGS)  # Skip test for OOM bug
        + _get_models_to_test(_SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS),
        skip_on_empty=True,
    )
    def test_trainer_fp16_pt_inference(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = get_ort_training_args(
                feature=feature,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
                fp16=True,
            )

            trainer, test_dataset = get_ort_trainer(
                model_name,
                feature,
                data_metric_config,
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            trainer.save_model()
            trainer.evaluate()
            trainer.predict(test_dataset)
            gc.collect()

    @slow
    @parameterized.expand(
        _get_models_to_test(_ENCODERS_TO_TEST, _ENCODER_TASKS_DATASETS_CONFIGS)
        # Exclude "with-past" tests as they fail for ORT inference after the mixed-precision training
        # + _get_models_to_test(_DECODERS_TO_TEST, _DECODER_TASKS_DATASETS_CONFIGS, excluded=["causal-lm-with-past"])  # Skip test for OOM bug
        + _get_models_to_test(
            _SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS, excluded=["seq2seq-lm-with-past"]
        ),
        skip_on_empty=True,
    )
    def test_trainer_fp16_ort_inference(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = get_ort_training_args(
                feature=feature,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
                fp16=True,
            )

            trainer, test_dataset = get_ort_trainer(
                model_name,
                feature,
                data_metric_config,
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            trainer.save_model()
            trainer.evaluate(inference_with_ort=True)
            trainer.predict(test_dataset, inference_with_ort=True)
            gc.collect()

    # Skip this test as a large amount of ops don't support bf16 yet.
    # @unittest.skip("Skip BF16 test.")
    # @slow
    # @require_torch_bf16_gpu
    # @parameterized.expand(
    #     _get_models_to_test(_ENCODERS_TO_TEST, _ENCODER_TASKS_DATASETS_CONFIGS)
    #     + _get_models_to_test(_DECODERS_TO_TEST, _DECODER_TASKS_DATASETS_CONFIGS)
    #     + _get_models_to_test(_SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS),
    #     skip_on_empty=True,
    # )
    # def test_trainer_bf16(self, test_name, model_name, feature, data_metric_config):
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         training_args = get_ort_training_args(
    #             feature=feature,
    #             output_dir=tmp_dir,
    #             num_train_epochs=self.n_epochs,
    #             per_device_train_batch_size=self.per_device_train_batch_size,
    #             per_device_eval_batch_size=self.per_device_eval_batch_size,
    #             warmup_steps=self.warmup_steps,
    #             weight_decay=self.weight_decay,
    #             logging_dir=tmp_dir,
    #             bf16=True,
    #         )

    #         trainer, test_dataset = get_ort_trainer(
    #             model_name,
    #             feature,
    #             data_metric_config,
    #             training_args,
    #             max_seq_length=self.max_seq_length,
    #             max_train_samples=self.max_train_samples,
    #             max_valid_samples=self.max_valid_samples,
    #             max_test_samples=self.max_test_samples,
    #         )

    #         trainer.train()
    #         trainer.save_model()
    #         trainer.evaluate()
    #         trainer.predict(test_dataset)
    #         gc.collect()


@slow
@require_deepspeed
class ORTTrainerIntegrationDeepSpeedTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        args = ORTTrainingArguments("..")
        self.n_epochs = min(args.num_train_epochs, 1)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.max_seq_length = 64
        self.max_train_samples = 30
        self.max_valid_samples = 10
        self.max_test_samples = 10

        self.warmup_steps = 10
        self.weight_decay = 0.01

    @parameterized.expand(
        random.sample(
            _get_models_to_test(_ENCODERS_TO_TEST, _ENCODER_TASKS_DATASETS_CONFIGS)
            + _get_models_to_test(_DECODERS_TO_TEST, _DECODER_TASKS_DATASETS_CONFIGS)
            + _get_models_to_test(_SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS),
            1,
        ),
        skip_on_empty=True,
    )
    def test_trainer_fp16_ds_stage1(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = get_ort_training_args(
                feature=feature,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
                fp16=True,
                deepspeed="tests/onnxruntime/ds_configs/ds_config_zero_stage_1.json",
            )

            trainer, _ = get_ort_trainer(
                model_name,
                feature,
                data_metric_config,
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            gc.collect()

    @parameterized.expand(
        random.sample(
            _get_models_to_test(_ENCODERS_TO_TEST, _ENCODER_TASKS_DATASETS_CONFIGS)
            + _get_models_to_test(_DECODERS_TO_TEST, _DECODER_TASKS_DATASETS_CONFIGS)
            + _get_models_to_test(_SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS),
            1,
        ),
        skip_on_empty=True,
    )
    def test_trainer_fp16_ds_stage2(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = get_ort_training_args(
                feature=feature,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
                fp16=True,
                deepspeed="tests/onnxruntime/ds_configs/ds_config_zero_stage_2.json",
            )

            trainer, _ = get_ort_trainer(
                model_name,
                feature,
                data_metric_config,
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            gc.collect()


@slow
class ORTTrainerIntegrationDDPTest(unittest.TestCase):
    def test_trainer_ddp_glue(self):
        subprocess.run(
            "cp examples/onnxruntime/training/text-classification/run_glue.py ./",
            shell=True,
        )

        subprocess.run(
            f"{sys.executable} -m torch.distributed.launch"
            " --nproc_per_node=1"
            " run_glue.py"
            " --model_name_or_path distilbert-base-uncased"
            " --task_name mnli"
            " --max_seq_length 128"
            " --learning_rate 3e-6"
            " --do_train"
            " --output_dir /tmp/distilbert"
            " --overwrite_output_dir"
            " --max_steps 200"
            " --logging_steps 20"
            " --per_device_train_batch_size 32"
            " --fp16 --optim adamw_ort_fused"
            " --max_train_samples 500",
            shell=True,
            check=True,
        )


# List supported ORT optimizers to test
optim_test_params = []
if is_torch_available():
    default_adam_kwargs = {
        "betas": (ORTTrainingArguments.adam_beta1, ORTTrainingArguments.adam_beta2),
        "eps": ORTTrainingArguments.adam_epsilon,
        "lr": ORTTrainingArguments.learning_rate,
    }

    optim_test_params = [
        (
            ORTOptimizerNames.ADAMW_ORT_FUSED,
            onnxruntime.training.optim.FusedAdam,
            default_adam_kwargs,
        ),
    ]


@slow
@require_torch
class ORTTrainerOptimizerChoiceTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        args = ORTTrainingArguments("..")
        self.n_epochs = min(args.num_train_epochs, 1)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.max_seq_length = 64
        self.max_train_samples = 50
        self.max_valid_samples = 20
        self.max_test_samples = 10

        self.warmup_steps = 10
        self.weight_decay = 0.01

        self.model_name = "bert-base-cased"
        self.feature = "sequence-classification"

    def check_optim_and_kwargs(self, optim: OptimizerNames, mandatory_kwargs, expected_cls):
        args = ORTTrainingArguments(optim=optim, output_dir="None")
        actual_cls, optim_kwargs = ORTTrainer.get_ort_optimizer_cls_and_kwargs(args)
        self.assertEqual(expected_cls, actual_cls)
        self.assertIsNotNone(optim_kwargs)

        for p, v in mandatory_kwargs.items():
            self.assertTrue(p in optim_kwargs)
            actual_v = optim_kwargs[p]
            self.assertTrue(actual_v == v, f"Failed check for {p}. Expected {v}, but got {actual_v}.")

    @parameterized.expand(optim_test_params, skip_on_empty=True)
    def test_optim_supported(self, name: str, expected_cls, mandatory_kwargs):
        # exercises all the valid --optim options
        self.check_optim_and_kwargs(name, mandatory_kwargs, expected_cls)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = ORTTrainingArguments(
                optim=name,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
            )

            trainer, _ = get_ort_trainer(
                self.model_name,
                self.feature,
                _ENCODER_TASKS_DATASETS_CONFIGS[self.feature],
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            gc.collect()

    def test_ort_fused_adam(self):
        # Pretend that onnxruntime-training is installed and mock onnxruntime.training.optim.FusedAdam exists.
        # Trainer.get_optimizer_cls_and_kwargs does not use FusedAdam. It only has to return the
        # class given, so mocking onnxruntime.training.optim.FusedAdam should be fine for testing and allow
        # the test to run without requiring an onnxruntime-training installation.
        mock = Mock()
        modules = {
            "onnxruntime.training": mock,
            "onnxruntime.training.optim": mock.optimizers,
            "onnxruntime.training.optim.FusedAdam": mock.optimizers.FusedAdam,
        }
        with patch.dict("sys.modules", modules):
            self.check_optim_and_kwargs(
                ORTOptimizerNames.ADAMW_ORT_FUSED,
                default_adam_kwargs,
                mock.optimizers.FusedAdam,
            )


class ORTSeq2SeqTrainerSpecificIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        args = ORTTrainingArguments("..")
        self.n_epochs = min(args.num_train_epochs, 1)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.max_seq_length = 32
        self.max_train_samples = 10
        self.max_valid_samples = 10
        self.max_test_samples = 10

        self.warmup_steps = 10
        self.weight_decay = 0.01

    @parameterized.expand(
        _get_models_to_test(_SEQ2SEQ_MODELS_TO_TEST, _SEQ2SEQ_TASKS_DATASETS_CONFIGS),
        skip_on_empty=True,
    )
    def test_predict_with_generate_ort(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = get_ort_training_args(
                feature=feature,
                output_dir=tmp_dir,
                evaluation_strategy="epoch",
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
                label_smoothing_factor=0.1,
                predict_with_generate=True,
            )

            trainer, test_dataset = get_ort_trainer(
                model_name,
                feature,
                data_metric_config,
                training_args,
                max_seq_length=self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )

            trainer.train()
            trainer.evaluate(inference_with_ort=True)
            trainer.predict(test_dataset, inference_with_ort=True)
            gc.collect()


if __name__ == "__main__":
    unittest.main()
