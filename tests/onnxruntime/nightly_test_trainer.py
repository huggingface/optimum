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
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, patch

import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    IntervalStrategy,
    TrainingArguments,
    default_data_collator,
    is_torch_available,
)
from transformers.onnx.features import FeaturesManager
from transformers.testing_utils import (
    is_staging_test,
    require_deepspeed,
    require_fairscale,
    require_onnx,
    require_optuna,
    require_ray,
    require_sigopt,
    require_torch,
    require_torch_gpu,
    require_wandb,
    slow,
)
from transformers.trainer_utils import (
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_sigopt,
    default_hp_space_wandb,
)

from optimum.onnxruntime import ORTSeq2SeqTrainer, ORTTrainer
from optimum.utils.testing_utils import require_hf_token, require_ort_training, require_sigopt_token_and_project
from parameterized import parameterized


nltk.download("punkt")

_MODELS_TO_TEST = {
    ("bert", "bert-base-cased"),
    # ("distilbert", "distilbert-base-cased"),
    # ("roberta", "roberta-base"),
    # ("gpt2", "gpt2"),
}

_TASKS_DATASETS_CONFIGS = {
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


def _get_models_to_test(model_list, task_list, excluded: Optional[List[str]] = None):
    models_to_test = []

    for name, model_name in model_list:
        for feature, data_metric_config in task_list.items():
            if excluded and name in excluded:
                continue
            models_to_test.append((f"{name}_{feature}", model_name, feature, data_metric_config))

    return sorted(models_to_test)


def _get_data_collator(data_metric_config, tokenizer=None):

    if "data_collator" in data_metric_config.keys():
        data_collator = data_metric_config["data_collator"]
    elif "data_collator_class" in data_metric_config.keys():
        data_collator = data_metric_config["data_collator_class"](tokenizer, pad_to_multiple_of=8)
    else:
        raise KeyError("You need to pass either `data_collator` or `data_collator_class` to create the data collator.")

    return data_collator


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

    (model, tokenizer, data_collator, train_dataset, valid_dataset, test_dataset, compute_metrics,) = load_and_prepare(
        feature
    )(
        model_name,
        data_metric_config,
        max_seq_length,
        max_train_samples=max_train_samples,
        max_valid_samples=max_valid_samples,
        max_test_samples=max_test_samples,
    )

    trainer = ORTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        feature=feature,
    )

    return trainer, test_dataset


def load_and_prepare(feature):
    preprocess_mapping = {
        "sequence-classification": load_and_prepare_glue,
        "token-classification": load_and_prepare_ner,
    }
    return preprocess_mapping[feature]


def load_and_prepare_glue(model_name, data_metric_config, max_seq_length, padding="max_length", **kwargs):

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare dataset
    dataset = load_dataset(*data_metric_config["dataset"])
    metric = load_metric(*data_metric_config["metric"])

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

    return model, tokenizer, data_collator, train_dataset, valid_dataset, test_dataset, compute_metrics


def load_and_prepare_ner(model_name, data_metric_config, max_seq_length, padding="max_length", **kwargs):

    # Load dataset and metric
    dataset = load_dataset(*data_metric_config["dataset"])
    metric = load_metric(*data_metric_config["metric"])
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
    test_dataset = tokenized_datasets["test"]

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

    return model, tokenizer, data_collator, train_dataset, valid_dataset, test_dataset, compute_metrics


@unittest.skip("Skip basic tests of `ORTTrainer`.")
@require_torch
# @require_ort_training
class ORTTrainerIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = min(args.num_train_epochs, 1)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.max_seq_length = 128
        self.max_train_samples = 200
        self.max_valid_samples = 50
        self.max_test_samples = 20

        self.warmup_steps = 500
        self.weight_decay = 0.01

    @parameterized.expand(_get_models_to_test(_MODELS_TO_TEST, _TASKS_DATASETS_CONFIGS), skip_on_empty=True)
    def test_trainer_inference_with_ort(self, test_name, model_name, feature, data_metric_config):

        with tempfile.TemporaryDirectory() as tmp_dir:

            training_args = TrainingArguments(
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

            train_result = trainer.train()
            trainer.save_model()
            train_metrics = train_result.metrics
            eval_metrics = trainer.evaluate(inference_with_ort=True)
            self.assertGreaterEqual(eval_metrics["eval_accuracy"], 0.75)
            prediction = trainer.predict(test_dataset, inference_with_ort=True)
            print("Training metrics:\n", train_metrics)
            print("Evaluation metrics:\n", eval_metrics)
            print("Prediction results:\n", prediction)
            gc.collect()

    @parameterized.expand(_get_models_to_test(_MODELS_TO_TEST, _TASKS_DATASETS_CONFIGS), skip_on_empty=True)
    def test_trainer_inference_with_pytorch(self, test_name, model_name, feature, data_metric_config):

        with tempfile.TemporaryDirectory() as tmp_dir:

            training_args = TrainingArguments(
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

            train_result = trainer.train()
            trainer.save_model()
            train_metrics = train_result.metrics
            eval_metrics = trainer.evaluate()
            self.assertGreaterEqual(eval_metrics["eval_accuracy"], 0.75)
            prediction = trainer.predict(test_dataset)
            print("Training metrics:\n", train_metrics)
            print("Evaluation metrics:\n", eval_metrics)
            print("Prediction results:\n", prediction)
            gc.collect()

    @parameterized.expand(
        _get_models_to_test(_MODELS_TO_TEST, _TASKS_DATASETS_CONFIGS, excluded=["gpt2"]), skip_on_empty=True
    )
    def test_trainer_fp16(self, test_name, model_name, feature, data_metric_config):

        with tempfile.TemporaryDirectory() as tmp_dir:

            training_args = TrainingArguments("..")

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

            train_result = trainer.train()
            trainer.save_model()
            train_metrics = train_result.metrics
            eval_metrics = trainer.evaluate()
            self.assertGreaterEqual(eval_metrics["eval_accuracy"], 0.75)
            prediction = trainer.predict(test_dataset)
            print("Training metrics:\n", train_metrics)
            print("Evaluation metrics:\n", eval_metrics)
            print("Prediction results:\n", prediction)
            gc.collect()

    @slow
    @parameterized.expand(_get_models_to_test(_MODELS_TO_TEST, _TASKS_DATASETS_CONFIGS), skip_on_empty=True)
    def test_trainer_bf16(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:

            training_args = TrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
                bf16=True,
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

            train_result = trainer.train()
            trainer.save_model()
            train_metrics = train_result.metrics
            eval_metrics = trainer.evaluate()
            self.assertGreaterEqual(eval_metrics["eval_accuracy"], 0.75)
            prediction = trainer.predict(test_dataset)
            print("Training metrics:\n", train_metrics)
            print("Evaluation metrics:\n", eval_metrics)
            print("Prediction results:\n", prediction)
            gc.collect()


class ORTTrainerIntegrationWithHubTester(unittest.TestCase):
    @require_hf_token
    def test_push_to_hub(
        self,
        test_name="bert_sequence-classification",
        model_name="bert-base-cased",
        feature="sequence-classification",
        data_metric_config=_TASKS_DATASETS_CONFIGS["sequence-classification"],
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                push_to_hub=True,
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

            trainer.push_to_hub()


@unittest.skip("Skip DeepSpeed tests of ORTTrainer.")
@slow
@require_deepspeed
class ORTTrainerIntegrationDeepSpeedTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = min(args.num_train_epochs, 1)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.max_seq_length = 128
        self.max_train_samples = 200
        self.max_valid_samples = 50
        self.max_test_samples = 20

        self.warmup_steps = 500
        self.weight_decay = 0.01

    @parameterized.expand(
        _get_models_to_test(_MODELS_TO_TEST, _TASKS_DATASETS_CONFIGS, excluded=["gpt2"]), skip_on_empty=True
    )
    def test_trainer_fp16_ds_stage1(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:

            training_args = TrainingArguments(
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

            train_result = trainer.train()
            trainer.save_model()
            train_metrics = train_result.metrics
            eval_metrics = trainer.evaluate()
            # self.assertGreaterEqual(eval_metrics["eval_accuracy"], 0.75)
            prediction = trainer.predict(test_dataset)
            print("Training metrics:\n", train_metrics)
            print("Evaluation metrics:\n", eval_metrics)
            print("Prediction results:\n", prediction)
            gc.collect()

    @parameterized.expand(
        _get_models_to_test(_MODELS_TO_TEST, _TASKS_DATASETS_CONFIGS, excluded=["gpt2"]), skip_on_empty=True
    )
    def test_trainer_fp16_ds_stage2(self, test_name, model_name, feature, data_metric_config):
        with tempfile.TemporaryDirectory() as tmp_dir:

            training_args = TrainingArguments(
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

            train_result = trainer.train()
            trainer.save_model()
            train_metrics = train_result.metrics
            eval_metrics = trainer.evaluate()
            # self.assertGreaterEqual(eval_metrics["eval_accuracy"], 0.75)
            prediction = trainer.predict(test_dataset)
            print("Training metrics:\n", train_metrics)
            print("Evaluation metrics:\n", eval_metrics)
            print("Prediction results:\n", prediction)
            gc.collect()


class ORTTrainerHyperParameterIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "bert-base-cased"
        self.feature = "sequence-classification"

        self.max_seq_length = 128
        self.max_train_samples = 200
        self.max_valid_samples = 50
        self.max_test_samples = 20

    @unittest.skip("Skip the hyperparameter search with Optuna")
    @require_optuna
    def test_hyperparameter_search_optuna(self):
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, return_dict=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
            )

            (_, tokenizer, data_collator, train_dataset, valid_dataset, _, compute_metrics,) = load_and_prepare(
                self.feature
            )(
                self.model_name,
                _TASKS_DATASETS_CONFIGS[self.feature],
                self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )
            trainer = ORTTrainer(
                model=None,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                feature=self.feature,
                model_init=model_init,
            )

            trainer.hyperparameter_search(direction="minimize", hp_space=default_hp_space_optuna, n_trials=2)
            gc.collect()

    @unittest.skip("Skip the hyperparameter search with Ray")
    @require_ray
    def test_hyperparameter_search_ray(self):
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, return_dict=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
            )

            (_, tokenizer, data_collator, train_dataset, valid_dataset, _, compute_metrics,) = load_and_prepare(
                self.feature
            )(
                self.model_name,
                _TASKS_DATASETS_CONFIGS[self.feature],
                self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )
            trainer = ORTTrainer(
                model=None,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                feature=self.feature,
                model_init=model_init,
            )

            trainer.hyperparameter_search(
                direction="minimize", backend="ray", hp_space=default_hp_space_ray, n_trials=2
            )
            gc.collect()

    @unittest.skip("Skip the hyperparameter search with SigOpt")
    @require_sigopt
    @require_sigopt_token_and_project
    def test_hyperparameter_search_sigopt(self):
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, return_dict=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
            )

            (_, tokenizer, data_collator, train_dataset, valid_dataset, _, compute_metrics,) = load_and_prepare(
                self.feature
            )(
                self.model_name,
                _TASKS_DATASETS_CONFIGS[self.feature],
                self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )
            trainer = ORTTrainer(
                model=None,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                feature=self.feature,
                model_init=model_init,
            )

            trainer.hyperparameter_search(
                direction="minimize", backend="sigopt", hp_space=default_hp_space_sigopt, n_trials=2
            )
            gc.collect()

    @unittest.skip("Skip the hyperparameter search with WanB")
    @require_wandb
    def test_hyperparameter_search_wandb(self):
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, return_dict=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
            )

            (_, tokenizer, data_collator, train_dataset, valid_dataset, _, compute_metrics,) = load_and_prepare(
                self.feature
            )(
                self.model_name,
                _TASKS_DATASETS_CONFIGS[self.feature],
                self.max_seq_length,
                max_train_samples=self.max_train_samples,
                max_valid_samples=self.max_valid_samples,
                max_test_samples=self.max_test_samples,
            )
            trainer = ORTTrainer(
                model=None,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                feature=self.feature,
                model_init=model_init,
            )

            trainer.hyperparameter_search(
                direction="minimize", backend="wandb", hp_space=default_hp_space_wandb, n_trials=2, anonymous="must"
            )
            gc.collect()


@unittest.skip("Skip")
@require_torch
class ORTTrainerOptimizerChoiceTest(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
