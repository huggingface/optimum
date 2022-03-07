#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tempfile
import unittest
from enum import Enum
from pathlib import Path

import datasets
import numpy as np
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, default_data_collator
from transformers.onnx import validate_model_outputs
from transformers.onnx.features import FeaturesManager

from optimum.onnxruntime.trainer import ORTTrainer


class TestORTTrainer(unittest.TestCase):
    def test_ort_trainer(self):

        model_names = {
            "gpt2"
        }  # "gpt2", "distilbert-base-uncased", "bert-base-cased", "roberta-base", "facebook/bart-base"
        dataset_names = {"sst2"}  # glue
        modes = {"ort_infer"}  # "ort_train", "ort_infer"

        def test_ort_train_or_infer(trainer, mode):
            if mode == "ort_train":
                test_ort_training(trainer)
            elif mode == "ort_infer":
                test_ort_inference(trainer)

        def test_ort_training(trainer):
            # Test 1: ORT training + PyTorch Inference
            train_result = trainer.train()
            train_metrics = train_result.metrics
            trainer.save_model()
            trainer.log_metrics("train", train_metrics)
            trainer.save_metrics("train", train_metrics)
            trainer.save_state()
            eval_metrics = trainer.evaluate(ort=False)
            prediction = trainer.predict(test_dataset, ort=False)
            print("Train metrics:\n", train_metrics)
            print("Evaluation metrics(PT):\n", eval_metrics)
            print("Prediction results(PT):\n", prediction)

        def test_ort_inference(trainer):
            # Test 2: PyTorch Training + ORT Inference
            train_result = trainer.train(ort=False)
            train_metrics = train_result.metrics
            ort_eval_metrics = trainer.evaluate()
            ort_prediction = trainer.predict(test_dataset)
            print("Evaluation metrics(ORT):\n", ort_eval_metrics)
            print("Prediction results(ORT):\n", ort_prediction)

        for mode in modes:
            for model_name in model_names:
                for dataset_name in dataset_names:
                    with self.subTest(mode=mode, model_name=model_name, dataset_name=dataset_name):
                        with tempfile.TemporaryDirectory() as tmp_dir:

                            # Prepare model
                            model = AutoModelForSequenceClassification.from_pretrained(model_name)
                            tokenizer = AutoTokenizer.from_pretrained(model_name)

                            # Prepare dataset
                            dataset = load_dataset("glue", dataset_name)
                            metric = load_metric("glue", dataset_name)

                            max_seq_length = min(64, tokenizer.model_max_length)
                            padding = "max_length"

                            if tokenizer.pad_token is None:
                                tokenizer.pad_token = tokenizer.eos_token
                                model.config.pad_token_id = model.config.eos_token_id

                            def preprocess_function(examples):
                                args = (examples["sentence"],)
                                return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

                            encoded_dataset = dataset.map(preprocess_function, batched=True)
                            max_train_samples = 500
                            max_valid_samples = 100
                            max_test_samples = 20
                            train_dataset = encoded_dataset["train"].select(range(max_train_samples))
                            valid_dataset = encoded_dataset["validation"].select(range(max_valid_samples))
                            test_dataset = (
                                encoded_dataset["test"].remove_columns(["label"]).select(range(max_test_samples))
                            )

                            def compute_metrics(eval_pred):
                                predictions = (
                                    eval_pred.predictions[0]
                                    if isinstance(eval_pred.predictions, tuple)
                                    else eval_pred.predictions
                                )
                                if dataset_name != "stsb":
                                    predictions = np.argmax(predictions, axis=1)
                                else:
                                    predictions = predictions[:, 0]
                                return metric.compute(predictions=predictions, references=eval_pred.label_ids)

                            training_args = TrainingArguments(
                                output_dir=tmp_dir,
                                num_train_epochs=1,
                                per_device_train_batch_size=8,
                                per_device_eval_batch_size=8,
                                warmup_steps=500,
                                weight_decay=0.01,
                                logging_dir=tmp_dir,
                            )

                            trainer = ORTTrainer(
                                model=model,
                                args=training_args,
                                train_dataset=train_dataset,
                                eval_dataset=valid_dataset,
                                compute_metrics=compute_metrics,
                                tokenizer=tokenizer,
                                data_collator=default_data_collator,
                            )

                            test_ort_train_or_infer(trainer, mode)


if __name__ == "__main__":
    unittest.main()
