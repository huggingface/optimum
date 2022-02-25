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

import numpy as np
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, default_data_collator
from transformers.onnx import validate_model_outputs
from transformers.onnx.features import FeaturesManager

from optimum.onnxruntime.trainer import ORTTrainer


class TestORTTrainer(unittest.TestCase):
    def test_ort_trainer(self):

        model_names = {
            "bert-base-cased"
        }  # "gpt2", "distilbert-base-uncased", "bert-base-cased", "roberta-base", "facebook/bart-base"
        dataset_names = {"sst2"}  # glue

        for model_name in model_names:
            for dataset_name in dataset_names:
                with self.subTest(model_name=model_name, dataset_name=dataset_name):
                    with tempfile.TemporaryDirectory() as tmp_dir:

                        # Prepare model
                        feature = "default"
                        model = BertForSequenceClassification.from_pretrained(model_name)
                        tokenizer = AutoTokenizer.from_pretrained(model_name)

                        # Prepare dataset
                        dataset = load_dataset("glue", dataset_name)
                        metric = load_metric("glue", dataset_name)

                        max_seq_length = min(128, tokenizer.model_max_length)
                        padding = "max_length"

                        def preprocess_function(examples):
                            args = (examples["sentence"],)
                            return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

                        encoded_dataset = dataset.map(preprocess_function, batched=True)
                        max_train_samples = 1000
                        max_valid_samples = 200
                        max_test_samples = 20
                        train_dataset = encoded_dataset["train"].select(range(max_train_samples))
                        valid_dataset = encoded_dataset["validation"].select(range(max_valid_samples))
                        test_dataset = (
                            encoded_dataset["test"].remove_columns(["label"]).select(range(max_test_samples))
                        )

                        def compute_metrics(eval_pred):
                            predictions, labels = eval_pred
                            if dataset_name != "stsb":
                                predictions = np.argmax(predictions, axis=1)
                            else:
                                predictions = predictions[:, 0]
                            return metric.compute(predictions=predictions, references=labels)

                        training_args = TrainingArguments(
                            output_dir="./results",  # './results'
                            num_train_epochs=1,
                            per_device_train_batch_size=16,
                            per_device_eval_batch_size=16,  # As for onnxruntime, the training and the evlaution shall set the same barch size
                            warmup_steps=500,
                            weight_decay=0.01,
                            logging_dir=tmp_dir,  # './logs'
                            # deepspeed="ds_config_zero2.json",  # Test the compatibility of deepspeed and ORTModule
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

                        # Test 2: (ORT Training) + ORT Inference
                        # -------- ORT training part --------
                        # train_result = trainer.train()
                        # train_metrics = train_result.metrics
                        # -------- ORT inference part --------
                        # ort_eval_metrics = trainer.evaluate()
                        # ort_prediction = trainer.predict(test_dataset)
                        # print("Evaluation metrics(ORT):\n", ort_eval_metrics)
                        # print("Prediction results(ORT):\n", ort_prediction)


if __name__ == "__main__":
    unittest.main()
