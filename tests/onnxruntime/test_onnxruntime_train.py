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
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartTokenizer,
    TrainingArguments,
    default_data_collator,
)
from transformers.onnx import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager
from transformers.trainer import Trainer

from optimum.onnxruntime import ORTTrainer


class TestORTTrainer(unittest.TestCase):
    def test_ort_trainer(self):

        model_names = {"distilbert-base-uncased", "bert-base-cased", "roberta-base"}
        # "gpt2", "distilbert-base-uncased", "bert-base-cased", "roberta-base", "facebook/bart-base"
        dataset_names = {"sst2"}  # glue

        for model_name in model_names:
            for dataset_name in dataset_names:
                with self.subTest(model_name=model_name, dataset_name=dataset_name):
                    with tempfile.TemporaryDirectory() as tmp_dir:

                        # Prepare model
                        model = AutoModelForSequenceClassification.from_pretrained(model_name)
                        tokenizer = AutoTokenizer.from_pretrained(model_name)

                        # Prepare dataset
                        dataset = load_dataset("glue", dataset_name)
                        metric = load_metric("glue", dataset_name)

                        max_seq_length = min(128, tokenizer.model_max_length)
                        padding = "max_length"

                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                            model.config.pad_token_id = model.config.eos_token_id

                        def preprocess_function(examples):
                            args = (examples["sentence"],)
                            return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

                        encoded_dataset = dataset.map(preprocess_function, batched=True)
                        max_train_samples = 200
                        max_valid_samples = 50
                        max_test_samples = 20
                        train_dataset = encoded_dataset["train"]  # .select(range(max_train_samples))
                        valid_dataset = encoded_dataset["validation"]  # .select(range(max_valid_samples))
                        test_dataset = encoded_dataset["test"].remove_columns(
                            ["label"]
                        )  # .select(range(max_test_samples))

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
                            feature="sequence-classification",
                        )

                        train_result = trainer.train()
                        trainer.save_model()
                        train_metrics = train_result.metrics
                        ort_eval_metrics = trainer.evaluate()
                        ort_prediction = trainer.predict(test_dataset)
                        print("Training metrics(ORT):\n", train_metrics)
                        print("Evaluation metrics(ORT):\n", ort_eval_metrics)
                        print("Prediction results(ORT):\n", ort_prediction)


if __name__ == "__main__":
    unittest.main()
