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

import datasets
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    default_data_collator,
)

from optimum.onnxruntime import ORTTrainer, Seq2SeqORTTrainer


class TestORTTrainer(unittest.TestCase):
    # @unittest.skip("Skip to just test seq2seq.")
    def test_ort_trainer(self):

        model_names = {"distilbert-base-uncased", "bert-base-cased", "roberta-base", "gpt2", "facebook/bart-base"}
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
                            # output_dir=tmp_dir,
                            output_dir="./results",
                            num_train_epochs=1,
                            per_device_train_batch_size=8,
                            per_device_eval_batch_size=8,
                            warmup_steps=500,
                            weight_decay=0.01,
                            logging_dir=tmp_dir,
                            fp16=True,
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
                        self.assertGreaterEqual(ort_eval_metrics["eval_accuracy"], 0.75)
                        ort_prediction = trainer.predict(test_dataset)
                        print("Training metrics(ORT):\n", train_metrics)
                        print("Evaluation metrics:\n", ort_eval_metrics)
                        print("Prediction results:\n", ort_prediction)

    # @unittest.skip("Skip")
    def test_ort_seq2seq_trainer(self):

        model_names = {"t5-small", "facebook/bart-base"}  # "t5-small", "facebook/bart-base"
        dataset_names = {"xsum"}
        metric_name = "rouge"
        batch_size = 8
        learning_rate = 2e-5
        weight_decay = 0.01
        num_train_epochs = 1
        if_predict_with_generate = {True, False}

        for model_name in model_names:
            for dataset_name in dataset_names:
                for predict_with_generate in if_predict_with_generate:
                    with self.subTest(
                        model_name=model_name, dataset_name=dataset_name, predict_with_generate=predict_with_generate
                    ):
                        with tempfile.TemporaryDirectory() as tmp_dir:

                            # Prepare model
                            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                            tokenizer = AutoTokenizer.from_pretrained(model_name)

                            # Prepare dataset
                            dataset = load_dataset(dataset_name)
                            metric = load_metric(metric_name)
                            label_pad_token_id = tokenizer.pad_token_id

                            if model_name in [
                                "t5-small",
                                "t5-base",
                                "t5-large",
                                "t5-3b",
                                "t5-11b",
                            ] and dataset_name in ["xsum"]:
                                prefix = "summarize: "
                            else:
                                prefix = ""

                            max_input_length = 512
                            max_target_length = 64

                            def preprocess_function(examples):
                                inputs = [prefix + doc for doc in examples["document"]]
                                model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

                                # Setup the tokenizer for targets
                                with tokenizer.as_target_tokenizer():
                                    labels = tokenizer(
                                        examples["summary"], max_length=max_target_length, truncation=True
                                    )

                                model_inputs["labels"] = labels["input_ids"]
                                return model_inputs

                            encoded_dataset = dataset.map(preprocess_function, batched=True)
                            max_train_samples = 200
                            max_valid_samples = 50
                            max_test_samples = 20
                            train_dataset = encoded_dataset["train"]  # .select(range(max_train_samples))
                            valid_dataset = encoded_dataset["validation"]  # .select(range(max_valid_samples))
                            test_dataset = encoded_dataset["test"]  # .select(range(max_test_samples))

                            def compute_metrics(eval_pred):
                                predictions, labels = eval_pred
                                decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                                # Replace -100 in the labels as we can't decode them.
                                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                                # Rouge expects a newline after each sentence
                                decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
                                decoded_labels = [
                                    "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
                                ]

                                result = metric.compute(
                                    predictions=decoded_preds, references=decoded_labels, use_stemmer=True
                                )
                                # Extract a few results
                                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

                                # Add mean generated length
                                prediction_lens = [
                                    np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
                                ]
                                result["gen_len"] = np.mean(prediction_lens)

                                return {k: round(v, 4) for k, v in result.items()}

                            training_args = Seq2SeqTrainingArguments(
                                f"{model_name}-finetuned",
                                evaluation_strategy="epoch",
                                learning_rate=learning_rate,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=weight_decay,
                                save_total_limit=3,
                                num_train_epochs=num_train_epochs,
                                predict_with_generate=False,
                                fp16=True,
                                do_train=True,
                                do_eval=True,
                                label_smoothing_factor=0.1,
                            )

                            data_collator = DataCollatorForSeq2Seq(
                                tokenizer,
                                model=model,
                                label_pad_token_id=label_pad_token_id,
                                pad_to_multiple_of=8 if training_args.fp16 else None,
                            )

                            trainer = Seq2SeqORTTrainer(
                                model=model,
                                args=training_args,
                                train_dataset=train_dataset if training_args.do_train else None,
                                eval_dataset=valid_dataset if training_args.do_eval else None,
                                compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                                tokenizer=tokenizer,
                                data_collator=data_collator,
                                feature="seq2seq-lm",
                            )

                            train_result = trainer.train()
                            trainer.save_model()
                            train_metrics = train_result.metrics
                            ort_eval_metrics = trainer.evaluate()
                            self.assertGreaterEqual(ort_eval_metrics["eval_bleu"], 30)
                            ort_prediction = trainer.predict(test_dataset)
                            print("Training metrics(ORT):\n", train_metrics)
                            print("Evaluation metrics:\n", ort_eval_metrics)
                            print("Prediction results):\n", ort_prediction)


if __name__ == "__main__":
    unittest.main()
