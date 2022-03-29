#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

import os
import tempfile
import unittest

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    default_data_collator,
)
from transformers.utils.fx import symbolic_trace

import yaml
from optimum.intel.neural_compressor import IncOptimizer, IncPruner, IncQuantizer, IncTrainer
from optimum.intel.neural_compressor.configuration import IncPruningConfig, IncQuantizationConfig
from optimum.intel.neural_compressor.pruning import IncPrunerForSequenceClassification
from optimum.intel.neural_compressor.quantization import (
    IncQuantizationMode,
    IncQuantizedModelForSequenceClassification,
    IncQuantizer,
    IncQuantizerForSequenceClassification,
)
from optimum.intel.neural_compressor.trainer import IncTrainer
from optimum.intel.neural_compressor.utils import CONFIG_NAME


os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestINCQuantization(unittest.TestCase):
    def test_quantizer_from_config(self):
        model_name = "bert-base-uncased"
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantization.yml")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        q8_config = IncQuantizationConfig.from_pretrained(config_path)
        quantizer = IncQuantizer(model, q8_config)
        quantizer_from_config = IncQuantizerForSequenceClassification.from_config(model_name, inc_config=config_path)
        self.assertEqual(quantizer_from_config.approach, quantizer.approach)
        self.assertEqual(quantizer_from_config.config.usr_cfg, quantizer.config.usr_cfg)
        self.assertIsInstance(quantizer_from_config.model, BertForSequenceClassification)

    @staticmethod
    def helper(model_name, output_dir, do_train=False, max_train_samples=128, max_eval_samples=128):
        task = "sst2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = load_metric("glue", task)

        if do_train:
            dataset = load_dataset("glue", task)
        else:
            dataset = load_dataset("glue", task, split="validation")

        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )

        if do_train:
            train_dataset = dataset["train"].select(range(max_train_samples))
            eval_dataset = dataset["validation"].select(range(max_eval_samples))
        else:
            train_dataset = None
            eval_dataset = dataset.select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

        training_args = TrainingArguments(output_dir, num_train_epochs=1.0 if do_train else 0.0)

        trainer = IncTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics.get("eval_accuracy")

        return model, trainer, eval_func

    def test_dynamic_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantization.yml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, trainer, eval_func = self.helper(model_name, tmp_dir)
            model_result = eval_func(model)
            q8_config = IncQuantizationConfig.from_pretrained(config_path)
            q8_config.set_config("quantization.approach", IncQuantizationMode.DYNAMIC.value)

            quantizer = IncQuantizer(model, q8_config, eval_func=eval_func)

            q_model = quantizer.fit_dynamic()
            q_model_result = eval_func(q_model.model)

            # Verification accuracy loss is under 2%
            self.assertGreaterEqual(q_model_result, model_result * 0.98)

            trainer.save_model(tmp_dir)
            with open(os.path.join(tmp_dir, CONFIG_NAME), "w") as f:
                yaml.dump(q_model.tune_cfg, f, default_flow_style=False)

            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            loaded_model.eval()
            loaded_model_result = eval_func(loaded_model)

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_result, loaded_model_result)

    def test_static_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantization.yml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, trainer, eval_func = self.helper(model_name, tmp_dir)
            model.config.save_pretrained(tmp_dir)
            model_result = eval_func(model)
            q8_config = IncQuantizationConfig.from_pretrained(config_path)
            q8_config.set_config("quantization.approach", IncQuantizationMode.STATIC.value)
            q8_config.set_config("tuning.accuracy_criterion.relative", 0.04)
            q8_config.set_config("model.framework", "pytorch_fx")
            input_names = ["input_ids", "attention_mask", "token_type_ids", "labels"]

            model = symbolic_trace(
                model,
                input_names=input_names,
                batch_size=8,
                sequence_length=128,
            )

            quantizer = IncQuantizer(model, q8_config)
            quantizer.eval_func = eval_func
            quantizer.calib_dataloader = trainer.get_eval_dataloader()
            q_model = quantizer.fit_static()
            q_model_result = eval_func(q_model.model)

            # Verification accuracy loss is under 4%
            self.assertGreaterEqual(q_model_result, model_result * 0.96)

            trainer.save_model(tmp_dir)
            with open(os.path.join(tmp_dir, CONFIG_NAME), "w") as f:
                yaml.dump(q_model.tune_cfg, f, default_flow_style=False)

            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(
                tmp_dir,
                input_names=input_names,
                batch_size=8,
                sequence_length=128,
            )
            loaded_model.eval()
            loaded_model_result = eval_func(loaded_model)

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_result, loaded_model_result)


class TestINCOptimizer(unittest.TestCase):
    def test_pruning_quantization_aware_training(self):

        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sst2"
        max_eval_samples = 64
        max_train_samples = 64
        target_sparsity = 0.02

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = load_metric("glue", task)
        dataset = load_dataset("glue", task)
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(max_train_samples))
        eval_dataset = dataset["validation"].select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

        def train_func(model):
            trainer.model_wrapped = model
            trainer.model = model
            _ = trainer.train(pruner)
            return trainer.model

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics.get("eval_accuracy")

        config_path = os.path.dirname(os.path.abspath(__file__))

        q8_config = IncQuantizationConfig.from_pretrained(config_path, config_file_name="quantization.yml")
        q8_config.set_config("quantization.approach", IncQuantizationMode.DYNAMIC.value)

        pruning_config = IncPruningConfig.from_pretrained(config_path, config_file_name="prune.yml")
        pruning_config.set_config("pruning.approach.weight_compression.start_epoch", 0)
        pruning_config.set_config("pruning.approach.weight_compression.end_epoch", 1)
        pruning_config.set_config("pruning.approach.weight_compression.initial_sparsity", 0.0)
        pruning_config.set_config("pruning.approach.weight_compression.target_sparsity", target_sparsity)

        inc_quantizer = IncQuantizer(model, q8_config, eval_func=eval_func)
        quantizer = inc_quantizer.fit()

        inc_pruner = IncPruner(model, pruning_config, eval_func=eval_func, train_func=train_func)
        pruner = inc_pruner.fit()

        inc_optimizer = IncOptimizer(model, quantizer=quantizer, pruner=pruner)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(tmp_dir, num_train_epochs=2.0)

            trainer = IncTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )

            model_result = eval_func(model)
            opt_model = inc_optimizer.fit()

            opt_model_result = eval_func(opt_model.model)
            _, sparsity = opt_model.report_sparsity()

            trainer.save_model(tmp_dir)
            with open(os.path.join(tmp_dir, CONFIG_NAME), "w") as f:
                yaml.dump(opt_model.tune_cfg, f, default_flow_style=False)

            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            loaded_model.eval()
            loaded_model_result = eval_func(loaded_model)

            # Verification final sparsity is equal to the targeted sparsity
            self.assertEqual(round(sparsity), target_sparsity * 100)

            # Verification accuracy loss is under 5%
            self.assertGreaterEqual(opt_model_result, model_result * 0.95)

            # Verification quantized model was correctly loaded
            self.assertEqual(opt_model_result, loaded_model_result)


if __name__ == "__main__":
    unittest.main()
