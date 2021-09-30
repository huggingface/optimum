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

import unittest
import os
import numpy as np
import tempfile
import yaml
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    default_data_collator,
    TrainingArguments,
)
from datasets import load_dataset, load_metric

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestLPOT(unittest.TestCase):

    def test_dynamic_quantization(self):

        from optimum.intel.lpot.quantization import (
            LpotQuantizer,
            LpotQuantizedModelForSequenceClassification,
        )

        model_name = "textattack/bert-base-uncased-SST-2"
        task = "sst2"
        max_eval_samples = 100
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = load_metric("glue", task)
        eval_dataset = load_dataset("glue", task, split="validation")
        eval_dataset = eval_dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128),
            batched=True
        )
        eval_dataset = eval_dataset.select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

        trainer = Trainer(
            model=model,
            train_dataset=None,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics.get("eval_accuracy")

        model_metric = eval_func(model)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantization.yml")

        quantizer = LpotQuantizer(config_path, model, eval_func=eval_func)

        q_model = quantizer.fit_dynamic()
        q_model_metric = eval_func(q_model.model)

        # Verification accuracy loss is under 2%
        self.assertTrue(q_model_metric >= model_metric * 0.98)

        # Verification model saving and loading
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer.save_model(tmp_dir)
            with open(os.path.join(tmp_dir, "lpot_config.yml"), 'w') as f:
                yaml.dump(q_model.tune_cfg, f, default_flow_style=False)

            from optimum.intel.lpot.quantization import LpotQuantizedModelForSequenceClassification

            loaded_model = LpotQuantizedModelForSequenceClassification.from_pretrained(
                model_name_or_path=tmp_dir,
                q_model_name="pytorch_model.bin",
                config_name="lpot_config.yml",
            )
            loaded_model.eval()
            loaded_model_metric = eval_func(loaded_model)
            self.assertEqual(q_model_metric, loaded_model_metric)

    def test_quantization_from_config(self):

        from optimum.intel.lpot.quantization import (
            LpotQuantizerForSequenceClassification,
            LpotQuantizedModelForSequenceClassification,
        )

        model_name = "textattack/bert-base-uncased-SST-2"
        task = "sst2"
        max_eval_samples = 100
        config_dir = os.path.dirname(os.path.abspath(__file__))

        quantizer = LpotQuantizerForSequenceClassification.from_config(
            config_dir,
            "quantization.yml",
            model_name_or_path=model_name,
        )
        tokenizer = quantizer.tokenizer
        model = quantizer.model

        metric = load_metric("glue", task)
        eval_dataset = load_dataset("glue", task, split="validation")
        eval_dataset = eval_dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128),
            batched=True
        )
        eval_dataset = eval_dataset.select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

        trainer = Trainer(
            model=model,
            train_dataset=None,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics.get("eval_accuracy")

        model_metric = eval_func(model)

        quantizer.eval_func = eval_func

        q_model = quantizer.fit_dynamic()
        q_model_metric = eval_func(q_model.model)

        # Verification accuracy loss is under 2%
        self.assertTrue(q_model_metric >= model_metric * 0.98)

        # Verification model saving and loading
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer.save_model(tmp_dir)
            with open(os.path.join(tmp_dir, "lpot_config.yml"), 'w') as f:
                yaml.dump(q_model.tune_cfg, f, default_flow_style=False)

            loaded_model = LpotQuantizedModelForSequenceClassification.from_pretrained(
                model_name_or_path=tmp_dir,
                q_model_name="pytorch_model.bin",
                config_name="lpot_config.yml",
            )
            loaded_model.eval()
            loaded_model_metric = eval_func(loaded_model)
            self.assertEqual(q_model_metric, loaded_model_metric)


if __name__ == "__main__":
    unittest.main()
