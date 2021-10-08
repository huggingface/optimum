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
import tempfile
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset, load_metric
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestLPOT(unittest.TestCase):

    def helper(self, model_name, output_dir, do_train=False, max_train_samples=512):

        task = "sst2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = load_metric("glue", task)

        if do_train:
            dataset = load_dataset("glue", task)
        else:
            dataset = load_dataset("glue", task, split="validation")

        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128),
            batched=True
        )

        if do_train:
            train_dataset = dataset["train"].select(range(max_train_samples))
            eval_dataset = dataset["validation"]
        else:
            train_dataset = None
            eval_dataset = dataset

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

        training_args = TrainingArguments(
            output_dir,
            num_train_epochs=1.0 if do_train else 0.0
        )

        trainer = Trainer(
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

        from optimum.intel.lpot.config import LpotConfig
        from optimum.intel.lpot.quantization import (
            LpotQuantizer,
            LpotQuantizedModelForSequenceClassification,
            LpotQuantizationMode,
        )
        import yaml

        model_name = "textattack/bert-base-uncased-SST-2"
        config_dir = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, trainer, eval_func = self.helper(model_name, tmp_dir)
            model_metric = eval_func(model)
            save_path = os.path.join(tmp_dir, "quantization.yml")
            q8_config = LpotConfig.from_pretrained(config_dir, "quantization.yml", save_path=save_path)
            q8_config.set_config("quantization.approach", LpotQuantizationMode.DYNAMIC.value)

            quantizer = LpotQuantizer(q8_config.path, model, eval_func=eval_func)

            q_model = quantizer.fit_dynamic()
            q_model_metric = eval_func(q_model.model)

            # Verification accuracy loss is under 2%
            self.assertTrue(q_model_metric >= model_metric * 0.98)

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

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_metric, loaded_model_metric)

    def test_static_quantization(self):

        from optimum.intel.lpot.config import LpotConfig
        from optimum.intel.lpot.quantization import (
            LpotQuantizer,
            LpotQuantizedModelForSequenceClassification,
            LpotQuantizationMode,
        )
        from transformers.utils.fx import symbolic_trace
        import yaml

        model_name = "textattack/bert-base-uncased-SST-2"
        config_dir = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, trainer, eval_func = self.helper(model_name, tmp_dir)
            model_metric = eval_func(model)
            save_path = os.path.join(tmp_dir, "quantization.yml")
            q8_config = LpotConfig.from_pretrained(config_dir, "quantization.yml", save_path=save_path)
            q8_config.set_config("quantization.approach", LpotQuantizationMode.STATIC.value)
            q8_config.set_config("tuning.accuracy_criterion.relative", 0.04)
            q8_config.set_config("model.framework", "pytorch_fx")

            model = symbolic_trace(
                model,
                input_names=["input_ids", "attention_mask", "token_type_ids", "labels"],
                batch_size=8,
                sequence_length=128
            )

            quantizer = LpotQuantizer(q8_config.path, model)
            quantizer.eval_func = eval_func
            quantizer.calib_dataloader = trainer.get_eval_dataloader()
            q_model = quantizer.fit_static()
            q_model_metric = eval_func(q_model.model)

            # Verification accuracy loss is under 4%
            self.assertTrue(q_model_metric >= model_metric * 0.96)

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

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_metric, loaded_model_metric)

    def test_aware_training_quantization(self):

        from optimum.intel.lpot.config import LpotConfig
        from optimum.intel.lpot.quantization import (
            LpotQuantizer,
            LpotQuantizedModelForSequenceClassification,
            LpotQuantizationMode,
        )
        from transformers.utils.fx import symbolic_trace
        import yaml

        model_name = "textattack/bert-base-uncased-SST-2"
        config_dir = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, trainer, eval_func = self.helper(model_name, tmp_dir, do_train=True)
            model_metric = eval_func(model)
            save_path = os.path.join(tmp_dir, "quantization.yml")
            q8_config = LpotConfig.from_pretrained(config_dir, "quantization.yml", save_path=save_path)
            q8_config.set_config("quantization.approach", LpotQuantizationMode.AWARE_TRAINING.value)
            q8_config.set_config("tuning.accuracy_criterion.relative", 0.03)
            q8_config.set_config("model.framework", "pytorch_fx")

            model = symbolic_trace(
                model,
                input_names=["input_ids", "attention_mask", "token_type_ids", "labels"],
                batch_size=8,
                sequence_length=128
            )

            def train_func(model):
                trainer.model_wrapped = model
                trainer.model = model
                _ = trainer.train()

            quantizer = LpotQuantizer(q8_config.path, model)
            quantizer.eval_func = eval_func
            quantizer.train_func = train_func

            q_model = quantizer.fit_aware_training()
            q_model_metric = eval_func(q_model.model)

            # Verification accuracy loss is under 3%
            self.assertTrue(q_model_metric >= model_metric * 0.97)

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

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_metric, loaded_model_metric)

    def test_quantization_from_config(self):

        from optimum.intel.lpot.quantization import (
            LpotQuantizerForSequenceClassification,
            LpotQuantizedModelForSequenceClassification,
        )
        import yaml

        model_name = "textattack/bert-base-uncased-SST-2"
        task = "sst2"
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

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

        with tempfile.TemporaryDirectory() as tmp_dir:

            trainer = Trainer(
                model=model,
                args=TrainingArguments(tmp_dir),
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

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_metric, loaded_model_metric)


if __name__ == "__main__":
    unittest.main()

