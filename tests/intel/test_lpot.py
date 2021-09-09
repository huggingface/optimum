import unittest
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
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
from optimus.intel.lpot.quantization import LpotQuantizerForSequenceClassification


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class TestLPOT(unittest.TestCase):

    def test_quantization(self):
        model_name = "textattack/bert-base-uncased-SST-2"
        config_path = os.path.dirname(os.path.abspath(__file__))
        task = "sst2"
        padding = "max_length"
        max_seq_length = 128
        max_eval_samples = 200
        metric_name = "eval_accuracy"
        dataset = load_dataset("glue", task)
        metric = load_metric("glue", task)
        data_collator = default_data_collator
        sentence1_key, sentence2_key = task_to_keys[task]

        quantizer = LpotQuantizerForSequenceClassification.from_config(
            config_path,
            "quantization.yml",
            model_name_or_path=model_name,
        )

        tokenizer = quantizer.tokenizer
        model = quantizer.model

        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
            return result

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        eval_dataset = encoded_dataset["validation"]
        eval_dataset = eval_dataset.select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

        trainer = Trainer(
            model=model,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        def take_eval_steps(model, trainer, metric_name):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics.get(metric_name)

        def eval_func(model):
            return take_eval_steps(model, trainer, metric_name)

        quantizer.eval_func = eval_func

        q_model = quantizer.fit_dynamic()
        metric = take_eval_steps(q_model.model, trainer, metric_name)
        print(f"Quantized model obtained with {metric_name} of {metric}.")


if __name__ == "__main__":
    unittest.main()

