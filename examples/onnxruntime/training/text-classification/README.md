<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Text classification

By running the script [`run_classification.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/text-classification/run_classification.py),
we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) accelerator to fine-tune the models from the
[HuggingFace hub](https://huggingface.co/models) for text classification task.


__The following example applies the acceleration features powered by ONNX Runtime.__


### ONNX Runtime Training

The following example fine-tunes [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) on the [Amazon Reviews Dataset](https://huggingface.co/datasets/amazon_reviews_multi).

```bash
torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE run_classification.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name amazon_reviews_multi \
    --dataset_config_name en \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name 'review_title,review_body,product_category' \
    --text_column_delimiter ' ' \
    --label_column_name stars \
    --do_train \
    --do_eval \
    --fp16 \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --deepspeed zero_stage_2.json \
    --use_peft \
    --output_dir /tmp/ort-llama-2/
```

### Performance

We get the following results for [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) using mixed-precision-training/LoRA/ZeRO-Stage-2 under PyTorch and ONNX Runtime backends. 8 Nvidia V100 cards were used to run the
experiment for 10 epochs:

| Model                       | Backend      | Runtime(s)      | Train samples(/s)   |
| --------------------------- |------------- | --------------- | ------------------- |
| meta-llama/Llama-2-7b-hf    | PyTorch      | 17035.9055      | 117.399             |
| meta-llama/Llama-2-7b-hf    | ONNX Runtime | 15532.2403      | 128.764             |

We observe the gain of ONNX Runtime compared to PyTorch as follow:

| Model                     | Latency | Throughput |
| ------------------------- | ------- | ---------- |
| meta-llama/Llama-2-7b-hf  | 8.83%   | 9.68%      |

#### DeepSpeed

[zero_stage_2.json](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/text-classification/zero_stage_2.json) is an example DeepSpeed config file to enable Stage-2 parameter sharing for training meta-llama/Llama-2-7b. More information can be found at [DeepSpeed's official repo](https://github.com/microsoft/DeepSpeed).

## GLUE Tasks

By running the script [`run_glue.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/text-classification/run_glue.py),
we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) accelerator to fine-tune the models from the
[HuggingFace hub](https://huggingface.co/models) for sequence classification on the [GLUE benchmark](https://gluebenchmark.com/).


__The following example applies the acceleration features powered by ONNX Runtime.__


### ONNX Runtime Training

The following example fine-tunes a BERT on the sst-2 task.

```bash
torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name sst2 \
    --do_train \
    --do_eval \
    --output_dir /tmp/ort-bert-sst2/
```

### Performance

We get the following results for [roberta-base](https://huggingface.co/roberta-base) and [roberta-large](https://huggingface.co/roberta-large)
mixed precision training(fp16) on sst2 dataset under PyTorch and ONNX Runtime backends. A single Nvidia A100 card was used to run the
experiment for 3 epochs::

| Model           | Backend      | Runtime(s) | Train samples(/s) |
| --------------- |------------- | ---------- | ----------------- |
| roberta-base    | PyTorch      | 752.3      | 268.6             |
| roberta-base    | ONNX Runtime | 729.7      | 276.9             |
| roberta-large   | PyTorch      | 3523.7     | 57.3              |
| roberta-large   | ONNX Runtime | 2986.6     | 67.7              |

We observe the gain of ONNX Runtime compared to PyTorch as follow:

| Model         | Latency | Throughput |
| ------------- | ------- | ---------- |
| roberta-base  | 2.99%   | 3.08%      |
| roberta-large | 15.24%  | 17.98%     |


__Note__
> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions*
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/torch_ort/docker/README.md).*

> *The inference will use PyTorch by default, if you want to use ONNX Runtime backend instead, add the flag `--inference_with_ort`.*
---
