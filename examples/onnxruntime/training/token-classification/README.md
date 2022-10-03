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

# Token classification

## NER Tasks

By running the script [`run_ner.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/token-classification/run_ner.py),
we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) accelerator to fine-tune the models from the
[HuggingFace hub](https://huggingface.co/models) for token classification tasks such as Named Entity Recognition (NER).


__The following example applies the acceleration features powered by ONNX Runtime.__


### ONNX Runtime Training

The following example fine-tunes a BERT on the sst-2 task.

```bash
python run_ner.py \
    --model_name_or_path bert-base-cased \
    --dataset_name conll2003 \
    --do_train \
    --do_eval \
    --output_dir /tmp/ort_bert_conll2003/
```

### Performance

We get the following results for [bert-large-cased](https://huggingface.co/bert-large-cased) model mixed precision training(fp16) on the previous
task under PyTorch and ONNX Runtime backends. A single Nvidia A100 card was used to run the experiment for 7 epochs:

| Model            | Backend      | Runtime(s) | Train samples(/s) |
| ---------------- | ------------ | ---------- | ----------------- |
| bert-large-cased | PyTorch      | 711.5      | 138.1             |
| bert-large-cased | ONNX Runtime | 637.2      | 154.3             |

We observe the gain of ONNX Runtime compared to PyTorch as follow:

|       | Latency | Throughput |
| ----- | ------- | ---------- |
| Gain  | 10.45%  | 11.67%     |


__Note__
> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions*
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/torch_ort/docker/README.md).*

> *The inference will use PyTorch by default, if you want to use ONNX Runtime backend instead, add the flag `--inference_with_ort`.*
---
