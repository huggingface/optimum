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

# Translation

By running the script [`run_translation.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/translation/run_translation.py),
we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) to fine-tune the models from the
[HuggingFace hub](https://huggingface.co/models) for translation tasks.

### Supported Architectures

- `BartForConditionalGeneration`
- `T5ForConditionalGeneration`

`run_translation.py` is a lightweight examples of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library
or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets.html#json-files.

__The following example applies the acceleration features powered by ONNX Runtime.__


### Onnxruntime Training

The following example fine-tunes a T5 large model on the wmt16 dataset.

```bash
torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE run_translation.py \
    --model_name_or_path t5-large \
    --dataset_name wmt16 \
    --dataset_config ro-en \
    --label_smoothing 0.1 \
    --predict_with_generate \
    --source_lang en \
    --target_lang ro \
    --do_train \
    --max_train_samples 30000 \
    --fp16 \
    --output_dir /tmp/ort_t5_translation/
```

### Performance

We get the following results for [t5-large](https://huggingface.co/t5-large) mixed precision training(fp16) on the previous
task under PyTorch and ONNX Runtime backends. A single Nvidia A100 card was used to run the experiment for 3 epochs::

| Model    | Backend      | Runtime(s) | Train samples(/s) |
| -------- | ------------ | ---------- | ----------------- |
| t5-large | PyTorch      | 2038.8     | 44.1              |
| t5-large | ONNX Runtime | 1536.7     | 58.6              |

We observe the gain of ONNX Runtime compared to PyTorch as follow:

|       | Latency | Throughput |
| ----- | ------- | ---------- |
| Gain  | 24.63%  | 32.67%     |


__Note__

> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions*
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/torch_ort/docker/README.md).*

> *The inference will use PyTorch by default, if you want to use ONNX Runtime backend instead, add the flag `--inference_with_ort`.*
---