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

# Summarization

By running the script [`run_summarization.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/summarization/run_summarization.py),
you will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) to fine-tune and evaluate models from the
[HuggingFace hub](https://huggingface.co/models) on summarization tasks.

### Supported models

Theorectically, all sequence-to-sequence models with [ONNXConfig](https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py) support in Transformers shall work, here are models that Optimum team has tested and validated.

* `Bart`
* `T5`

`run_summarization.py` is a lightweight example of how to download and preprocess a dataset from the 🤗 Datasets library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.


__The following example applies the acceleration features powered by ONNX Runtime.__


### Onnxruntime Training

The following example fine-tunes a BERT on the SQuAD 1.0 dataset.

```bash
python run_summarization.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir /tmp/ort_summarization_t5/ \
    --overwrite_output_dir \
    --predict_with_generate
```

__Note__
> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions*
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/torch_ort/docker/README.md).*

> *The inference will use PyTorch by default, if you want to use ONNX Runtime backend instead, add the flag `--inference_with_ort`.*
---