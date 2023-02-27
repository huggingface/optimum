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

# Question answering

## SQuAD Tasks

By running the script [`run_qa.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/question-answering/run_qa.py),
we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) to fine-tune the models from the
[HuggingFace hub](https://huggingface.co/models) for question answering tasks such as SQuAD.

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along
the flag `--version_2_with_negative`.

__The following example applies the acceleration features powered by ONNX Runtime.__


### Onnxruntime Training

The following example fine-tunes a BERT on the SQuAD 1.0 dataset.

```bash
torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --output_dir /tmp/ort_bert_squad/
```

__Note__
> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions*
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/torch_ort/docker/README.md).*

> *The inference will use PyTorch by default, if you want to use ONNX Runtime backend instead, add the flag `--inference_with_ort`.*
---