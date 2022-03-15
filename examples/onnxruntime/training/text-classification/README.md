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

## GLUE Tasks

By running the script [`run_glue.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/text-classification/run_glue.py),
we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) accelerator to fine-tune the models from the 
[HuggingFace hub](https://huggingface.co/models) for sequence classification on the [GLUE benchmark](https://gluebenchmark.com/).

 
__The following example applies the acceleration features powered by ONNX Runtime.__


### ONNX Runtime Training

The following example fine-tunes a BERT on the sst-2 task.

```bash
python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name sst2 \
    --do_train \
    --do_eval \
    --output_dir /tmp/ort-bert-sst2/
```

__Note__
> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions* 
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/docs/install.md).*

---

## Benchmark

GLUE has 9 different tasks: cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli. Here are the results and performances 
of fine-tuning with `bert-base-uncased` by ONNX Runtime compared with PyTorch. (Work in progress...)
