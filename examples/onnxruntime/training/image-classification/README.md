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

# Image Classification 

By running the scripts [`run_image_classification.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/image-classification/run_image_classification.py) we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) accelerator to train the language models from the
[HuggingFace hub](https://huggingface.co/models).


__The following example applies the acceleration features powered by ONNX Runtime.__


### ONNX Runtime Training

The following example trains ViT on beans dataset with mixed precision (fp16).

```bash
torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name beans \
    --output_dir ./beans_outputs/ \
    --remove_unused_columns False \
    --label_column_name labels \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --seed 1337
```


__Note__
> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions*
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/torch_ort/docker/README.md).*
---
