<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

# Image image_classification

The script [`run_image_classification.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/quantization/image_classification/run_image_classification.py) allows us to apply different quantization approaches (such as dynamic and static quantization) as well as graph optimizations using [ONNX Runtime](https://github.com/microsoft/onnxruntime) for image classification tasks.

The following example applies dynamic quantization on a ViT model fine-tuned on the beans classification dataset.

```bash
python run_image_classification.py \
    --model_name_or_path nateraw/vit-base-beans \
    --dataset_name beans \
    --quantization_approach dynamic \
    --do_eval \
    --output_dir /tmp/image_classification_vit_beans
```

In order to apply dynamic or static quantization, `quantization_approach` must be set to  respectively `dynamic` or `static`.
