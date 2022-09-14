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

# Language Modeling

## Language Modeling Training

By running the scripts [`run_clm.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/language-modeling/run_clm.py)
and [`run_mlm.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/language-modeling/run_mlm.py),
we will be able to leverage the [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) accelerator to train the language models from the
[HuggingFace hub](https://huggingface.co/models).


__The following example applies the acceleration features powered by ONNX Runtime.__


### ONNX Runtime Training

The following example trains GPT2 on wikitext-2 with mixed precision (fp16).

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --output_dir /tmp/test-clm \
    --fp16
```


__Note__
> *To enable ONNX Runtime training, your devices need to be equipped with GPU. Install the dependencies either with our prepared*
*[Dockerfiles](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/) or follow the instructions*
*in [`torch_ort`](https://github.com/pytorch/ort/blob/main/torch_ort/docker/README.md).*

> *The inference will use PyTorch by default, if you want to use ONNX Runtime backend instead, add the flag `--inference_with_ort`.*
---
