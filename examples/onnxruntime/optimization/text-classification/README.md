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

# Text classification

## GLUE tasks

The script [`run_glue.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/optimization/text-classification/run_glue.py) allows us to apply graph optimizations and fusion using [ONNX Runtime](https://github.com/microsoft/onnxruntime) for sequence classification tasks such as the ones from the [GLUE benchmark](https://gluebenchmark.com/).

The following example applies graph optimization on a DistilBERT fine-tuned on the sst-2 task. Here the optimization level is selected to be 1, enabling basic optimizations such as redundant node eliminations and constant folding. Higher optimization level will result in hardware dependent optimized graph.

```bash
python run_glue.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --optimization_level 1 \
    --do_eval \
    --output_dir /tmp/optimized_distilbert_sst2
```
