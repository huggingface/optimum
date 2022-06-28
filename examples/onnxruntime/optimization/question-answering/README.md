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

# Question answering

The script [`run_qa.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/optimization/question-answering/run_qa.py)
allows us to apply graph optimizations using [ONNX Runtime](https://github.com/microsoft/onnxruntime) for question answering tasks.

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along 
the flag `--version_2_with_negative`.

The following example applies graph optimizations on a DistilBERT fine-tuned on the SQuAD1.0 dataset. Here the optimization level is selected to be 1, enabling basic optimizations such as redundant node eliminations and constant folding. Higher optimization level will result in hardware dependent optimized graph.

```bash
python run_qa.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --optimization_level 1 \
    --do_eval \
    --output_dir /tmp/optimized_distilbert_squad
```

In order to apply dynamic or static quantization, `quantization_approach` must be set to  respectively `dynamic` or `static`.
