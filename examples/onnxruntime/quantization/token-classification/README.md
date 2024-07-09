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

# Token classification

The script [`run_ner.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/quantization/token-classification/run_ner.py) allows us to apply different quantization approaches (such as dynamic and static quantization) as well as graph optimizations using [ONNX Runtime](https://github.com/microsoft/onnxruntime) for token classification tasks.

The following example applies post-training dynamic quantization on a DistilBERT fine-tuned on the CoNLL-2003 task

```bash
python run_ner.py \
    --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english \
    --dataset_name conll2003 \
    --quantization_approach dynamic \
    --do_eval \
    --output_dir /tmp/quantized_distilbert_conll2003
```

In order to apply dynamic or static quantization, `quantization_approach` must be set to  respectively `dynamic` or `static`.
