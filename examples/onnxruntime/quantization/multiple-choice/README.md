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

# Multiple choice

The script [`run_swag.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/quantization/multiple-choice/run_swag.py) allows us to apply different quantization approaches (such as dynamic and static quantization) using the [ONNX Runtime](https://github.com/microsoft/onnxruntime) quantization tool for multiple choice tasks.

The following example applies post-training dynamic quantization on a BERT fine-tuned on the SWAG dataset.

```bash
python run_swag.py \
    --model_name_or_path ehdwns1516/bert-base-uncased_SWAG \
    --quantization_approach dynamic \
    --do_eval \
    --output_dir /tmp/quantized_bert_swag
```

In order to apply dynamic or static quantization, `quantization_approach` must be set to  respectively `dynamic` or `static`.
