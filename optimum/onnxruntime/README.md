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

# ONNX Runtime (ORT) 

## Usage examples

Export a model to an ONNX Intermediate Representation (IR):

```bash
optimum_export \
    --model_name_or_path bert-base-uncased \
    --output /tmp/onnx_models/model.onnx
```

Optimize a model and apply dynamic quantization using ONNX Runtime:

```bash
optimum_optimize \
    --onnx_model_path /tmp/onnx_models/model.onnx \
    --opt_level 1 \
    --quantize 
```

The two steps mentioned above can be performed in one step using the following command line:

```bash
optimum_export_optimize \
    --model_name_or_path bert-base-uncased \
    --opt_level 1 \
    --quantize \
    --atol 1.5 \
    --output /tmp/onnx_models/model.onnx
```

