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

The script [`run_swag.py`](https://github.com/huggingface/optimum/blob/main/examples/pytorch/multiple-choice/run_swag.py) 
allows us to apply different quantization approaches (such as dynamic, static and aware-training quantization) as well as pruning 
using the [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) library for 
multiple choice tasks.

The following example applies post-training static quantization on a BERT fine-tuned on the SWAG datasets.

```bash
python run_swag.py \
    --model_name_or_path ehdwns1516/bert-base-uncased_SWAG \
    --quantize \
    --quantization_approach static \
    --do_train \
    --do_eval \
    --pad_to_max_length \
    --dataloader_drop_last \
    --verify_loading \
    --output_dir /tmp/swag_output
```


The following example fine-tunes BERT on the SWAG dataset while applying magnitude pruning and then applies 
dynamic quantization as a second step.

```bash
python run_swag.py \
    --model_name_or_path ehdwns1516/bert-base-uncased_SWAG \
    --quantize \
    --quantization_approach dynamic \
    --prune \
    --target_sparsity 0.1 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/swag_output
```

In order to apply dynamic, static or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static` or `aware_training`.

The configuration file containing all the information related to the model quantization and pruning objectives can be 
specified using respectively `quantization_config` and `pruning_config`. If not specified, the default
[quantization](https://github.com/huggingface/optimum/blob/main/examples/inc/pytorch/config/quantization.yml) 
and [pruning](https://github.com/huggingface/optimum/blob/main/examples/inc/pytorch/config/prune.yml) 
config files will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.