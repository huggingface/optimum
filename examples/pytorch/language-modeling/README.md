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

# Language modeling 

## language modeling

The script [`run_clm.py`](https://github.com/huggingface/optimum/blob/main/examples/pytorch/language-modeling/run_clm.py), [`run_mlm.py`](https://github.com/huggingface/optimum/blob/main/examples/pytorch/language-modeling/run_mlm.py).
allows us to apply different quantization approach such as dynamic, post-training and quantization aware-training 
using different provider such as [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) for 
language modeling tasks.


The following example applies dynamic quantization on a distilroberta fine-tuned on the wikitext datasets, using the
[`inc`](https://github.com/intel/neural-compressor) provider: 


```bash
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path bert-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --provider inc \
    --quantize \
    --quantization_approach dynamic \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/clm_output
```
```bash
python examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path google/electra-base-generator \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --provider inc \
    --quantize \
    --quantization_approach dynamic \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/mlm_output
```
In order to apply dynamic, post-training or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static` or `aware_training`.

For post-training static quantization and aware-training quantization, we must [set_config("model.framework", "pytorch_fx")](run_clm.py#L571)

The configuration file can be specified by `config_name_or_path` and contains all the information related 
to the model quantization and tuning objective.  If no `config_name_or_path` is specified, the 
[default config file](https://github.com/huggingface/optimum/blob/main/examples/pytorch/language-modeling/config/inc/quantization.yml) 
will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
