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

# Language modeling training

The scripts [`run_clm.py`](https://github.com/huggingface/optimum/blob/main/examples/pytorch/language-modeling/run_clm.py) 
and [`run_mlm.py`](https://github.com/huggingface/optimum/blob/main/examples/pytorch/language-modeling/run_mlm.py)
allow us to apply different quantization approaches such as dynamic, static and aware-training quantization using 
different provider such as [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) for language 
modeling tasks.


GPT and GPT-2 are trained or fine-tuned using a causal language modeling (CLM) loss. ALBERT, BERT, DistilBERT and 
RoBERTa are trained or fine-tuned using a masked language modeling (MLM) loss, more information about the differences 
between those objectives can be found in our [model summary](https://huggingface.co/transformers/model_summary.html).


### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2 while applying [`inc`](https://github.com/intel/neural-compressor) 
quantization aware training. We're using the raw WikiText-2 (no tokens were replaced before the tokenization). The loss here is that of causal language modeling (CLM). 

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --provider inc \
    --quantize \
    --quantization_approach dynamic \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/clm_output
```

### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2 while applying [`inc`](https://github.com/intel/neural-compressor) 
quantization aware training. We're using the raw  WikiText-2. The loss is different as BERT/RoBERTa have a bidirectional 
mechanism, we are therefore using the same loss that was used during their pre-training: masked language modeling (MLM) loss. 

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --provider inc \
    --quantize \
    --quantization_approach dynamic \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/mlm_output
```

In order to apply dynamic, static or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static` or `aware_training`.

The configuration file can be specified by `config_name_or_path` and contains all the information related 
to the model quantization and tuning objective.  If no `config_name_or_path` is specified, the 
[default config file](https://github.com/huggingface/optimum/blob/main/examples/pytorch/language-modeling/config/inc/quantization.yml) 
will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
