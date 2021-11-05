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


The script [`run_qa.py`](https://github.com/huggingface/optimum/blob/main/examples/pytorch/question-answering/run_qa.py)
allows us to apply different quantization approach such as dynamic, post-training and quantization aware-training 
using different provider such as [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) for
question answering tasks.

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along 
the flag `--version_2_with_negative`.

The following example applies dynamic quantization on a DistilBERT fine-tuned on the SQuAD1.0 dataset, using the
[`inc`](https://github.com/intel/neural-compressor) provider: 

```bash
python examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --provider inc \
    --quantize \
    --quantization_approach dynamic \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/squad_output/
```

In order to apply dynamic, post-training or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static` or `aware_training`.

The configuration file can be specified by `config_name_or_path` and contains all the information related 
to the model quantization and tuning objective.  If no `config_name_or_path` is specified, the 
[default config file](https://github.com/huggingface/optimum/blob/main/examples/pytorch/question-answering/config/inc/quantization.yml) 
will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
