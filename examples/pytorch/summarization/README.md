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

# Summarization 

The script [`run_summarization.py`](https://github.com/huggingface/optimum/blob/main/examples/pytorch/summarization/run_summarization.py) 
allows us to apply different quantization approach such as dynamic, post-training and quantization aware-training 
using different provider such as [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) for 
language modeling tasks.


The following example applies dynamic quantization on a pegasus fine-tuned on the cnn_dailymail datasets, using the
[`inc`](https://github.com/intel/neural-compressor) provider: 


```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --provider inc \
    --quantize \
    --quantization_approach dynamic \
    --do_eval \
    --verify_loading
```

In order to apply dynamic, post-training or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static` or `aware_training`.

For post-training static quantization and aware-training quantization, we must [set_config("model.framework", "pytorch_fx")](run_summarization.py#L649)

The configuration file can be specified by `config_name_or_path` and contains all the information related 
to the model quantization and tuning objective.  If no `config_name_or_path` is specified, the 
[default config file](https://github.com/huggingface/optimum/blob/main/examples/pytorch/summarization/config/inc/quantization.yml) 
will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.