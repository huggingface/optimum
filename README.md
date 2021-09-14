[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml)
[![LPOT](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml)

# Optimum

## Install
To install the latest release of this package:

`pip install optimum`

or from current main branch:

`pip install https://github.com/huggingface/optimum.git`

or for development, clone the repo and install it from the local copy:

```
git clone https://github.com/huggingface/optimum.git
cd optimum 
pip install -e .
```


## Usage
Convert a hub model:
`optimum_convert` 

Optimize a hub model:
`optimum_optimize` 

Apply LPOT quantization:

```bash
python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path textattack/bert-base-uncased-SST-2 \
    --task_name sst2 \
    --provider lpot \
    --quantize \
    --quantization_approach dynamic \
    --config_name_or_path echarlaix/bert-base-dynamic-quant-test \
    --do_eval \
    --output_dir /tmp/sst2_output/
```
