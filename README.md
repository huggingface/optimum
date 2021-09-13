[![ONNX Runtime](https://github.com/huggingface/optimus/actions/workflows/test-onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimus/actions/workflows/test-onnxruntime.yml)
[![LPOT](https://github.com/huggingface/optimus/actions/workflows/test-intel.yml/badge.svg)](https://github.com/huggingface/optimus/actions/workflows/test-intel.yml)

# Optimus

## Install
To install the latest release of this package:

`pip install optimus`

or from current main branch:

`pip install https://github.com/huggingface/optimus.git`

or for development, clone the repo and install it from the local copy:

```
git clone https://github.com/huggingface/optimus.git
cd optimus 
pip install -e .
```


## Usage
Convert a hub model:
`optimus_convert` 

Optimize a hub model:
`optimus_optimize` 

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
