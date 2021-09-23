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

Apply Neural Compressor (LPOT) dynamic quantization:


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


Export a model to an ONNX Intermediate Representation (IR):

```bash
optimum_export \
    --model textattack/bert-base-uncased-SST-2 \
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
    --model textattack/bert-base-uncased-SST-2 \
    --output /tmp/onnx_models/model.onnx
    --opt_level 1 \
    --quantize \
    --atol 1.5 
```


