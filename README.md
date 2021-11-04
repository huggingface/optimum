[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml)
[![neural_compressor](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml)

# Optimum

## Install
To install the latest release of this package:

`pip install optimum`

or from current main branch:

`pip install git+https://github.com/huggingface/optimum.git`

or for development, clone the repo and install it from the local copy:

```
git clone https://github.com/huggingface/optimum.git
cd optimum 
pip install -e .
```

To use Intel Neural Compressor (INC),

`pip install optimum[intel]`

To use ONNX Runtime,

`pip install optimum[onnxruntime]`

## Usage

Apply quantization from Intel Neural Compressor (INC):

```bash
python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english\
    --task_name sst2 \
    --provider inc \
    --quantize \
    --quantization_approach dynamic \
    --config_name_or_path echarlaix/bert-base-dynamic-quant-test \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/sst2_output/
```


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
    --output /tmp/onnx_models/model.onnx
    --opt_level 1 \
    --quantize \
    --atol 1.5 
```


