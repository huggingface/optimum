[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml)
[![neural_compressor](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml)

# Optimum

🤗  Optimum is an extension of 🤗 Transformers, providing a set of performance optimization tools enabling maximum efficiency to train and run models on targeted hardwares.
We currently support [ONNX runtime](https://github.com/microsoft/onnxruntime) dynamic quantization as well as [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) dynamic, post-training and aware training quantization on a variety of NLP tasks.

## Install

🤗 Optimum can be installed using pip as follows:

`pip install optimum`

🤗 Optimum with [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) or [ONNX runtime](https://github.com/microsoft/onnxruntime) dependencies can be installed respectively using pip as follows:

`pip install optimum[intel]`

`pip install optimum[onnxruntime]`

If you'd like to play with the examples or need the bleeding edge of the code and can't wait for a new release, you must install the library from source:

`pip install git+https://github.com/huggingface/optimum.git`
