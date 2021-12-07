[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-onnxruntime.yml)
[![neural_compressor](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test-intel.yml)

# Hugging Face - Optimum

🤗 Optimum is an extension of 🤗 Transformers, providing a set of performance optimization tools enabling maximum efficiency to train and run models on targeted hardware.

The AI ecosystem evolves quickly and more and more specialized hardware along with their own optimizations are emerging every day.
As such, Optimum enables users to efficiently use any of these platforms with the same ease inherent to transformers.


## Integration with Hardware Partners  

🤗 Optimum aims at providing more diversity towards the kind of hardware users can target to train and finetune their models.

To achieve this, we are collaborating with the following hardware manufacturers in order to provide the best transformers integration:
- [GraphCore IPUs](https://github.com/huggingface/optimum-graphcore) - IPUs are a completely new kind of massively parallel processor to accelerate machine intelligence. [More information here](https://www.graphcore.ai/products/ipu)
- More to come soon! :star:

## Optimizing models towards inference

Along with supporting dedicated AI hardware for training, Optimum also provides inference optimizations towards various frameworks and
platforms.


We currently support [ONNX runtime](https://github.com/microsoft/onnxruntime) along with [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor).

| Features                           | ONNX Runtime          | Intel Neural Compressor |
|:----------------------------------:|:---------------------:|:-----------------------:|
| Post-training Dynamic Quantization |  :heavy_check_mark:   |    :heavy_check_mark:   |  
| Post-training Static Quantization  |  Stay tuned! :star:   |    :heavy_check_mark:   |  
| Quantization Aware Training (QAT)  |        :x:            |    :heavy_check_mark:   |
| Pruning                            |        N/A            |    :heavy_check_mark:   |


## Install

🤗 Optimum can be installed using pip as follows:

`pip install optimum`

🤗 Optimum with [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) or [ONNX runtime](https://github.com/microsoft/onnxruntime) dependencies can be installed respectively using pip as follows:

`pip install optimum[intel]`

`pip install optimum[onnxruntime]`

If you'd like to play with the examples or need the bleeding edge of the code and can't wait for a new release, you must install the library from source:

`pip install git+https://github.com/huggingface/optimum.git`
