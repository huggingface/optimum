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

# 🤗 Optimum Notebooks
🤗  [Optimum](https://github.com/huggingface/optimum) is an extension of 🤗 [Transformers](https://github.com/huggingface/transformers), providing a set of performance optimization tools enabling maximum efficiency to train and run models on targeted hardwares.

You can find here a list of the official notebooks, leveraging the usage of the Optimum library for different providers.

## Hugging Face's notebooks 🤗

### PyTorch Examples

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [How to quantize a model with ONNX Runtime for text classification](https://github.com/huggingface/notebooks/blob/master/examples/text_classification_quantization_ort.ipynb)| Show how to apply static and dynamic quantization on a model using [ONNX Runtime](https://github.com/microsoft/onnxruntime) for any GLUE task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_quantization_ort.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/examples/text_classification_quantization_ort.ipynb)|
| [How to quantize a model with Intel Neural Compressor for text classification](https://github.com/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb)| Show how to apply static, dynamic and aware training quantization on a model using [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) for any GLUE task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb)|
| [How to fine-tune a model on text classification with ONNX Runtime](https://github.com/huggingface/notebooks/blob/main/examples/text_classification_ort.ipynb)| Show how to preprocess the data and fine-tune a model on any GLUE task using [ONNX Runtime](https://github.com/microsoft/onnxruntime). | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/huggingface/notebooks/blob/main/examples/text_classification_ort.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github.com/huggingface/notebooks/blob/main/examples/text_classification_ort.ipynb)|
| [How to fine-tune a model on summarization with ONNX Runtime](https://github.com/huggingface/notebooks/blob/main/examples/summarization_ort.ipynb)| Show how to preprocess the data and fine-tune a model on XSUM using [ONNX Runtime](https://github.com/microsoft/onnxruntime). | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/huggingface/notebooks/blob/main/examples/summarization_ort.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github.com/huggingface/notebooks/blob/main/examples/summarization_ort.ipynb)|