<!---
Copyright 2025 The HuggingFace Team. All rights reserved.

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

<h1 align="center"><p>🤗 Optimum</p></h1>

<p align="center">
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/optimum"/></a>
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/optimum"/></a>
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/optimum"/></a>
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/optimum"/></a>
<a href="https://huggingface.co/docs/optimum/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/optimum/index.svg?down_color=red&down_message=offline&up_message=online"/></a>
</p>

<p align="center">
Optimum is an extension of Transformers 🤖 Diffusers 🧨 TIMM 🖼️ and Sentence-Transformers 🤗, providing a set of optimization tools and enabling maximum efficiency to train and run models on targeted hardware, while keeping things easy to use.
</p>

## Installation

Optimum can be installed using `pip` as follows:

```bash
python -m pip install optimum
```

If you'd like to use the accelerator-specific features of Optimum, you can check the documentation and install the required dependencies according to the table below:

| Accelerator                                                                         | Installation                                                                |
| :---------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| [ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/overview)            | `pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]`       |
| [Intel Neural Compressor](https://huggingface.co/docs/optimum/intel/index)          | `pip install --upgrade --upgrade-strategy eager optimum[neural-compressor]` |
| [OpenVINO](https://huggingface.co/docs/optimum/intel/index)                         | `pip install --upgrade --upgrade-strategy eager optimum[openvino]`          |
| [IPEX](https://huggingface.co/docs/optimum/intel/ipex/inference)                    | `pip install --upgrade --upgrade-strategy eager optimum[ipex]`              |
| [NVIDIA TensorRT-LLM](https://huggingface.co/docs/optimum/main/en/nvidia_overview)  | `docker run -it --gpus all --ipc host huggingface/optimum-nvidia`           |
| [AMD Instinct GPUs and Ryzen AI NPU](https://huggingface.co/docs/optimum/amd/index) | `pip install --upgrade --upgrade-strategy eager optimum[amd]`               |
| [AWS Trainum & Inferentia](https://huggingface.co/docs/optimum-neuron/index)        | `pip install --upgrade --upgrade-strategy eager optimum[neuronx]`           |
| [Intel Gaudi Accelerators (HPU)](https://huggingface.co/docs/optimum/habana/index)  | `pip install --upgrade --upgrade-strategy eager optimum[habana]`            |
| [FuriosaAI](https://huggingface.co/docs/optimum/furiosa/index)                      | `pip install --upgrade --upgrade-strategy eager optimum[furiosa]`           |

The `--upgrade --upgrade-strategy eager` option is needed to ensure the different packages are upgraded to the latest possible version.

To install from source:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```

For the accelerator-specific features, append `optimum[accelerator_type]` to the above command:

```bash
python -m pip install optimum[onnxruntime]@git+https://github.com/huggingface/optimum.git
```

## Accelerated Inference

Optimum provides multiple tools to export and run optimized models on various ecosystems:

- [ONNX](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model) / [ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models), one of the most popular open formats for model export, and a high-performance inference engine for deployment.
- [OpenVINO](https://huggingface.co/docs/optimum/intel/inference), a toolkit for optimizing, quantizing and deploying deep learning models on Intel hardware.
- [ExecuTorch](https://huggingface.co/docs/optimum-executorch/guides/export), PyTorch’s native solution for on-device inference across mobile and edge devices.
- [TensorFlow Lite](https://huggingface.co/docs/optimum/exporters/tflite/usage_guides/export_a_model), a lightweight solution for running TensorFlow models on mobile and edge.
- [Intel Gaudi Accelerators](https://huggingface.co/docs/optimum/main/en/habana/usage_guides/accelerate_inference) enabling optimal performance on first-gen Gaudi, Gaudi2 and Gaudi3.
- [AWS Inferentia](https://huggingface.co/docs/optimum-neuron/en/guides/models) for accelerated inference on Inf2 and Inf1 instances.
- [NVIDIA TensorRT-LLM](https://huggingface.co/blog/optimum-nvidia).

The [export](https://huggingface.co/docs/optimum/exporters/overview) and optimizations can be done both programmatically and with a command line.

### ONNX + ONNX Runtime

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[exporters,onnxruntime]
```

It is possible to export Transformers and Diffusers models to the [ONNX](https://onnx.ai/) format and perform graph optimization as well as quantization easily.

For more information on the ONNX export, please check the [documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model).

Once the model is exported to the ONNX format, we provide Python classes enabling you to run the exported ONNX model in a seemless manner using [ONNX Runtime](https://onnxruntime.ai/) in the backend.

More details on how to run ONNX models with `ORTModelForXXX` classes [here](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models).

### Intel (OpenVINO + Neural Compressor + IPEX)

Before you begin, make sure you have all the necessary [libraries installed](https://huggingface.co/docs/optimum/main/en/intel/installation).

You can find more information on the different integration in our [documentation](https://huggingface.co/docs/optimum/main/en/intel/index) and in the examples of [`optimum-intel`](https://github.com/huggingface/optimum-intel).

### ExecuTorch

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum-executorch@git+https://github.com/huggingface/optimum-executorch.git
```

Users can export Transformers models to [ExecuTorch](https://github.com/pytorch/executorch) and run inference on edge devices within PyTorch's ecosystem.

For more information about export Transformers to ExecuTorch, please check the doc for [Optimum-ExecuTorch](https://huggingface.co/docs/optimum-executorch/guides/export).

### TensorFlow Lite

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[exporters-tf]
```

Just as for ONNX, it is possible to export models to [TensorFlow Lite](https://www.tensorflow.org/lite) and quantize them.
You can find more information in our [documentation](https://huggingface.co/docs/optimum/main/exporters/tflite/usage_guides/export_a_model).

### Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a pytorch quantization backend which allows you to quantize a model either using the python API or the `optimum-cli`.

You can see more details and [examples](https://github.com/huggingface/optimum-quanto/tree/main/examples) in the [Quanto](https://github.com/huggingface/optimum-quanto) repository.

## Accelerated training

Optimum provides wrappers around the original Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to enable training on powerful hardware easily.
We support many providers:

- [Intel Gaudi Accelerators (HPU)](https://huggingface.co/docs/optimum/main/en/habana/usage_guides/accelerate_training) enabling optimal performance on first-gen Gaudi, Gaudi2 and Gaudi3.
- [AWS Trainium](https://huggingface.co/docs/optimum-neuron/training_tutorials/sft_lora_finetune_llm) for accelerated training on Trn1 and Trn1n instances.
- ONNX Runtime (optimized for GPUs).

### Intel Gaudi Accelerators

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install --upgrade --upgrade-strategy eager optimum[habana]
```

You can find examples in the [documentation](https://huggingface.co/docs/optimum/habana/quickstart) and in the [examples](https://github.com/huggingface/optimum-habana/tree/main/examples).

### AWS Trainium

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install --upgrade --upgrade-strategy eager optimum[neuronx]
```

You can find examples in the [documentation](https://huggingface.co/docs/optimum-neuron/index) and in the [tutorials](https://huggingface.co/docs/optimum-neuron/tutorials/fine_tune_bert).

### ONNX Runtime

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[onnxruntime-training]
```

You can find examples in the [documentation](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/trainer) and in the [examples](https://github.com/huggingface/optimum/tree/main/examples/onnxruntime/training).
