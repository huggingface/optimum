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
--->

<h1 align="center"><p>🤗 Optimum</p></h1>

<p align="center">
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/optimum"/></a>
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/optimum"/></a>
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/optimum"/></a>
<a href="https://pypi.org/project/optimum/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/optimum"/></a>
<a href="https://huggingface.co/docs/optimum/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/optimum/index.svg?down_color=red&down_message=offline&up_message=online"/></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
Optimum 是针对 Transformers 🤖、Diffusers 🧨、TIMM 🖼️ 和 Sentence-Transformers 🤗 的拓展工具库，提供了一系列优化工具，使模型能够在目标硬件上以最高效率进行训练和运行，同时保持极其简便易用。
</p>

## 安装指南

可以使用 `pip` 安装 Optimum，命令如下：

```bash
python -m pip install optimum
```

如果您希望使用特定硬件加速器专属的 Optimum 功能，可查阅官方文档并参考下表安装对应依赖：

| 加速器 (Accelerator)                                                                 | 安装命令                                                                    |
| :---------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| [ONNX](https://huggingface.co/docs/optimum-onnx/en/index)                           | `pip install --upgrade --upgrade-strategy eager optimum[onnx]`              |
| [ONNX Runtime](https://huggingface.co/docs/optimum-onnx/onnxruntime/overview)       | `pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]`       |
| [ONNX Runtime GPU](https://huggingface.co/docs/optimum-onnx/onnxruntime/overview)   | `pip install --upgrade --upgrade-strategy eager optimum[onnxruntime-gpu]`   |
| [OpenVINO](https://huggingface.co/docs/optimum/intel/index)                         | `pip install --upgrade --upgrade-strategy eager optimum[openvino]`          |
| [NVIDIA TensorRT-LLM](https://huggingface.co/docs/optimum/main/en/nvidia_overview)  | `docker run -it --gpus all --ipc host huggingface/optimum-nvidia`           |
| [AMD Instinct GPU 与 Ryzen AI NPU](https://huggingface.co/docs/optimum/amd/index)    | `pip install --upgrade --upgrade-strategy eager optimum[amd]`               |
| [AWS Trainium 与 Inferentia](https://huggingface.co/docs/optimum-neuron/index)      | `pip install --upgrade --upgrade-strategy eager optimum[neuronx]`           |
| [Intel Gaudi 加速器 (HPU)](https://huggingface.co/docs/optimum/habana/index)         | `pip install --upgrade --upgrade-strategy eager optimum[habana]`            |
| [FuriosaAI](https://huggingface.co/docs/optimum/furiosa/index)                      | `pip install --upgrade --upgrade-strategy eager optimum[furiosa]`           |

`--upgrade --upgrade-strategy eager` 选项用于确保各个依赖包均升级到最新兼容版本。

从源码安装：

```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```

如需安装特定加速器功能，在上述命令后添加 `optimum[accelerator_type]`：

```bash
python -m pip install optimum[onnxruntime]@git+https://github.com/huggingface/optimum.git
```

## 加速推理 (Accelerated Inference)

Optimum 提供了多种工具，支持将优化后的模型导出并在多样化的生态系统中高效运行：

- [ONNX](https://huggingface.co/docs/optimum-onnx/en/onnx/usage_guides/export_a_model) / [ONNX Runtime](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/models)：最广泛使用的开源模型导出格式之一，以及面向生产环境部署的高性能推理引擎。
- [OpenVINO](https://huggingface.co/docs/optimum/intel/inference)：专为在 Intel 硬件上优化、量化和部署深度学习模型设计的工具包。
- [ExecuTorch](https://huggingface.co/docs/optimum-executorch/guides/export)：PyTorch 原生面向移动端和边缘设备的端侧推理解决方案。
- [Intel Gaudi 加速器](https://huggingface.co/docs/optimum/main/en/habana/usage_guides/accelerate_inference)：为第一代 Gaudi、Gaudi2 和 Gaudi3 提供极致推理性能。
- [AWS Inferentia](https://huggingface.co/docs/optimum-neuron/en/guides/models)：面向 Inf2 和 Inf1 实例的高性能推理加速。
- [NVIDIA TensorRT-LLM](https://huggingface.co/blog/optimum-nvidia)：面向 NVIDIA GPU 的前沿大语言模型加速。

模型[导出](https://huggingface.co/docs/optimum/exporters/overview)与优化既可通过 Python API 编程完成，也可以直接通过命令行执行。

### ONNX + ONNX Runtime

🚨🚨🚨 ONNX 集成已迁移至 [`optimum-onnx`](https://github.com/huggingface/optimum-onnx)，请务必遵循对应的安装指引 🚨🚨🚨

在开始之前，请确保已安装所需依赖：

```bash
pip install --upgrade --upgrade-strategy eager optimum[onnx]
```

支持将 Transformers、Diffusers、Sentence Transformers 和 Timm 模型导出为 [ONNX](https://onnx.ai/) 格式，并轻松完成计算图优化（Graph Optimization）与模型量化（Quantization）。

关于 ONNX 导出的更多信息，请查阅[官方文档](https://huggingface.co/docs/optimum-onnx/en/onnx/usage_guides/export_a_model)。

模型导出为 ONNX 格式后，Optimum 提供了 Python 类，使您能够以底层 [ONNX Runtime](https://onnxruntime.ai/) 为后端无缝运行导出的 ONNX 模型。

请确保已安装 ONNX Runtime，详情可查阅[安装指南](https://onnxruntime.ai/docs/install/)。

有关如何使用 `ORTModelForXXX` 类运行 ONNX 模型的更多细节，请参见[此页面](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/models)。

### Intel (OpenVINO + NNCF)

在开始之前，请确保已[安装所有必需依赖库](https://huggingface.co/docs/optimum/main/en/intel/installation)。

```bash
pip install --upgrade --upgrade-strategy eager optimum[openvino]
```

您可以在我们的[官方文档](https://huggingface.co/docs/optimum/main/en/intel/index)以及 [`optimum-intel`](https://github.com/huggingface/optimum-intel) 示例中了解有关不同集成的更多详细信息。

### ExecuTorch

在开始之前，请确保已安装所有必需依赖：

```bash
pip install optimum-executorch@git+https://github.com/huggingface/optimum-executorch.git
```

用户可以将 Transformers 模型导出至 [ExecuTorch](https://github.com/pytorch/executorch)，并在 PyTorch 生态体系内的各类边缘设备上运行推理。

有关将 Transformers 导出到 ExecuTorch 的更多信息，请参阅 [Optimum-ExecuTorch 文档](https://huggingface.co/docs/optimum-executorch/guides/export)。

### Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) 是一个 PyTorch 量化后端，支持使用 Python API 或 `optimum-cli` 命令行对模型进行量化。

您可以在 [Quanto 代码库](https://github.com/huggingface/optimum-quanto)中查看更多详细信息和[使用示例](https://github.com/huggingface/optimum-quanto/tree/main/examples)。

## 加速训练 (Accelerated training)

Optimum 针对原始 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) 提供了封装适配器，以轻松实现在高性能硬件上的训练。
支持多家主流硬件平台：

- [Intel Gaudi 加速器 (HPU)](https://huggingface.co/docs/optimum/main/en/habana/usage_guides/accelerate_training)：为第一代 Gaudi、Gaudi2 和 Gaudi3 提供最佳训练效能。
- [AWS Trainium](https://huggingface.co/docs/optimum-neuron/training_tutorials/sft_lora_finetune_llm)：在 Trn1 和 Trn1n 实例上实现高速训练。
- ONNX Runtime（针对 GPU 优化）。

### Intel Gaudi 加速器

在开始之前，请确保已安装所有必需依赖库：

```bash
pip install --upgrade --upgrade-strategy eager optimum[habana]
```

您可以在[官方文档](https://huggingface.co/docs/optimum/habana/quickstart)以及[示例目录](https://github.com/huggingface/optimum-habana/tree/main/examples)中找到相关示例。

### AWS Trainium

在开始之前，请确保已安装所有必需依赖库：

```bash
pip install --upgrade --upgrade-strategy eager optimum[neuronx]
```

您可以在[官方文档](https://huggingface.co/docs/optimum-neuron/index)以及[实战教程](https://huggingface.co/docs/optimum-neuron/tutorials/fine_tune_bert)中找到相关示例。

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月13日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
