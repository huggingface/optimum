[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml)

# Hugging Face Optimum

🤗 Optimum is an extension of 🤗 Transformers and Diffusers, providing a set of optimization tools enabling maximum efficiency to train and run models on targeted hardware, while keeping things easy to use.

## Installation

🤗 Optimum can be installed using `pip` as follows:

```bash
python -m pip install optimum
```

If you'd like to use the accelerator-specific features of 🤗 Optimum, you can install the required dependencies according to the table below:

| Accelerator                                                                                                            | Installation                                                      |
|:-----------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|
| [ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/overview)                                               | `pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]`      |
| [Intel Neural Compressor](https://huggingface.co/docs/optimum/intel/index)                                             | `pip install --upgrade --upgrade-strategy eager optimum[neural-compressor]`|
| [OpenVINO](https://huggingface.co/docs/optimum/intel/index)                                                            | `pip install --upgrade --upgrade-strategy eager optimum[openvino]`         |
| [NVIDIA TensorRT-LLM](https://huggingface.co/docs/optimum/main/en/nvidia_overview)                                     | `docker run -it --gpus all --ipc host huggingface/optimum-nvidia`          |
| [AMD Instinct GPUs and Ryzen AI NPU](https://huggingface.co/docs/optimum/amd/index)                                    | `pip install --upgrade --upgrade-strategy eager optimum[amd]`              |
| [AWS Trainum & Inferentia](https://huggingface.co/docs/optimum-neuron/index)                                           | `pip install --upgrade --upgrade-strategy eager optimum[neuronx]`          |
| [Habana Gaudi Processor (HPU)](https://huggingface.co/docs/optimum/habana/index)                                       | `pip install --upgrade --upgrade-strategy eager optimum[habana]`           |
| [FuriosaAI](https://huggingface.co/docs/optimum/furiosa/index)                                                         | `pip install --upgrade --upgrade-strategy eager optimum[furiosa]`          |

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

🤗 Optimum provides multiple tools to export and run optimized models on various ecosystems:

- [ONNX](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model) / [ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models)
- TensorFlow Lite
- [OpenVINO](https://huggingface.co/docs/optimum/intel/inference)
- Habana first-gen Gaudi / Gaudi2, more details [here](https://huggingface.co/docs/optimum/main/en/habana/usage_guides/accelerate_inference)
- AWS Inferentia 2 / Inferentia 1, more details [here](https://huggingface.co/docs/optimum-neuron/en/guides/models)
- NVIDIA TensorRT-LLM , more details [here](https://huggingface.co/blog/optimum-nvidia)

The [export](https://huggingface.co/docs/optimum/exporters/overview) and optimizations can be done both programmatically and with a command line.


### ONNX + ONNX Runtime

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[exporters,onnxruntime]
```

It is possible to export 🤗 Transformers and Diffusers models to the [ONNX](https://onnx.ai/) format and perform graph optimization as well as quantization easily.

For more information on the ONNX export, please check the [documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model).

Once the model is exported to the ONNX format, we provide Python classes enabling you to run the exported ONNX model in a seemless manner using [ONNX Runtime](https://onnxruntime.ai/) in the backend.

More details on how to run ONNX models with `ORTModelForXXX` classes [here](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models).

### TensorFlow Lite

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[exporters-tf]
```

Just as for ONNX, it is possible to export models to [TensorFlow Lite](https://www.tensorflow.org/lite) and quantize them.
You can find more information in our [documentation](https://huggingface.co/docs/optimum/main/exporters/tflite/usage_guides/export_a_model).

### Intel (OpenVINO + Neural Compressor + IPEX)

Before you begin, make sure you have all the necessary [libraries installed](https://huggingface.co/docs/optimum/main/en/intel/installation).

You can find more information on the different integration in our [documentation](https://huggingface.co/docs/optimum/main/en/intel/index) and in the examples of [`optimum-intel`](https://github.com/huggingface/optimum-intel).


### Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a pytorch quantization backenb which allowss you to quantize a model either using the python API or the `optimum-cli`.

You can see more details and [examples](https://github.com/huggingface/optimum-quanto/tree/main/examples) in the [Quanto](https://github.com/huggingface/optimum-quanto) repository.

## Accelerated training

🤗 Optimum provides wrappers around the original 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to enable training on powerful hardware easily.
We support many providers:

- Habana's Gaudi processors
- AWS Trainium instances, check [here](https://huggingface.co/docs/optimum-neuron/en/guides/distributed_training)
- ONNX Runtime (optimized for GPUs)

### Habana

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install --upgrade --upgrade-strategy eager optimum[habana]
```

You can find examples in the [documentation](https://huggingface.co/docs/optimum/habana/quickstart) and in the [examples](https://github.com/huggingface/optimum-habana/tree/main/examples).

### ONNX Runtime


Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[onnxruntime-training]
```

You can find examples in the [documentation](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/trainer) and in the [examples](https://github.com/huggingface/optimum/tree/main/examples/onnxruntime/training).
