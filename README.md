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

### Features summary

| Features                           | [ONNX Runtime](https://huggingface.co/docs/optimum/main/en/onnxruntime/overview)| [Neural Compressor](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc)| [OpenVINO](https://huggingface.co/docs/optimum/main/en/intel/inference)| [TensorFlow Lite](https://huggingface.co/docs/optimum/main/en/exporters/tflite/overview)|
|:----------------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Graph optimization                 | :heavy_check_mark: | N/A                | :heavy_check_mark: | N/A                |
| Post-training dynamic quantization | :heavy_check_mark: | :heavy_check_mark: | N/A                | :heavy_check_mark: |
| Post-training static quantization  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Quantization Aware Training (QAT)  | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |
| FP16 (half precision)              | :heavy_check_mark: | N/A                | :heavy_check_mark: | :heavy_check_mark: |
| Pruning                            | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |
| Knowledge Distillation             | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |


### OpenVINO

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install --upgrade --upgrade-strategy eager optimum[openvino]
```

It is possible to export 🤗 Transformers and Diffusers models to the OpenVINO format easily:

```bash
optimum-cli export openvino --model distilbert-base-uncased-finetuned-sst-2-english distilbert_sst2_ov
```

If you add `--weight-format int8`, the weights will be quantized to `int8`, check out our [documentation](https://huggingface.co/docs/optimum/main/intel/openvino/export) for more detail. To apply quantization on both weights and activations, you can find more information [here](https://huggingface.co/docs/optimum/main/intel/openvino/optimization#static-quantization).

To load a model and run inference with OpenVINO Runtime, you can just replace your `AutoModelForXxx` class with the corresponding `OVModelForXxx` class. To load a PyTorch checkpoint and convert it to the OpenVINO format on-the-fly, you can set `export=True` when loading your model.

```diff
- from transformers import AutoModelForSequenceClassification
+ from optimum.intel import OVModelForSequenceClassification
  from transformers import AutoTokenizer, pipeline

  model_id = "distilbert-base-uncased-finetuned-sst-2-english"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
- model = AutoModelForSequenceClassification.from_pretrained(model_id)
+ model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)

  classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
  results = classifier("He's a dreadful magician.")
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/main/intel/openvino/inference) and in the [examples](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino).

### Neural Compressor

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install --upgrade --upgrade-strategy eager optimum[neural-compressor]
```

Dynamic quantization can be applied on your model:

```bash
optimum-cli inc quantize --model distilbert-base-cased-distilled-squad --output ./quantized_distilbert
```

To load a model quantized with Intel Neural Compressor, hosted locally or on the 🤗 hub, you can do as follows :
```python
from optimum.intel import INCModelForSequenceClassification

model_id = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model = INCModelForSequenceClassification.from_pretrained(model_id)
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/intel/optimization_inc) and in the [examples](https://github.com/huggingface/optimum-intel/tree/main/examples/neural_compressor).

### ONNX + ONNX Runtime

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[exporters,onnxruntime]
```

It is possible to export 🤗 Transformers and Diffusers models to the [ONNX](https://onnx.ai/) format and perform graph optimization as well as quantization easily:

```plain
optimum-cli export onnx -m deepset/roberta-base-squad2 --optimize O2 roberta_base_qa_onnx
```

The model can then be quantized using `onnxruntime`:

```bash
optimum-cli onnxruntime quantize \
  --avx512 \
  --onnx_model roberta_base_qa_onnx \
  -o quantized_roberta_base_qa_onnx
```

These commands will export `deepset/roberta-base-squad2` and perform [O2 graph optimization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization#optimization-configuration) on the exported model, and finally quantize it with the [avx512 configuration](https://huggingface.co/docs/optimum/main/en/onnxruntime/package_reference/configuration#optimum.onnxruntime.AutoQuantizationConfig.avx512).

For more information on the ONNX export, please check the [documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model).

#### Run the exported model using ONNX Runtime

Once the model is exported to the ONNX format, we provide Python classes enabling you to run the exported ONNX model in a seemless manner using [ONNX Runtime](https://onnxruntime.ai/) in the backend:

```diff
- from transformers import AutoModelForQuestionAnswering
+ from optimum.onnxruntime import ORTModelForQuestionAnswering
  from transformers import AutoTokenizer, pipeline

  model_id = "deepset/roberta-base-squad2"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
- model = AutoModelForQuestionAnswering.from_pretrained(model_id)
+ model = ORTModelForQuestionAnswering.from_pretrained("roberta_base_qa_onnx")
  qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
  question = "What's Optimum?"
  context = "Optimum is an awesome library everyone should use!"
  results = qa_pipe(question=question, context=context)
```

More details on how to run ONNX models with `ORTModelForXXX` classes [here](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models).

### TensorFlow Lite

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install optimum[exporters-tf]
```

Just as for ONNX, it is possible to export models to [TensorFlow Lite](https://www.tensorflow.org/lite) and quantize them:

```plain
optimum-cli export tflite \
  -m deepset/roberta-base-squad2 \
  --sequence_length 384  \
  --quantize int8-dynamic roberta_tflite_model
```

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

```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.habana import GaudiTrainer, GaudiTrainingArguments

  # Download a pretrained model from the Hub
  model = AutoModelForXxx.from_pretrained("bert-base-uncased")

  # Define the training arguments
- training_args = TrainingArguments(
+ training_args = GaudiTrainingArguments(
      output_dir="path/to/save/folder/",
+     use_habana=True,
+     use_lazy_mode=True,
+     gaudi_config_name="Habana/bert-base-uncased",
      ...
  )

  # Initialize the trainer
- trainer = Trainer(
+ trainer = GaudiTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      ...
  )

  # Use Habana Gaudi processor for training!
  trainer.train()
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/habana/quickstart) and in the [examples](https://github.com/huggingface/optimum-habana/tree/main/examples).

### ONNX Runtime

```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

  # Download a pretrained model from the Hub
  model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

  # Define the training arguments
- training_args = TrainingArguments(
+ training_args = ORTTrainingArguments(
      output_dir="path/to/save/folder/",
      optim="adamw_ort_fused",
      ...
  )

  # Create a ONNX Runtime Trainer
- trainer = Trainer(
+ trainer = ORTTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      ...
  )

  # Use ONNX Runtime for training!
  trainer.train()
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/trainer) and in the [examples](https://github.com/huggingface/optimum/tree/main/examples/onnxruntime/training).


### Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a pytorch quantization backend.

You can quantize a model either using the python API or the `optimum-cli`.

```python
from transformers import AutoModelForCausalLM
from optimum.quanto import QuantizedModelForCausalLM, qint4

model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B')
qmodel = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude='lm_head')
```

The quantized model can be saved using `save_pretrained`:

```python
qmodel.save_pretrained('./Llama-3.1-8B-quantized')
```

It can later be reloaded using `from_pretrained`:

```python
from optimum.quanto import QuantizedModelForCausalLM

qmodel = QuantizedModelForCausalLM.from_pretrained('Llama-3.1-8B-quantized')
```

You can see more details and [examples](https://github.com/huggingface/optimum-quanto/tree/main/examples) in the [Quanto](https://github.com/huggingface/optimum-quanto) repository.
