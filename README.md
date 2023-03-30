[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml)

# Hugging Face Optimum

🤗 Optimum is an extension of 🤗 Transformers and Diffusers, providing a set of optimization tools enabling maximum efficiency to train and run models on targeted hardware, while keeping things easy to use.

## Installation

🤗 Optimum can be installed using `pip` as follows:

```bash
python -m pip install optimum
```

If you'd like to use the accelerator-specific features of 🤗 Optimum, you can install the required dependencies according to the table below:

| Accelerator                                                                                                            | Installation                                      |
|:-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|
| [ONNX Runtime](https://onnxruntime.ai/docs/)                                                                           | `python -m pip install optimum[onnxruntime]`      |
| [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html)       | `python -m pip install optimum[neural-compressor]`|
| [OpenVINO](https://docs.openvino.ai/latest/index.html)                                                                 | `python -m pip install optimum[openvino,nncf]`    |
| [Graphcore IPU](https://www.graphcore.ai/products/ipu)                                                                 | `python -m pip install optimum[graphcore]`        |
| [Habana Gaudi Processor (HPU)](https://habana.ai/training/)                                                            | `python -m pip install optimum[habana]`           |

To install from source:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```

For the accelerator-specific features, append `#egg=optimum[accelerator_type]` to the above command:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]
```

## Accelerated Inference

🤗 Optimum provides multiple tools to export and run optimized models on various ecosystems: 

- [ONNX](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model) / [ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models)
- TensorFlow Lite
- [OpenVINO](https://huggingface.co/docs/optimum/intel/inference)
- Habana first-gen Gaudi / Gaudi2, more details [here](https://huggingface.co/docs/optimum/main/en/habana/usage_guides/accelerate_inference)

The [export](https://huggingface.co/docs/optimum/exporters/overview) and optimizations can be done both programmatically and with a command line.

### Features summary

| Features                           | ONNX Runtime       | Neural Compressor  | OpenVINO           | TensorFlow Lite    |
|:----------------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Graph optimization                 | :heavy_check_mark: | N/A                | N/A                | N/A                |
| Post-training Dynamic Quantization | :heavy_check_mark: | :heavy_check_mark: | N/A                | :heavy_check_mark: |
| Post-training Static Quantization  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Quantization Aware Training (QAT)  | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |
| fp16 (half precision)              | :heavy_check_mark: | N/A                | N/A                | :heavy_check_mark: |
| Pruning                            | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |

### ONNX + ONNX Runtime

It is possible to export 🤗 Transformers models to the [ONNX](https://onnx.ai/) format and perform graph optimization as well as quantization easily:

```plain
optimum-cli export onnx -m deepset/roberta-base-squad2 --optimize O2 roberta_base_qa_onnx
```

The model can then be quantized using `onnxruntime`:

```bash
optimum-cli onnxruntime quantize --avx512 --onnx_model roberta_base_qa_onnx
```

These commands will export `deepset/roberta-base-squad2` and perform [O2 graph optimization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization#optimization-configuration) on the exported model, and finally quantize it with the [avx512 configuration](https://huggingface.co/docs/optimum/main/en/onnxruntime/package_reference/configuration#optimum.onnxruntime.AutoQuantizationConfig.avx512).

For more information on the ONNX export, please check the [documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model).

#### Run the exported model using ONNX Runtime

Once the model is exported to the ONNX format, we provide Python classes enabling you to run the exported ONNX model in a seemless manner using [ONNX Runtime](https://onnxruntime.ai/) in the backend:

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering

model_name = "roberta_base_qa_onnx"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ort_model = ORTModelForQuestionAnswering.from_pretrained(model_name)

question = "What's Optimum?"
text = "Optimum is an awesome library everyone should use!"
inputs = tokenizer(question, text, return_tensors="pt") 

# Run with ONNX Runtime.
outputs = ort_model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
```

More details on how to run ONNX models with `ORTModelForXXX` classes [here](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models).

### TensorFlow Lite

Just as for ONNX, it is possible to export models to [TensorFlow Lite](https://www.tensorflow.org/lite) and quantize them:

```plain
optimum-cli export tflite \
  -m deepset/roberta-base-squad2 \
  --sequence_length 384  \
  --quantize int8-dynamic roberta_tflite_model
```
### OpenVINO

*This requires to install the Optimum OpenVINO extra by doing `pip install optimum[openvino,nncf]`.*

To load a model and run inference with [OpenVINO Runtime](https://docs.openvino.ai/latest/home.html), you can just replace your `AutoModelForXxx` class with the corresponding `OVModelForXxx` class.
If you want to load a PyTorch checkpoint, set `from_transformers=True` to convert your model to the OpenVINO IR (Intermediate Representation).

```diff
- from transformers import AutoModelForSequenceClassification
+ from optimum.intel.openvino import OVModelForSequenceClassification
  from transformers import AutoTokenizer, pipeline

  # Download a tokenizer and model from the Hub and convert to OpenVINO format
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model_id = "distilbert-base-uncased-finetuned-sst-2-english"
- model = AutoModelForSequenceClassification.from_pretrained(model_id)
+ model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)

  # Run inference!
  classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
  results = classifier("He's a dreadful magician.")
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/intel/inference) and in the [examples](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino).


## Accelerated training

🤗 Optimum provides wrappers around the original 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to enable training on powerful hardware easily.
We support many providers:

- Habana's Gaudi processors
- Graphcore's IPUs 
- ONNX Runtime (optimized for GPUs)

### Habana

<!--
To train transformers on Habana's Gaudi processors, 🤗 Optimum provides a `GaudiTrainer` that is very similar to the 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer). Here is a simple example:

-->
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


### Graphcore

<!--
To train transformers on Graphcore's IPUs, 🤗 Optimum provides a `IPUTrainer` that is very similar to the 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer). Here is a simple example:

-->
```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.graphcore import IPUConfig, IPUTrainer, IPUTrainingArguments

  # Download a pretrained model from the Hub
  model = AutoModelForXxx.from_pretrained("bert-base-uncased")

  # Define the training arguments
- training_args = TrainingArguments(
+ training_args = IPUTrainingArguments(
      output_dir="path/to/save/folder/",
+     ipu_config_name="Graphcore/bert-base-ipu", # Any IPUConfig on the Hub or stored locally
      ...
  )

  # Define the configuration to compile and put the model on the IPU
+ ipu_config = IPUConfig.from_pretrained(training_args.ipu_config_name)

  # Initialize the trainer
- trainer = Trainer(
+ trainer = IPUTrainer(
      model=model,
+     ipu_config=ipu_config
      args=training_args,
      train_dataset=train_dataset
      ...
  )

  # Use Graphcore IPU for training!
  trainer.train()
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/graphcore/quickstart) and in the [examples](https://github.com/huggingface/optimum-graphcore/tree/main/examples).


### ONNX Runtime

<!--
To train transformers with ONNX Runtime's acceleration features, 🤗 Optimum provides a `ORTTrainer` that is very similar to the 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer). Here is a simple example:
-->

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
+     feature="sequence-classification", # The model type to export to ONNX
      ...
  )

  # Use ONNX Runtime for training!
  trainer.train()
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/trainer) and in the [examples](https://github.com/huggingface/optimum/tree/main/examples/onnxruntime/training).
