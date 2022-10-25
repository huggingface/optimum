[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml)

# Hugging Face Optimum

🤗 Optimum is an extension of 🤗 Transformers, providing a set of performance optimization tools enabling maximum efficiency to train and run models on targeted hardware.

The AI ecosystem evolves quickly and more and more specialized hardware along with their own optimizations are emerging every day.
As such, Optimum enables users to efficiently use any of these platforms with the same ease inherent to transformers.


## Integration with Hardware Partners

🤗 Optimum aims at providing more diversity towards the kind of hardware users can target to train and finetune their models.

To achieve this, we are collaborating with the following hardware manufacturers in order to provide the best transformers integration:
- [Graphcore IPUs](https://github.com/huggingface/optimum-graphcore) - IPUs are a completely new kind of massively parallel processor to accelerate machine intelligence. More information [here](https://www.graphcore.ai/products/ipu).
- [Habana Gaudi Processor (HPU)](https://github.com/huggingface/optimum-habana) - [HPUs](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) are designed to maximize training throughput and efficiency. More information [here](https://habana.ai/training/).
- [Intel](https://github.com/huggingface/optimum-intel) - Enabling the usage of Intel tools to accelerate end-to-end pipelines on Intel architectures. More information about [Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) and [OpenVINO](https://docs.openvino.ai/latest/index.html).
- More to come soon! :star:

## Optimizing models towards inference

Along with supporting dedicated AI hardware for training, Optimum also provides inference optimizations towards various frameworks and
platforms.

Optimum enables the usage of popular compression techniques such as quantization and pruning by supporting [ONNX Runtime](https://onnxruntime.ai/docs/) along with [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html).

| Features                           | ONNX Runtime          | Intel Neural Compressor |
|:----------------------------------:|:---------------------:|:-----------------------:|
| Post-training Dynamic Quantization |  :heavy_check_mark:   |    :heavy_check_mark:   |
| Post-training Static Quantization  |  :heavy_check_mark:   |    :heavy_check_mark:   |
| Quantization Aware Training (QAT)  |  Stay tuned! :star:   |    :heavy_check_mark:   |
| Pruning                            |        N/A            |    :heavy_check_mark:   |


## Installation

🤗 Optimum can be installed using `pip` as follows:

```bash
python -m pip install optimum
```

If you'd like to use the accelerator-specific features of 🤗 Optimum, you can install the required dependencies according to the table below:

| Accelerator                                                                                                            | Installation                                      |
|:-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|
| [ONNX Runtime](https://onnxruntime.ai/docs/)                                                                           | `python -m pip install optimum[onnxruntime]`      |
| [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html)       | `python -m pip install optimum[intel]`            |
| [Graphcore IPU](https://www.graphcore.ai/products/ipu)                                                                 | `python -m pip install optimum[graphcore]`        |
| [Habana Gaudi Processor (HPU)](https://habana.ai/training/)                                                            | `python -m pip install optimum[habana]`           |


If you'd like to play with the examples or need the bleeding edge of the code and can't wait for a new release, you can install the base library from source as follows:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```

For the accelerator-specific features, you can install them by appending `#egg=optimum[accelerator_type]` to the `pip` command, e.g.

```bash
python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]
```

## Quick tour

Check out the examples below to see how 🤗 Optimum can be used to train and run inference on various hardware accelerators.

### Accelerated training

#### Optimum Graphcore

To train transformers on Graphcore's IPUs, 🤗 Optimum provides a `IPUTrainer` that is very similar to the [🤗 Transformers trainer](https://huggingface.co/docs/transformers/main_classes/trainer). Here is a simple example:

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


#### Optimum Habana

To train transformers on Habana's Gaudi processors, 🤗 Optimum provides a `GaudiTrainer` that is very similar to the [🤗 Transformers trainer](https://huggingface.co/docs/transformers/main_classes/trainer). Here is a simple example:

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

#### ONNX Runtime

To train transformers with ONNX Runtime's acceleration features, 🤗 Optimum provides a `ORTTrainer` that is very similar to the [🤗 Transformers trainer](https://huggingface.co/docs/transformers/main_classes/trainer). Here is a simple example:

```diff
- from transformers import Trainer
+ from optimum.onnxruntime import ORTTrainer

  # Download a pretrained model from the Hub
  model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

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


### Accelerated inference

#### ONNX Runtime

To accelerate inference with ONNX Runtime, 🤗 Optimum uses _configuration objects_ to define parameters for optimization. These objects are then used to instantiate dedicated _optimizers_ and _quantizers_.

Before applying quantization or optimization, first export our model to the ONNX format:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
save_directory = "tmp/onnx/"
# Load a model from transformers and export it to ONNX
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, from_transformers=True)
# Save the onnx model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
```

Let's see now how we can apply dynamic quantization with ONNX Runtime:

```python
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# Define the quantization methodology
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained(ort_model)
# Apply dynamic quantization on the model
quantizer.quantize(save_dir=save_directory, quantization_config=qconfig)
```

In this example, we've quantized a model from the Hugging Face Hub, but it could also be a path to a local model directory. The result from applying the `quantize()` method is a `model_quantized.onnx` file that can be used to run inference. Here's an example of how to load an ONNX Runtime model and generate predictions with it:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained(save_directory, file_name="model_quantized.onnx")
tokenizer = AutoTokenizer.from_pretrained(save_directory)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
results = classifier("I love burritos!")
```

#### Optimum Intel

Here is an example on how to perform inference with the OpenVINO Runtime:

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

