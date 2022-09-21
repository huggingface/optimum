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
- [Intel](https://github.com/huggingface/optimum-intel) - Enabling the usage of Intel tools to accelerate end-to-end pipelines on Intel architectures. More information [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html).
- More to come soon! :star:

## Optimizing models towards inference

Along with supporting dedicated AI hardware for training, Optimum also provides inference optimizations towards various frameworks and
platforms.

Optimum enables the usage of popular compression techniques such as quantization and pruning by supporting [ONNX Runtime](https://onnxruntime.ai/docs/) along with Intel [Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) (INC).

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

| Accelerator                                                                                                            | Installation                                 |
|:-----------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|
| [ONNX Runtime](https://onnxruntime.ai/docs/)                                                                           | `python -m pip install optimum[onnxruntime]` |
| [Intel Neural Compressor (INC)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) | `python -m pip install optimum[intel]`       |
| [Graphcore IPU](https://www.graphcore.ai/products/ipu)                                                                 | `python -m pip install optimum[graphcore]`   |
| [Habana Gaudi Processor (HPU)](https://habana.ai/training/)                                                            | `python -m pip install optimum[habana]`      |


If you'd like to play with the examples or need the bleeding edge of the code and can't wait for a new release, you can install the base library from source as follows:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```

For the accelerator-specific features, you can install them by appending `#egg=optimum[accelerator_type]` to the `pip` command, e.g.

```bash
python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]
```

## Quickstart

At its core, 🤗 Optimum uses configuration objects to define parameters for optimization on different accelerators. These objects are then used to instantiate dedicated _optimizers_, _quantizers_, and _pruners_.

### Exporting Transformers models to ONNX

Before applying quantization or optimization, we first need to export our model to the ONNX format.

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
save_directory = "tmp/onnx/"

# Load a model from transformers and export it to ONNX
ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Save the onnx model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
```

### Quantization

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

In this example, we've quantized a model from the Hugging Face Hub, but it could also be a path to a local model directory. The result from applying the `quantize()` method is a `model_quantized.onnx` file that can be used to run inference.

Here's an example of how to load an ONNX Runtime model and generate predictions with it:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained(save_directory, file_name="model_quantized.onnx")
tokenizer = AutoTokenizer.from_pretrained(save_directory)

cls_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

results = cls_pipeline("I love burritos!")
```

Similarly, you can apply static quantization by simply setting `is_static` to `True` when instantiating the `QuantizationConfig` object:

```python
qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
```

Static quantization relies on feeding batches of data through the model to estimate the activation quantization parameters ahead of inference time. To support this, 🤗 Optimum allows you to provide a _calibration dataset_. The calibration dataset can be a simple `Dataset` object from the 🤗 Datasets library, or any dataset that's hosted on the Hugging Face Hub. For this example, we'll pick the [`sst2`](https://huggingface.co/datasets/glue/viewer/sst2/test) dataset that the model was originally trained on:

```python
from functools import partial
from optimum.onnxruntime.configuration import AutoCalibrationConfig

# Define the processing function to apply to each example after loading the dataset
def preprocess_fn(ex, tokenizer):
    return tokenizer(ex["sentence"])

# Create the calibration dataset
calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=50,
    dataset_split="train",
)
# Create the calibration configuration containing the parameters related to calibration.
calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
# Perform the calibration step: computes the activations quantization ranges
ranges = quantizer.fit(
    dataset=calibration_dataset,
    calibration_config=calibration_config,
    operators_to_quantize=qconfig.operators_to_quantize,
)

# Apply static quantization on the model
quantizer.quantize(
    save_dir=save_directory,
    calibration_tensors_range=ranges,
    quantization_config=qconfig,
)
```

### Graph optimization

Then let's take a look at applying _graph optimizations_ techniques such as operator fusion and constant folding. As before, we load a configuration object, but this time by setting the optimization level instead of the quantization approach:

```python
from optimum.onnxruntime.configuration import OptimizationConfig

# Here the optimization level is selected to be 1, enabling basic optimizations such as redundant
# node eliminations and constant folding. Higher optimization level will result in a hardware
# dependent optimized graph.
optimization_config = OptimizationConfig(optimization_level=1)
```

Next, we load an _optimizer_ to apply these optimisations to our model:

```python
from optimum.onnxruntime import ORTOptimizer

optimizer = ORTOptimizer.from_pretrained(ort_model)

# Optimize the model
optimizer.optimize(save_dir=save_directory, optimization_config=optimization_config)
```

And that's it - the model is now optimized and ready for inference!

As you can see, the process is similar in each case:

1. Define the optimization / quantization strategies via an `OptimizationConfig` / `QuantizationConfig` object
2. Instantiate an `ORTQuantizer` or `ORTOptimizer` class
3. Apply the `quantize()` or `optimize()` method
4. Run inference

### Training

Besides supporting ONNX Runtime inference, 🤗 Optimum also supports training with ONNX Runtime backend. The `ORTTrainer` class possess a similar behavior than the `Trainer` of 🤗 Transformers, but reduces the memory consumption and optimize the computation graphs during training. As a result, you will experience an acceleration and feed larger batch size to your device.

Replace `Trainer` with `ORTTrainer` to leverage ONNX Runtime on fine-tuning tasks:

```diff
-from transformers import Trainer
+from optimum.onnxruntime import ORTTrainer

# Step 1: Create your ONNX Runtime Trainer
-trainer = Trainer(
+trainer = ORTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
+    feature="sequence-classification",
)

# Step 2: Use ONNX Runtime for training!🤗
train_result = trainer.train()
```

Check out the [`examples`](https://github.com/huggingface/optimum/tree/main/examples) for more sophisticated usage.

Happy optimizing 🤗!


