[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml)
[![neural_compressor](https://github.com/huggingface/optimum/actions/workflows/test_intel.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_intel.yml)

# Hugging Face Optimum

🤗 Optimum is an extension of 🤗 Transformers, providing a set of performance optimization tools enabling maximum efficiency to train and run models on targeted hardware.

The AI ecosystem evolves quickly and more and more specialized hardware along with their own optimizations are emerging every day.
As such, Optimum enables users to efficiently use any of these platforms with the same ease inherent to transformers.


## Integration with Hardware Partners

🤗 Optimum aims at providing more diversity towards the kind of hardware users can target to train and finetune their models.

To achieve this, we are collaborating with the following hardware manufacturers in order to provide the best transformers integration:
- [Graphcore IPUs](https://github.com/huggingface/optimum-graphcore) - IPUs are a completely new kind of massively parallel processor to accelerate machine intelligence. [More information here](https://www.graphcore.ai/products/ipu)
- More to come soon! :star:

## Optimizing models towards inference

Along with supporting dedicated AI hardware for training, Optimum also provides inference optimizations towards various frameworks and
platforms.


We currently support [ONNX runtime](https://github.com/microsoft/onnxruntime) along with [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor).

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

| Accelerator                                                                 | Installation                                 |
|:----------------------------------------------------------------------------|:---------------------------------------------|
| [ONNX runtime](https://github.com/microsoft/onnxruntime)                    | `python -m pip install optimum[onnxruntime]` |
| [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) | `python -m pip install optimum[intel]`       |
| [Graphcore IPU](https://www.graphcore.ai/products/ipu)                      | `python -m pip install optimum[graphcore]`   |


If you'd like to play with the examples or need the bleeding edge of the code and can't wait for a new release, you can install the base library from source as follows:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```

For the accelerator-specific features, you can install them by appending `#egg=optimum[accelerator_type]` to the `pip` command, e.g.

```bash
python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]
```

## Quickstart

At its core, 🤗 Optimum uses _configuration objects_ to define parameters for optimization on different accelerators. These objects are then used to instantiate dedicated _optimizers_, _quantizers_, and _pruners_. For example, here's how you can apply dynamic quantization with ONNX Runtime:

```python
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# The model we wish to quantize
model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# The type of quantization to apply
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained(model_checkpoint, feature="sequence-classification")

# Quantize the model!
quantizer.export(
    onnx_model_path="model.onnx",
    onnx_quantized_model_output_path="model-quantized.onnx",
    quantization_config=qconfig,
)
```

In this example, we've quantized a model from the Hugging Face Hub, but it could also be a path to a local model directory. The `feature` argument in the `from_pretrained()` method corresponds to the type of task that we wish to quantize the model for. The result from applying the `export()` method is a `model-quantized.onnx` file that can be used to run inference. Here's an example of how to load an ONNX Runtime model and generate predictions with it:

```python
from functools import partial
from datasets import Dataset
from optimum.onnxruntime import ORTModel

# Load quantized model
ort_model = ORTModel("model-quantized.onnx", quantizer._onnx_config)
# Create a dataset or load one from the Hub
ds = Dataset.from_dict({"sentence": ["I love burritos!"]})
# Tokenize the inputs
def preprocess_fn(ex, tokenizer):
    return tokenizer(ex["sentence"])

tokenized_ds = ds.map(partial(preprocess_fn, tokenizer=quantizer.tokenizer))
ort_outputs = ort_model.evaluation_loop(tokenized_ds)
# Extract logits!
ort_outputs.predictions
```

Similarly, you can apply static quantization by simply setting `is_static` to `True` when instantiating the `QuantizationConfig` object:

```python
qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
```

Static quantization relies on feeding batches of data through the model to estimate the activation quantization parameters ahead of inference time. To support this, 🤗 Optimum allows you to provide a _calibration dataset_. The calibration dataset can be a simple `Dataset` object from the 🤗 Datasets library, or any dataset that's hosted on the Hugging Face Hub. For this example, we'll pick the [`sst2`](https://huggingface.co/datasets/glue/viewer/sst2/test) dataset that the model was originally trained on:

```python
from optimum.onnxruntime.configuration import AutoCalibrationConfig

# Create the calibration dataset
calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=quantizer.tokenizer),
    num_samples=50,
    dataset_split="train",
)
# Create the calibration configuration containing the parameters related to calibration.
calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
# Perform the calibration step: computes the activations quantization ranges
ranges = quantizer.fit(
    dataset=calibration_dataset,
    calibration_config=calibration_config,
    onnx_model_path="model.onnx",
    operators_to_quantize=qconfig.operators_to_quantize,
)
# Quantize the same way we did for dynamic quantization!
quantizer.export(
    onnx_model_path="model.onnx",
    onnx_quantized_model_output_path="model-quantized.onnx",
    calibration_tensors_range=ranges,
    quantization_config=qconfig,
)
```

As a final example, let's take a look at applying _graph optimizations_ techniques such as operator fusion and constant folding. As before, we load a configuration object, but this time by setting the optimization level instead of the quantization approach:

```python
from optimum.onnxruntime.configuration import OptimizationConfig, ORTConfig

# optimization_config=99 enables all available graph optimisations
optimization_config = OptimizationConfig(optimization_level=99)
ort_config = ORTConfig(optimization_config=optimization_config)
```

Next, we load an _optimizer_ to apply these optimisations to our model:

```python
from optimum.onnxruntime import ORTOptimizer

optimizer = ORTOptimizer(ort_config)
optimizer.fit(model_checkpoint, output_dir=".", feature="sequence-classification")
```

And that's it - the model is now optimized and ready for inference!

Check out the [`examples`](https://github.com/huggingface/optimum/tree/main/examples) directory for more sophisticated usage.

Happy optimising 🤗!


