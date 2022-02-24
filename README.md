[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml)
[![neural_compressor](https://github.com/huggingface/optimum/actions/workflows/test_intel.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_intel.yml)

# Hugging Face Optimum

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

For the acclerator-specific features, you can install them by appending `#egg=optimum[accelerator_type]` to the `pip` command, e.g.

```bash
python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]
```

## Quickstart

At its core, 🤗 Optimum uses _configuration objects_ to define parameters for optimization on different accelerators. These objects are then used to instantiate dedicated _optimizers_, _quantizers_, and _pruners_. For example, here's how you can apply dynamic quantization with ONNX Runtime:

```python
from optimum.onnxruntime import ORTConfig, ORTQuantizer

# The model we wish to quantize
model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
# The type of quantization to apply
ort_config = ORTConfig(quantization_approach="dynamic")
quantizer = ORTQuantizer(ort_config)
# Quantize the model!
quantizer.fit(model_ckpt, output_dir=".", feature="sequence-classification")
```

In this example, we've quantized a model from the Hugging Face Hub, but it could also be a path to a local model directory. The `feature` argument in the `fit()` method corresponds to the type of task that we wish to quantize the model for. The result from applying the `fit()` method is a `model-quantized.onnx` file that can be used to run inference. Here's an example of how to load an ONNX Runtime model and generate predictions with it:

```python
from datasets import Dataset
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModel

# Load quantized model
ort_model = ORTModel("model-quantized.onnx", quantizer.onnx_config)
# Create a dataset or load one from the Hub
ds = Dataset.from_dict({"sentence": ["I love burritos!"]})
# Tokenize the inputs & convert to PyTorch tensors
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def preprocess_fn(ex):
    return tokenizer(ex["sentence"])

tokenized_ds = ds.map(preprocess_fn, remove_columns=ds.column_names)
tokenized_ds.set_format("torch")
# Create dataloader and run evaluation
dataloader = DataLoader(tokenized_ds)
ort_outputs = ort_model.evaluation_loop(dataloader)
# Extract logits!
ort_outputs.predictions
```

Similarly, you can apply static quantization by simply changing the `quantization_approach` in the `ORTConfig` object:

```python
ort_config = ORTConfig(quantization_approach="static")
```

Static quantization relies on feeding batches of data through the model to observe the activation patterns ahead of inference time. The ideal quantization scheme is then calculated and saved. To support this, 🤗 Optimum allows you to provide a _calibration dataset_. The calibration dataset can be a simple `Dataset` object from the 🤗 Datasets library, or any dataset that's hosted on the Hugging Face Hub. For this example, we'll pick the [`sst2`](https://huggingface.co/datasets/glue/viewer/sst2/test) dataset that the model was originally trained on:

```python
from transformers import DataCollatorWithPadding

# We use a data collator to pad the examples in a batch
data_collator = DataCollatorWithPadding(tokenizer)
# For calibration we define the dataset and preprocessing function
quantizer = ORTQuantizer(
    ort_config,
    dataset_name="glue",
    dataset_config_name="sst2",
    preprocess_function=preprocess_fn,
    data_collator=data_collator,
)
# Quantize the same way we did for dynamic quantization!
quantizer.fit(model_ckpt, output_dir=".", feature="sequence-classification")
```

As a final example, let's take a look at applying _graph optimizations_ techniques such as operator fusion and constant folding. As before, we load a configuration object, but this time by setting the optimization level instead of the quantization approach:

```python
# opt_level=99 enables all graph optimisations
ort_config = ORTConfig(opt_level=99)
```

Next, we load an _optimizer_ to apply these optimisations to our model:

```python
from optimum.onnxruntime import ORTOptimizer

optimizer = ORTOptimizer(ort_config)
optimizer.fit(model_ckpt, output_dir=".", feature="sequence-classification")
```

And that's it - the model is now optimized and ready for inference! As you can see, the process is similar in each case:

1. Define the optimization / quantization strategies via an `ORTConfig` object
2. Instantiate a `ORTQuantizer` or `ORTOptimizer` class
3. Apply the `fit()` method
4. Run inference

Check out the [`examples`](https://github.com/huggingface/optimum/tree/main/examples) directory for more sophisticated usage.

Happy optimising 🤗!


