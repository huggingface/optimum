# Intel OpenVINO backend for Hugging Face Transformers

This module allows to use [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) backend for the models for [Transformers](https://github.com/huggingface/transformers) library. There are options to use models with PyTorch\*, TensorFlow\* pretrained weights or use native OpenVINO IR format (a pair of files `ov_model.xml` and `ov_model.bin`).

## Usage

To use OpenVINO backend, import one of the `AutoModel` classes with `OV` prefix. Specify a model name or local path in `from_pretrained` method.

```python
from optimum.intel.openvino import OVAutoModel

# PyTorch trained model with OpenVINO backend
model = OVAutoModel.from_pretrained(<name_or_path>, from_pt=True)

# TensorFlow trained model with OpenVINO backend
model = OVAutoModel.from_pretrained(<name_or_path>, from_tf=True)

# Initialize a model from OpenVINO IR
model = OVAutoModel.from_pretrained(<name_or_path>)
```

Supported tasks:

* `OVAutoModel`
* `OVAutoModelForMaskedLM`
* `OVAutoModelWithLMHead`
* `OVAutoModelForQuestionAnswering`
* `OVAutoModelForSequenceClassification`

For more examples please refer to [tests](../../../../tests/openvino/)
