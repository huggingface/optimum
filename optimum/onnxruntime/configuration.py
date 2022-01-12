#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Any, Dict, Tuple, Union

from ..configuration_utils import BaseConfig


class ORTConfig(BaseConfig):
    """
    ORTConfig is the configuration class handling all the ONNX Runtime optimization and quantization parameters.

    Arg:
        opset (`int`, `optional`):
            ONNX opset version to export the model with.
        use_external_data_format (`bool`, `optional`, defaults to `False`):
            Allow exporting model >= than 2Gb.
        seed (`int`, `optional`, defaults to 42):
            The seed used to ensure reproducibility across runs.

        > Parameters for optimization

        opt_level (`int`, `optional`):
            Optimization level performed by ONNX Runtime of the loaded graph.
            Supported optimization level are 0, 1, 2 and 99.
            0 will disable all optimizations.
            1 will enable basic optimizations.
            2 will enable basic and extended optimizations, including complex node fusions applied to the nodes
            assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent.
            99 will enable all available optimizations including layout optimizations.
        use_gpu (`bool`, `optional`, defaults to `False`):
            Whether to optimize the model for GPU inference.
            The optimized graph might contain operators for GPU or CPU only when opt_level > 1.
        only_onnxruntime (`bool`, `optional`, defaults to `False`):
            Whether to only use ONNX Runtime to optimize the model and no graph fusion in Python.

        > Parameters for quantization

        quantization_approach (`str`, `optional`):
            The quantization approach to apply. Supported approach are static and dynamic.
        optimize_model (`bool`, `optional`, defaults to `True`):
            Whether to optimize the model before quantization.
        per_channel (`bool`, `optional`, defaults to `False`):
            Whether to quantize the weights per channel.
        reduce_range (`bool`, `optional`, defaults to `False`):
            Whether to quantize the weights with 7-bits. It may improve the accuracy for some models running on
            non-VNNI machine, especially for per-channel mode
        activation_type (`str`, `optional`, defaults to `"uint8"`):
            The quantization data type of activation.
            Currently, OnnxRuntime CPU only supports activation with type uint8.
        weight_type (`str`, `optional`, defaults to `"uint8"`):
            The quantization data type of weight. Supported data type are uint8 and int8.
        quant_format (`str`, `optional`, defaults to `"operator"`):
            ONNX quantization representation format.
            Supported quantization representation format are "operator" and "qdq".
            "operator" : Operator Oriented (QOperator) : all the quantized operators have their own ONNX definitions.
            "qdq" : Tensor Oriented (QDQ) : this format quantize the model by inserting QuantizeLinear/DeQuantizeLinear
                    on the tensor to simulate the quantize and dequantize process.
                    QuantizeLinear and DeQuantizeLinear operators carry the quantization parameters.
        calibration_method (`str`, `optional`, defaults to `"minmax"`):
            The method chosen to calculate the activations quantization parameters using the calibration dataset.
            Current supported calibration methods are "minmax" and "entropy"
        split (`str`, `optional`, defaults to `"train"`):
            Which split of the calibration dataset to load.
            Depending on the calibration dataset to load, the possible values are "train", "validation" and "test".
        max_samples (`int`, `optional`, defaults to 80):
            Maximum number of examples to use for the calibration step resulting from static quantization.
        calib_batch_size (`int`, `optional`, defaults to 8):
            The batch size to use for the calibration step resulting from static quantization.
    """

    CONFIG_NAME = "ort_config.json"
    FULL_CONFIGURATION_FILE = "ort_config.json"

    def __init__(self, **kwargs):
        self.opset = kwargs.pop("opset", None)
        self.use_external_data_format = kwargs.pop("use_external_data_format", False)
        self.seed = kwargs.pop("seed", 42)
        self.opt_level = kwargs.pop("opt_level", None)
        self.use_gpu = kwargs.pop("use_gpu", False)
        self.only_onnxruntime = kwargs.pop("only_onnxruntime", False)
        self.quantization_approach = kwargs.pop("quantization_approach", None)
        self.optimize_model = kwargs.pop("optimize_model", True)
        self.per_channel = kwargs.pop("per_channel", False)
        self.reduce_range = kwargs.pop("reduce_range", False)
        self.activation_type = kwargs.pop("activation_type", "uint8")
        self.weight_type = kwargs.pop("weight_type", "uint8")
        self.quant_format = kwargs.pop("quant_format", "operator")
        self.calibration_method = kwargs.pop("calibration_method", "minmax")
        self.split = kwargs.pop("split", "train")
        self.max_samples = kwargs.pop("max_samples", 80)
        self.calib_batch_size = kwargs.pop("calib_batch_size", 8)
