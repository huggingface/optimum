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

from typing import Any, Dict, List, Optional

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
            Current supported calibration methods are "minmax", "entropy" and "percentile".
        split (`str`, `optional`, defaults to `"train"`):
            Which split of the calibration dataset to load.
            Depending on the calibration dataset to load, the possible values are "train", "validation" and "test".
        max_samples (`int`, `optional`, defaults to 80):
            Maximum number of examples to use for the calibration step resulting from static quantization.
        calib_batch_size (`int`, `optional`, defaults to 8):
            The batch size to use for the calibration step resulting from static quantization.
        op_types_to_quantize (`List`, `optional`):
            List of the types of operators to quantize. By default, all the supported operators are quantized.
        nodes_to_quantize (`List`, `optional`):
            List of the nodes names to quantize.
        nodes_to_exclude (`List`, `optional`):
            List of the nodes names to exclude when applying quantization.
        extra_options (`Dict[str, Any]`, `optional`):
            The dictionary mapping each extra options to the desired value, such as :
                ActivationSymmetric (`bool`, `optional`, defaults to `False`):
                    Symmetrize calibration data for activations.
                WeightSymmetric (`bool`, `optional`, defaults to `True`):
                    Symmetrize calibration data for weights.
                EnableSubgraph (`bool`, `optional`, defaults to `False`):
                    If enabled, subgraph will be quantized.
                DisableShapeInference (`bool`, `optional`, defaults to `False`):
                    In dynamic quantization mode, shape inference is not mandatory and can be disabled in case it causes
                    issues.
                ForceQuantizeNoInputCheck (`bool`, `optional`, defaults to `False`):
                    By default, the outputs of some latent operators such as maxpool or transpose are not quantized if
                    the corresponding input is not already quantized. When set to True, this option will force such
                    operator to always quantize their input, resulting in quantized output.
                MatMulConstBOnly (`bool`, `optional`, defaults to `False`):
                    If enabled, only MatMul with const B will be quantized.
                AddQDQPairToWeight (`bool`, `optional`, defaults to `False`):
                    By default, floating-point weights are quantized and feed to solely inserted DeQuantizeLinear node.
                    If set to True, the floating-point weights will remain and both QuantizeLinear/DeQuantizeLinear
                    nodes will be inserted.
                OpTypesToExcludeOutputQuantization (`List`, `optional`, defaults to `[]`):
                    If any op type is specified, the output of ops with this specific op types will not be quantized.
                DedicatedQDQPair (`bool`, `optional`, defaults to `False`):
                    When inserting QDQ pair, multiple nodes can share a single QDQ pair as their inputs. If True, it
                    will create an identical and dedicated QDQ pair for each node.
                QDQOpTypePerChannelSupportToAxis (`Dict`, `optional`, defaults to `{}`):
                    Set the channel axis for a specific op type. Effective only when per channel quantization is
                    supported and per_channel is set to True.
                CalibMovingAverage (`bool`, `optional`, defaults to `False`):
                    If enabled, the moving average of the minimum and maximum values will be computed when the
                    calibration method selected is MinMax.
                CalibMovingAverage (`float`, `optional`, defaults to `0.01`):
                    Constant smoothing factor to use when computing the moving average of the minimum and maximum
                    values. Effective only when the calibration method selected is MinMax and when CalibMovingAverage
                    is set to True.
    """

    CONFIG_NAME = "ort_config.json"
    FULL_CONFIGURATION_FILE = "ort_config.json"

    def __init__(
        self,
        opset: Optional[int] = None,
        opt_level: Optional[int] = None,
        use_gpu: Optional[bool] = False,
        only_onnxruntime: Optional[bool] = False,
        quantization_approach: Optional[str] = None,
        optimize_model: Optional[bool] = True,
        per_channel: Optional[bool] = False,
        reduce_range: Optional[bool] = False,
        activation_type: Optional[str] = "uint8",
        weight_type: Optional[str] = "uint8",
        quant_format: Optional[str] = "operator",
        calibration_method: Optional[str] = "minmax",
        split: Optional[str] = "train",
        max_samples: Optional[int] = 80,
        calib_batch_size: Optional[int] = 8,
        seed: Optional[int] = 42,
        use_external_data_format: Optional[bool] = False,
        op_types_to_quantize: Optional[List] = None,
        nodes_to_quantize: Optional[List] = None,
        nodes_to_exclude: Optional[List] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        self.opset = opset
        self.opt_level = opt_level
        self.use_gpu = use_gpu
        self.only_onnxruntime = only_onnxruntime
        self.quantization_approach = quantization_approach
        self.optimize_model = optimize_model
        self.per_channel = per_channel
        self.reduce_range = reduce_range
        self.activation_type = activation_type
        self.weight_type = weight_type
        self.quant_format = quant_format
        self.calibration_method = calibration_method
        self.split = split
        self.max_samples = max_samples
        self.calib_batch_size = calib_batch_size
        self.seed = seed
        self.use_external_data_format = use_external_data_format
        self.op_types_to_quantize = op_types_to_quantize
        self.nodes_to_quantize = nodes_to_quantize
        self.nodes_to_exclude = nodes_to_exclude
        self.extra_options = {} if extra_options is None else extra_options
