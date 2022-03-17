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
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset
from packaging.version import Version, parse

from onnxruntime import GraphOptimizationLevel
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibraterBase, CalibrationMethod, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.calibrate import create_calibrator
from optimum.onnxruntime import ORT_DEFAULT_CHANNEL_FOR_OPERATORS, ORTQuantizableOperator

from ..configuration_utils import BaseConfig


NodeName = NodeType = str


@dataclass
class CalibrationConfig:
    dataset_name: str
    dataset_config_name: Optional[str]
    dataset_split: Optional[str]
    dataset_num_samples: int
    method: CalibrationMethod
    num_bins: Optional[int] = None
    num_quantized_bins: Optional[int] = None
    percentiles: Optional[float] = None
    moving_average: Optional[bool] = None
    averaging_constant: Optional[float] = None

    def create_calibrator(
        self,
        onnx_model_path: Union[str, os.PathLike, Path],
        operators_to_quantize: Optional[List[Union[NodeType, ORTQuantizableOperator]]] = [
            ORTQuantizableOperator.FullyConnected
        ],
        use_external_data_format: bool = False,
        force_symmetric_range: bool = False,
        augmented_model_name: str = "augmented_model.onnx",
    ) -> CalibraterBase:

        operators_to_calibrate = (operators_to_quantize or []).copy()

        if ORTQuantizableOperator.FullyConnected in operators_to_calibrate:
            operators_to_calibrate.remove(ORTQuantizableOperator.FullyConnected)
            operators_to_calibrate += [ORTQuantizableOperator.MatMul, ORTQuantizableOperator.Add]

        operators_to_calibrate = [
            operator.value if isinstance(operator, ORTQuantizableOperator) else operator
            for operator in operators_to_calibrate
        ]

        kwargs = {
            "model": onnx_model_path,
            "op_types_to_calibrate": operators_to_calibrate,
            "calibrate_method": self.method,
            "augmented_model_path": augmented_model_name,
        }
        if parse(ort_version) > Version("1.10.0"):
            kwargs["use_external_data_format"] = use_external_data_format
            kwargs["extra_options"] = {
                "symmetric": force_symmetric_range,
                "num_bins": self.num_bins,
                "num_quantized_bins": self.num_quantized_bins,
                "percentiles": self.percentiles,
                "moving_average": self.moving_average,
                "averaging_constant": self.averaging_constant,
            }
        return create_calibrator(**kwargs)


class AutoCalibrationConfig:
    @staticmethod
    def minmax(dataset: Dataset, moving_average: bool = False, averaging_constant: float = 0.01) -> CalibrationConfig:
        """

        :param dataset: The dataset to use to calibrate the model
        :param moving_average:
        :param averaging_constant:
        :return:
        """
        if moving_average and parse(ort_version) < Version("1.10.99"):
            raise NotImplementedError(
                "MinMax calibration method using the moving average for the activations quantization parameters "
                "computation is only implemented for onnxruntime >= 1.11.0."
            )

        if moving_average and not 0 <= averaging_constant <= 1:
            raise ValueError(f"Invalid averaging constant value ({averaging_constant}) should be within [0, 1]")

        return CalibrationConfig(
            dataset_name=dataset.info.builder_name,
            dataset_config_name=dataset.info.config_name,
            dataset_split=str(dataset.split),
            dataset_num_samples=dataset.num_rows,
            method=CalibrationMethod.MinMax,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
        )

    @staticmethod
    def entropy(
        dataset: Dataset,
        num_bins: int = 128,
        num_quantized_bins: int = 128,
    ) -> CalibrationConfig:
        """

        :param dataset:
        :param num_bins:
        :param num_quantized_bins:
        :return:
        """
        if parse(ort_version) <= Version("1.10.99"):
            raise NotImplementedError("entropy calibration method is only implemented for onnxruntime >= 1.11.0")

        if num_bins <= 0:
            raise ValueError(f"Invalid value num_bins ({num_bins}) should be >= 1")

        if num_quantized_bins <= 0:
            raise ValueError(f"Invalid value num_quantized_bins ({num_quantized_bins}) should be >= 1")

        return CalibrationConfig(
            dataset_name=dataset.info.builder_name,
            dataset_config_name=dataset.info.config_name,
            dataset_split=str(dataset.split),
            dataset_num_samples=dataset.num_rows,
            method=CalibrationMethod.Entropy,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )

    @staticmethod
    def percentiles(
        dataset: Dataset,
        num_bins: int = 2048,
        num_quantized_bins: int = 128,
        percentiles: float = 99.999,
    ) -> CalibrationConfig:
        """

        :param dataset:
        :param num_bins:
        :param num_quantized_bins:
        :param percentiles:
        :return:
        """

        # if parse(ort_version) <= Version("1.10.99"):
        #     raise NotImplementedError("percentiles calibration method is only implemented for onnxruntime > 1.10.0")

        if num_bins <= 0:
            raise ValueError(f"Invalid value num_bins ({num_bins}) should be >= 1")

        if num_quantized_bins <= 0:
            raise ValueError(f"Invalid value num_quantized_bins ({num_quantized_bins}) should be >= 1")

        if not 0 <= percentiles <= 100:
            raise ValueError(f"Invalid value percentiles ({percentiles}) should be within [0; 100.[")

        return CalibrationConfig(
            dataset_name=dataset.info.builder_name,
            dataset_config_name=dataset.info.config_name,
            dataset_split=str(dataset.split),
            dataset_num_samples=dataset.num_rows,
            method=CalibrationMethod.Percentile,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
            percentiles=percentiles,
        )


@dataclass
class QuantizationConfig:
    is_static: bool
    format: QuantFormat.QDQ
    mode: QuantizationMode = QuantizationMode.QLinearOps
    activations_dtype: QuantType = QuantType.QInt8
    activations_symmetric: bool = False
    weights_dtype: QuantType = QuantType.QInt8
    weights_symmetric: bool = True
    per_channel: bool = False
    reduce_range: bool = False
    nodes_to_quantize: List[NodeName] = None
    nodes_to_exclude: List[NodeName] = field(default_factory=list)
    operators_to_quantize: List[NodeType] = field(default_factory=list)
    qdq_add_pair_to_weight: bool = False
    qdq_dedicated_pair: bool = False
    qdq_op_type_per_channel_support_to_axis: Dict[str, int] = field(
        default_factory=lambda: ORT_DEFAULT_CHANNEL_FOR_OPERATORS
    )

    def __post_init__(self):
        ensure_valid_mode_or_raise(self.is_static, self.mode)
        ensure_valid_data_type_or_raise(self.is_static, self.activations_dtype, self.weights_dtype)

    @staticmethod
    def quantization_type_str(activations_dtype: QuantType, weights_dtype: QuantType) -> str:
        return (
            f"{'s8' if activations_dtype == QuantType.QInt8 else 'u8'}"
            f"/"
            f"{'s8' if weights_dtype == QuantType.QInt8 else 'u8'}"
        )

    def __str__(self):
        return (
            f"{self.format} ("
            f"mode: {self.mode}, "
            f"schema: {QuantizationConfig.quantization_type_str(self.activations_dtype, self.weights_dtype)}, "
            f"channel-wise: {self.per_channel})"
        )


def ensure_valid_mode_or_raise(use_static_quantization: bool, mode: QuantizationMode):
    if not use_static_quantization and mode == QuantizationMode.QLinearOps:
        raise ValueError(
            "Invalid combination of "
            "use_static_quantization = False "
            "and "
            "mode = QuantizationMode.QLinearOps. "
            "OnnxRuntime dynamic quantization requires mode = QuantizationMode.IntegerOps"
        )


def ensure_valid_data_type_or_raise(
    use_static_quantization: bool, activations_dtype: QuantType, weights_dtype: QuantType
):
    if not use_static_quantization and activations_dtype == QuantType.QInt8:
        raise ValueError(
            "Invalid combination of "
            "use_static_quantization = False "
            "and "
            "activations_dtype = QuantType.QInt8. "
            "OnnxRuntime dynamic quantization requires activations_dtype = QuantType.QUInt8"
        )

    if use_static_quantization and activations_dtype == QuantType.QInt8 and weights_dtype == QuantType.QUInt8:
        raise ValueError(
            "Invalid combination of "
            "use_static_quantization = True, "
            "activations_dtype = QuantType.QInt8 "
            "and "
            "weights_dtype = QuantType.QUInt8."
            "OnnxRuntime static quantization does not support "
            "activations_dtype = QuantType.QInt8 with weights_dtype = QuantType.QUInt8."
        )


def default_quantization_parameters(
    is_static: bool, format: Optional[QuantFormat] = None, mode: Optional[QuantizationMode] = None
) -> Tuple[QuantFormat, QuantizationMode]:
    if format is None:
        format = QuantFormat.QDQ if is_static else QuantFormat.QOperator

    if mode is None:
        mode = QuantizationMode.QLinearOps if is_static else QuantizationMode.IntegerOps

    return format, mode


class AutoQuantizationConfig:
    @staticmethod
    def arm64(
        is_static: bool,
        format: Optional[QuantFormat] = None,
        mode: Optional[QuantizationMode] = None,
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[Union[NodeType, ORTQuantizableOperator]] = [ORTQuantizableOperator.FullyConnected],
    ) -> QuantizationConfig:
        """

        :param is_static: Boolean flag to indicate whether we target static or dynamic quantization.
        :param format: Targeted ONNX Runtime quantization format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param mode: Targeted ONNX Runtime quantization mode, default is QLinearOps to match QDQ format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param use_symmetric_activations:
        :param use_symmetric_weights:
        :param per_channel: Whether we should quantize per-channel (also known as "per-row"). Enabling this can
            increase overall accuracy while making the quantized model heavier.
        :param nodes_to_quantize:
        :param nodes_to_exclude:
        :param operators_to_quantize:
        :return:
        """
        if format is None:
            format = QuantFormat.QDQ if is_static else QuantFormat.QOperator

        if mode is None:
            mode = QuantizationMode.QLinearOps if is_static else QuantizationMode.IntegerOps

        # u8/s8 is faster (than u8/u8) on lower-end ARM64 and identical on higher-end ARM64,
        # so let's use u8/s8 by default
        return QuantizationConfig(
            is_static=is_static,
            format=format,
            mode=mode,
            activations_dtype=QuantType.QUInt8,
            activations_symmetric=use_symmetric_activations,
            weights_dtype=QuantType.QInt8,
            weights_symmetric=use_symmetric_weights,
            per_channel=per_channel,
            reduce_range=False,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            operators_to_quantize=operators_to_quantize,
        )

    @staticmethod
    def avx2(
        is_static: bool,
        format: Optional[QuantFormat] = None,
        mode: Optional[QuantizationMode] = None,
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        reduce_range: bool = False,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[Union[NodeType, ORTQuantizableOperator]] = [ORTQuantizableOperator.FullyConnected],
    ) -> QuantizationConfig:
        """

        :param is_static: Boolean flag to indicate whether we target static or dynamic quantization.
        :param format: Targeted ONNX Runtime quantization format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param mode: Targeted ONNX Runtime quantization mode, default is QLinearOps to match QDQ format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param use_symmetric_activations:
        :param use_symmetric_weights:
        :param per_channel: Whether we should quantize per-channel (also known as "per-row"). Enabling this can
            increase overall accuracy while making the quantized model heavier.
        :param reduce_range: Indicate whether to use 8-bits integers (False) or reduce-range 7-bits integers (True).
            As a baseline, it is always recommended testing with full range (reduce_range = False) and then, if
            accuracy drop is significant, to try with reduced range (reduce_range = True).
            Intel's CPUs using AVX512 (non VNNI) can suffer from saturation issue when invoking
            the VPMADDUBSW instruction. To counter this, one should use 7-bits rather than 8-bits integers.
        :param nodes_to_quantize:
        :param nodes_to_exclude:
        :param operators_to_quantize:
        :return:
        """
        format, mode = default_quantization_parameters(is_static, format, mode)

        return QuantizationConfig(
            is_static=is_static,
            format=format,
            mode=mode,
            activations_dtype=QuantType.QUInt8,
            activations_symmetric=use_symmetric_activations,
            weights_dtype=QuantType.QUInt8,
            weights_symmetric=use_symmetric_weights,
            per_channel=per_channel,
            reduce_range=reduce_range,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            operators_to_quantize=operators_to_quantize,
        )

    @staticmethod
    def avx512(
        is_static: bool,
        format: Optional[QuantFormat] = None,
        mode: Optional[QuantizationMode] = None,
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        reduce_range: bool = False,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[Union[NodeType, ORTQuantizableOperator]] = [ORTQuantizableOperator.FullyConnected],
    ) -> QuantizationConfig:
        """

        :param is_static: Boolean flag to indicate whether we target static or dynamic quantization.
        :param format: Targeted ONNX Runtime quantization format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param mode: Targeted ONNX Runtime quantization mode, default is QLinearOps to match QDQ format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param use_symmetric_activations:
        :param use_symmetric_weights:
        :param per_channel: Whether we should quantize per-channel (also known as "per-row"). Enabling this can
            increase overall accuracy while making the quantized model heavier.
        :param reduce_range: Indicate whether to use 8-bits integers (False) or reduce-range 7-bits integers (True).
            As a baseline, it is always recommended testing with full range (reduce_range = False) and then, if
            accuracy drop is significant, to try with reduced range (reduce_range = True).
            Intel's CPUs using AVX512 (non VNNI) can suffer from saturation issue when invoking
            the VPMADDUBSW instruction. To counter this, one should use 7-bits rather than 8-bits integers.
        :param nodes_to_quantize:
        :param nodes_to_exclude:
        :param operators_to_quantize:
        :return:
        """
        format, mode = default_quantization_parameters(is_static, format, mode)

        return QuantizationConfig(
            is_static=is_static,
            format=format,
            mode=mode,
            activations_dtype=QuantType.QUInt8,
            activations_symmetric=use_symmetric_activations,
            weights_dtype=QuantType.QInt8,
            weights_symmetric=use_symmetric_weights,
            per_channel=per_channel,
            reduce_range=reduce_range,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            operators_to_quantize=operators_to_quantize,
        )

    @staticmethod
    def avx512_vnni(
        is_static: bool,
        format: Optional[QuantFormat] = None,
        mode: Optional[QuantizationMode] = None,
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[Union[NodeType, ORTQuantizableOperator]] = [ORTQuantizableOperator.FullyConnected],
    ) -> QuantizationConfig:
        """
        When targeting Intel AVX512-VNNI CPU underlying execution engine leverage the CPU instruction VPDPBUSD to
        compute  \\i32 += i8(w) * u8(x)\\ within a single instruction.

        AVX512-VNNI (AVX512 Vector Neural Network Instruction)
        is an x86 extension Instruction set and is a part of the AVX-512 ISA.

        AVX512 VNNI is designed to accelerate convolutional neural network for INT8 inference.

        :param is_static: Boolean flag to indicate whether we target static or dynamic quantization.
        :param format: Targeted ONNX Runtime quantization format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param mode: Targeted ONNX Runtime quantization mode, default is QLinearOps to match QDQ format.
            When targeting dynamic quantization mode, the default value is `QuantFormat.QOperator` whereas the default
            value for static quantization mode is `QuantFormat.QLinearOps`
        :param use_symmetric_activations:
        :param use_symmetric_weights:
        :param per_channel: Whether we should quantize per-channel (also known as "per-row"). Enabling this can
            increase overall accuracy while making the quantized model heavier.
        :param nodes_to_quantize:
        :param nodes_to_exclude:
        :param operators_to_quantize:
        :return:
        """
        format, mode = default_quantization_parameters(is_static, format, mode)

        return QuantizationConfig(
            is_static=is_static,
            format=format,
            mode=mode,
            activations_dtype=QuantType.QUInt8,
            activations_symmetric=use_symmetric_activations,
            weights_dtype=QuantType.QInt8,
            weights_symmetric=use_symmetric_weights,
            per_channel=per_channel,
            reduce_range=False,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            operators_to_quantize=operators_to_quantize,
        )

    @staticmethod
    def tensorrt(
        is_static: bool,
        format: Optional[QuantFormat] = None,
        mode: Optional[QuantizationMode] = None,
        per_channel: bool = True,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[Union[NodeType, ORTQuantizableOperator]] = [ORTQuantizableOperator.FullyConnected],
    ) -> QuantizationConfig:
        format, mode = default_quantization_parameters(is_static, format, mode)

        return QuantizationConfig(
            is_static=is_static,
            format=format,
            mode=mode,
            activations_dtype=QuantType.QInt8,
            activations_symmetric=True,  # TRT only supports symmetric
            weights_dtype=QuantType.QInt8,
            weights_symmetric=True,  # TRT only supports symmetric
            per_channel=per_channel,
            reduce_range=False,
            nodes_to_quantize=nodes_to_quantize or [],
            nodes_to_exclude=nodes_to_exclude or [],
            operators_to_quantize=operators_to_quantize,
            qdq_add_pair_to_weight=True,
            qdq_dedicated_pair=True,
        )


@dataclass
class OptimizationConfig:
    """
    Optimization level performed by ONNX Runtime of the loaded graph.
    Supported optimization level are 0, 1, 2 and 99.
    0 will disable all optimizations (GraphOptimizationLevel.ORT_DISABLE_ALL).
    1 will enable basic optimizations. (GraphOptimizationLevel.ORT_ENABLE_BASIC)
    2 will enable basic and extended optimizations, including complex node fusions applied to the nodes
    assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent.
    (GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    99 will enable all available optimizations including layout optimizations. (GraphOptimizationLevel.ORT_ENABLE_ALL)
    """

    optimization_level: Union[int, GraphOptimizationLevel] = GraphOptimizationLevel.ORT_ENABLE_BASIC

    """
    Whether to optimize the model for GPU inference.
    The optimized graph might contain operators for GPU or CPU only when opt_level > 1.
    """
    optimize_for_gpu: bool = False

    """
    Whether to only use ONNX Runtime to optimize the model and no graph fusion in Python. 
    Graph fusion might require offline, Python scripts, to be run.
    """
    optimize_with_onnxruntime_only: bool = False


class ORTConfig(BaseConfig):
    """
    ORTConfig is the configuration class handling all the ONNX Runtime optimization and quantization parameters.

    Arg:
        opset (`int`, `optional`):
            ONNX opset version to export the model with.
        use_external_data_format (`bool`, `optional`, defaults to `False`):
            Allow exporting model >= than 2Gb.
        optimization_config (`OptimizationConfig`, `optional`, defaults to None):
            Specify a configuration to optimize ONNX Runtime model
        quantization_config (`QuantizationConfig`, `optional`, defaults to None):
            Specify a configuration to quantize ONNX Runtime model
    """

    CONFIG_NAME = "ort_config.json"
    FULL_CONFIGURATION_FILE = "ort_config.json"

    def __init__(
        self,
        opset: Optional[int] = None,
        use_external_data_format: bool = False,
        optimization_config: Optional[OptimizationConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.opset = opset
        self.use_external_data_format = use_external_data_format
        self.optimization = self.dataclass_to_dict(optimization_config)
        self.quantization = self.dataclass_to_dict(quantization_config)

    @staticmethod
    def dataclass_to_dict(config) -> dict:
        new_config = {}
        if config is None:
            return new_config
        for k, v in asdict(config).items():
            if isinstance(v, Enum):
                v = v.name
            elif isinstance(v, list):
                v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
            new_config[k] = v
        return new_config

