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
"""Configuration classes for graph optimization and quantization with ONNX Runtime."""

import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset
from packaging.version import Version, parse

from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibraterBase, CalibrationMethod, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.transformers.fusion_options import FusionOptions

from ..configuration_utils import BaseConfig
from ..utils import logging


logger = logging.get_logger(__name__)

NodeName = NodeType = str

# This value is used to indicate ORT which axis it should use to quantize an operator "per-channel"
ORT_DEFAULT_CHANNEL_FOR_OPERATORS = {"MatMul": 1}
ORT_FULLY_CONNECTED_OPERATORS = ["MatMul", "Add"]


@dataclass
class CalibrationConfig:
    """
    CalibrationConfig is the configuration class handling all the ONNX Runtime parameters related to the calibration
    step of static quantization.

    Args:
        dataset_name (`str`):
            The name of the calibration dataset.
        dataset_config_name (`str`):
            The name of the calibration dataset configuration.
        dataset_split (`str`):
            Which split of the dataset is used to perform the calibration step.
        dataset_num_samples (`int`):
            The number of samples composing the calibration dataset.
        method (`CalibrationMethod`):
            The method chosen to calculate the activations quantization parameters using the calibration dataset.
        num_bins (`Optional[int]`, defaults to `None`):
            The number of bins to use when creating the histogram when performing the calibration step using the
            Percentile or Entropy method.
        num_quantized_bins (`Optional[int]`, defaults to `None`):
            The number of quantized bins to use when performing the calibration step using the Entropy method.
        percentile (`Optional[float]`, defaults to `None`):
            The percentile to use when computing the activations quantization ranges when performing the calibration
            step using the Percentile method.
        moving_average (`Optional[bool]`, defaults to `None`):
            Whether to compute the moving average of the minimum and maximum values when performing the calibration step
            using the MinMax method.
        averaging_constant (`Optional[float]`, defaults to `None`):
            The constant smoothing factor to use when computing the moving average of the minimum and maximum values.
            Effective only when the MinMax calibration method is selected and `moving_average` is set to True.
    """

    dataset_name: str
    dataset_config_name: str
    dataset_split: str
    dataset_num_samples: int
    method: CalibrationMethod
    num_bins: Optional[int] = None
    num_quantized_bins: Optional[int] = None
    percentile: Optional[float] = None
    moving_average: Optional[bool] = None
    averaging_constant: Optional[float] = None

    def create_calibrator(
        self,
        onnx_model_path: Union[str, os.PathLike, Path],
        operators_to_quantize: Optional[List[NodeType]],
        use_external_data_format: bool = False,
        force_symmetric_range: bool = False,
        augmented_model_name: str = "augmented_model.onnx",
    ) -> CalibraterBase:
        kwargs = {
            "model": onnx_model_path,
            "op_types_to_calibrate": operators_to_quantize or [],
            "calibrate_method": self.method,
            "augmented_model_path": augmented_model_name,
        }
        if parse(ort_version) > Version("1.10.0"):
            kwargs["use_external_data_format"] = use_external_data_format
            kwargs["extra_options"] = {
                "symmetric": force_symmetric_range,
                "num_bins": self.num_bins,
                "num_quantized_bins": self.num_quantized_bins,
                "percentile": self.percentile,
                "moving_average": self.moving_average,
                "averaging_constant": self.averaging_constant,
            }
        return create_calibrator(**kwargs)


class AutoCalibrationConfig:
    @staticmethod
    def minmax(dataset: Dataset, moving_average: bool = False, averaging_constant: float = 0.01) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            moving_average (`bool`):
                Whether to compute the moving average of the minimum and maximum values.
            averaging_constant (`float`):
                The constant smoothing factor to use when computing the moving average of the minimum and maximum
                values.

        Returns:
            The calibration configuration.
        """
        if moving_average and parse(ort_version) < Version("1.11.0"):
            raise NotImplementedError(
                "MinMax calibration using the moving average method is only implemented for onnxruntime >= 1.11.0"
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
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            num_bins (`int`):
                The number of bins to use when creating the histogram.
            num_quantized_bins (`int`):
                The number of quantized bins used to find the optimal threshold when computing the activations
                quantization ranges.

        Returns:
            The calibration configuration.
        """
        if parse(ort_version) < Version("1.11.0"):
            raise NotImplementedError("Entropy calibration method is only implemented for onnxruntime >= 1.11.0")

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
    def percentiles(dataset: Dataset, num_bins: int = 2048, percentile: float = 99.999) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            num_bins (`int`):
                The number of bins to use when creating the histogram.
            percentile (`float`):
                The percentile to use when computing the activations quantization ranges.

        Returns:
            The calibration configuration.
        """
        if parse(ort_version) < Version("1.11.0"):
            raise NotImplementedError("Percentile calibration method is only implemented for onnxruntime >= 1.11.0")

        if num_bins <= 0:
            raise ValueError(f"Invalid value num_bins ({num_bins}) should be >= 1")

        if not 0 <= percentile <= 100:
            raise ValueError(f"Invalid value percentile ({percentile}) should be within  [0, 100]")

        return CalibrationConfig(
            dataset_name=dataset.info.builder_name,
            dataset_config_name=dataset.info.config_name,
            dataset_split=str(dataset.split),
            dataset_num_samples=dataset.num_rows,
            method=CalibrationMethod.Percentile,
            num_bins=num_bins,
            percentile=percentile,
        )


@dataclass
class QuantizationConfig:
    """
    QuantizationConfig is the configuration class handling all the ONNX Runtime quantization parameters.

    Args:
        is_static (`bool`):
            Whether to apply static quantization or dynamic quantization.
        format (`QuantFormat`):
            Targeted ONNX Runtime quantization representation format.
            For the Operator Oriented (QOperator) format, all the quantized operators have their own ONNX definitions.
            For the Tensor Oriented (QDQ) format, the model is quantized by inserting QuantizeLinear / DeQuantizeLinear
            operators.
        mode (`QuantizationMode`, defaults to `QuantizationMode.QLinearOps`):
            Targeted ONNX Runtime quantization mode, default is QLinearOps to match QDQ format.
            When targeting dynamic quantization mode, the default value is `QuantizationMode.IntegerOps` whereas the
            default value for static quantization mode is `QuantizationMode.QLinearOps`.
        activations_dtype (`QuantType`, defaults to `QuantType.QUInt8`):
            The quantization data types to use for the activations.
        activations_symmetric (`bool`, defaults to `False`):
            Whether to apply symmetric quantization on the activations.
        weights_dtype (`QuantType`, defaults to `QuantType.QInt8`):
            The quantization data types to use for the weights.
        weights_symmetric (`bool`, defaults to `True`):
            Whether to apply symmetric quantization on the weights.
        per_channel (`bool`, defaults to `False`):
            Whether we should quantize per-channel (also known as "per-row"). Enabling this can increase overall
            accuracy while making the quantized model heavier.
        reduce_range (`bool`, defaults to `False`):
            Whether to use reduce-range 7-bits integers instead of 8-bits integers.
        nodes_to_quantize (`list`):
            List of the nodes names to quantize.
        nodes_to_exclude (`list`):
            List of the nodes names to exclude when applying quantization.
        operators_to_quantize (`list`, defaults to `["MatMul", "Add"]`):
            List of the operators types to quantize.
        qdq_add_pair_to_weight (`bool`, defaults to `False`):
            By default, floating-point weights are quantized and feed to solely inserted DeQuantizeLinear node.
            If set to True, the floating-point weights will remain and both QuantizeLinear / DeQuantizeLinear nodes
            will be inserted.
        qdq_dedicated_pair (`bool`, defaults to `False`):
            When inserting QDQ pair, multiple nodes can share a single QDQ pair as their inputs. If True, it will
            create an identical and dedicated QDQ pair for each node.
        qdq_op_type_per_channel_support_to_axis (`Dict[str, int]`):
            Set the channel axis for a specific operator type. Effective only when per channel quantization is
            supported and `per_channel` is set to True.
    """

    is_static: bool
    format: QuantFormat
    mode: QuantizationMode = QuantizationMode.QLinearOps
    activations_dtype: QuantType = QuantType.QUInt8
    activations_symmetric: bool = False
    weights_dtype: QuantType = QuantType.QInt8
    weights_symmetric: bool = True
    per_channel: bool = False
    reduce_range: bool = False
    nodes_to_quantize: List[NodeName] = field(default_factory=list)
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

    @property
    def use_symmetric_calibration(self) -> bool:
        return self.activations_symmetric and self.weights_symmetric

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
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[NodeName] = ORT_FULLY_CONNECTED_OPERATORS,
    ):
        """
        Creates a [`~onnxruntime.QuantizationConfig`] fit for ARM64.

        Args:
            is_static (`bool`):
                Boolean flag to indicate whether we target static or dynamic quantization.
            use_symmetric_activations (`bool`, defaults to `False`):
                Whether to use symmetric quantization for activations.
            use_symmetric_weights (`bool`, defaults to `True`):
                Whether to use symmetric quantization for weights.
            per_channel (`bool`, defaults to `True`):
                Whether we should quantize per-channel (also known as "per-row"). Enabling this can
                increase overall accuracy while making the quantized model heavier.
            nodes_to_quantize (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to quantize. If `None`, all nodes being operators from `operators_to_quantize` will be quantized.
            nodes_to_exclude (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to exclude from quantization.
            operators_to_quantize (`List[NodeName]`, defaults to `["MatMul", "Add"]`):
                Type of nodes to perform quantization on.
        """
        format, mode = default_quantization_parameters(is_static)

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
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        reduce_range: bool = False,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[NodeName] = ORT_FULLY_CONNECTED_OPERATORS,
    ) -> QuantizationConfig:
        """
        Creates a [`~onnxruntime.QuantizationConfig`] fit for CPU with AVX2 instruction set.

        Args:
            is_static (`bool`):
                Boolean flag to indicate whether we target static or dynamic quantization.
            use_symmetric_activations (`bool`, defaults to `False`):
                Whether to use symmetric quantization for activations.
            use_symmetric_weights (`bool`, defaults to `True`):
                Whether to use symmetric quantization for weights.
            per_channel (`bool`, defaults to `True`):
                Whether we should quantize per-channel (also known as "per-row"). Enabling this can
                increase overall accuracy while making the quantized model heavier.
            reduce_range (`bool`, defaults to `False`):
                Indicate whether to use 8-bits integers (False) or reduce-range 7-bits integers (True).
                As a baseline, it is always recommended testing with full range (reduce_range = False) and then, if
                accuracy drop is significant, to try with reduced range (reduce_range = True).
                Intel's CPUs using AVX512 (non VNNI) can suffer from saturation issue when invoking
                the VPMADDUBSW instruction. To counter this, one should use 7-bits rather than 8-bits integers.
            nodes_to_quantize (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to quantize. If `None`, all nodes being operators from `operators_to_quantize` will be quantized.
            nodes_to_exclude (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to exclude from quantization.
            operators_to_quantize (`List[NodeName]`, defaults to `["MatMul", "Add"]`):
                Type of nodes to perform quantization on.
        """
        format, mode = default_quantization_parameters(is_static)

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
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        reduce_range: bool = False,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[NodeName] = ORT_FULLY_CONNECTED_OPERATORS,
    ) -> QuantizationConfig:
        """
        Creates a [`~onnxruntime.QuantizationConfig`] fit for CPU with AVX512 instruction set.

        Args:
            is_static (`bool`):
                Boolean flag to indicate whether we target static or dynamic quantization.
            use_symmetric_activations (`bool`, defaults to `False`):
                Whether to use symmetric quantization for activations.
            use_symmetric_weights (`bool`, defaults to `True`):
                Whether to use symmetric quantization for weights.
            per_channel (`bool`, defaults to `True`):
                Whether we should quantize per-channel (also known as "per-row"). Enabling this can
                increase overall accuracy while making the quantized model heavier.
            reduce_range (`bool`, defaults to `False`):
                Indicate whether to use 8-bits integers (False) or reduce-range 7-bits integers (True).
                As a baseline, it is always recommended testing with full range (reduce_range = False) and then, if
                accuracy drop is significant, to try with reduced range (reduce_range = True).
                Intel's CPUs using AVX512 (non VNNI) can suffer from saturation issue when invoking
                the VPMADDUBSW instruction. To counter this, one should use 7-bits rather than 8-bits integers.
            nodes_to_quantize (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to quantize. If `None`, all nodes being operators from `operators_to_quantize` will be quantized.
            nodes_to_exclude (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to exclude from quantization.
            operators_to_quantize (`List[NodeName]`, defaults to `["MatMul", "Add"]`):
                Type of nodes to perform quantization on.
        """
        format, mode = default_quantization_parameters(is_static)

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
        use_symmetric_activations: bool = False,
        use_symmetric_weights: bool = True,
        per_channel: bool = True,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[NodeName] = ORT_FULLY_CONNECTED_OPERATORS,
    ) -> QuantizationConfig:
        """
        Creates a [`~onnxruntime.QuantizationConfig`] fit for CPU with AVX512-VNNI instruction set.

        When targeting Intel AVX512-VNNI CPU underlying execution engine leverage the CPU instruction VPDPBUSD to
        compute  \\i32 += i8(w) * u8(x)\\ within a single instruction.

        AVX512-VNNI (AVX512 Vector Neural Network Instruction)
        is an x86 extension Instruction set and is a part of the AVX-512 ISA.

        AVX512 VNNI is designed to accelerate convolutional neural network for INT8 inference.

        Args:
            is_static (`bool`):
                Boolean flag to indicate whether we target static or dynamic quantization.
            use_symmetric_activations (`bool`, defaults to `False`):
                Whether to use symmetric quantization for activations.
            use_symmetric_weights (`bool`, defaults to `True`):
                Whether to use symmetric quantization for weights.
            per_channel (`bool`, defaults to `True`):
                Whether we should quantize per-channel (also known as "per-row"). Enabling this can
                increase overall accuracy while making the quantized model heavier.
            nodes_to_quantize (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to quantize. If `None`, all nodes being operators from `operators_to_quantize` will be quantized.
            nodes_to_exclude (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to exclude from quantization.
            operators_to_quantize (`List[NodeName]`, defaults to `["MatMul", "Add"]`):
                Type of nodes to perform quantization on.
        """
        format, mode = default_quantization_parameters(is_static)

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
        per_channel: bool = True,
        nodes_to_quantize: Optional[List[NodeName]] = None,
        nodes_to_exclude: Optional[List[NodeName]] = None,
        operators_to_quantize: List[NodeName] = ORT_FULLY_CONNECTED_OPERATORS,
    ) -> QuantizationConfig:
        """
        Creates a [`~onnxruntime.QuantizationConfig`] fit for TensorRT static quantization, targetting NVIDIA GPUs.

        Args:
            per_channel (`bool`, defaults to `True`):
                Whether we should quantize per-channel (also known as "per-row"). Enabling this can
                increase overall accuracy while making the quantized model heavier.
            nodes_to_quantize (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to quantize. If `None`, all nodes being operators from `operators_to_quantize` will be quantized.
            nodes_to_exclude (`Optional[List[NodeName]]`, defaults to `None`):
                Specific nodes to exclude from quantization.
            operators_to_quantize (`List[NodeName]`, defaults to `["MatMul", "Add"]`):
                Type of nodes to perform quantization on.
        """
        format, mode = default_quantization_parameters(is_static=True)

        return QuantizationConfig(
            is_static=True,
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
            # `qdq_dedicated_pair=True` argument is required by TensorRT, since it expects a single node after each
            # `QuantizeLinear` + `DequantizeLinear` (QDQ) pair.
            qdq_add_pair_to_weight=True,
            # `qdq_dedicated_pair=True` is required because TensorRT expects QDQ pairs on weights, not only DequantizeLinear
            qdq_dedicated_pair=True,
        )


@dataclass
class OptimizationConfig:
    """
    OptimizationConfig is the configuration class handling all the ONNX Runtime optimization parameters.
    There are two stacks of optimizations:
        1. The ONNX Runtime general-purpose optimization tool: it can work on any ONNX model.
        2. The ONNX Runtime transformers optimization tool: it can only work on a subset of transformers models.

    Attributes:
        optimization_level (`int`, defaults to 1):
            Optimization level performed by ONNX Runtime of the loaded graph.
            Supported optimization level are 0, 1, 2 and 99.
                - 0: will disable all optimizations
                - 1: will enable basic optimizations
                - 2: will enable basic and extended optimizations, including complex node fusions applied to the nodes
                assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent
                - 99: will enable all available optimizations including layout optimizations
        optimize_for_gpu (`bool`, defaults to `False`):
            Whether to optimize the model for GPU inference.
            The optimized graph might contain operators for GPU or CPU only when `optimization_level` > 1.
        fp16 (`bool`, defaults to `False`):
            Whether all weights and nodes should be converted from float32 to float16.
        enable_transformers_specific_optimizations (`bool`, defaults to `True`):
            Whether to only use `transformers` specific optimizations on top of ONNX Runtime general optimizations.
        disable_gelu_fusion (`bool`, defaults to `False`):
            Whether to disable the Gelu fusion.
        disable_layer_norm_fusion (`bool`, defaults to `False`):
            Whether to disable Layer Normalization fusion.
        disable_attention_fusion (`bool`, defaults to `False`):
            Whether to disable Attention fusion.
        disable_skip_layer_norm_fusion (`bool`, defaults to `False`):
            Whether to disable SkipLayerNormalization fusion.
        disable_bias_skip_layer_norm_fusion (`bool`, defaults to `False`):
            Whether to disable Add Bias and SkipLayerNormalization fusion.
        disable_bias_gelu_fusion (`bool`, defaults to `False`):
            Whether to disable Add Bias and Gelu / FastGelu fusion.
        disable_embed_layer_norm_fusion (`bool`, defaults to `True`):
            Whether to disable EmbedLayerNormalization fusion.
            The default value is set to `True` since this fusion is incompatible with ONNX Runtime quantization.
        enable_gelu_approximation (`bool`, defaults to `False`):
            Whether to enable Gelu / BiasGelu to FastGelu conversion.
            The default value is set to `False` since this approximation might slightly impact the model's accuracy.
        use_mask_index (`bool`, defaults to `False`):
            Whether to use mask index instead of raw attention mask in the attention operator.
        no_attention_mask (`bool`, defaults to `False`):
            Whether to not use attention masks. Only works for bert model type.
        disable_embed_layer_norm (`bool`, defaults to `True`):
            Whether to disable EmbedLayerNormalization fusion.
            The default value is set to `True` since this fusion is incompatible with ONNX Runtime quantization
        disable_shape_inference (`bool`, defaults to `False`):
            Whether to disable symbolic shape inference.
            The default value is set to `False` but symbolic shape inference might cause issues sometimes.
        use_multi_head_attention (`bool`, defaults to `False`):
            Experimental argument. Use MultiHeadAttention instead of Attention operator, which has merged weights for Q/K/V projection,
            which might be faster in some cases since 3 MatMul is merged into one."
            "Note that MultiHeadAttention might be slower than Attention since MatMul of input projection is excluded. "
            "MultiHeadAttention has only CUDA implementation so the model can only run with CUDAExecutionProvider.
        enable_gemm_fast_gelu (`bool`, defaults to `True`):
            Enable GemmfastGelu fusion.
        use_raw_attention_mask (`bool`, defaults to `False`):
            Use raw attention mask. Use this option if your input is not right-side padding. This might deactivate fused attention and get worse performance.
        disable_group_norm (`bool`, defaults to `False`):
            Do not fuse GroupNorm. Only works for model_type=unet.
        disable_packed_kv (`bool`, defaults to `False`):
            Do not use packed kv in cross attention. Only works for model_type=unet.
    """

    optimization_level: int = 1
    optimize_for_gpu: bool = False

    fp16: bool = False

    optimize_with_onnxruntime_only: Optional[bool] = None
    enable_transformers_specific_optimizations: bool = True

    disable_gelu: Optional[bool] = None
    disable_gelu_fusion: bool = False

    disable_layer_norm: Optional[bool] = None
    disable_layer_norm_fusion: bool = False

    disable_attention: Optional[bool] = None
    disable_attention_fusion: bool = False

    disable_skip_layer_norm: Optional[bool] = None
    disable_skip_layer_norm_fusion: bool = False

    disable_bias_skip_layer_norm: Optional[bool] = None
    disable_bias_skip_layer_norm_fusion: bool = False

    disable_bias_gelu: Optional[bool] = None
    disable_bias_gelu_fusion: bool = False

    disable_embed_layer_norm: Optional[bool] = None
    disable_embed_layer_norm_fusion: bool = True

    enable_gelu_approximation: bool = False
    use_mask_index: bool = False
    no_attention_mask: bool = False
    disable_embed_layer_norm: bool = True
    disable_shape_inference: bool = False

    # ONNX Runtime 1.14.0 arguments
    use_multi_head_attention = False
    enable_gemm_fast_gelu_fusion = False
    use_raw_attention_mask = False
    disable_group_norm_fusion = True
    disable_packed_kv = True

    def __post_init__(self):
        def deprecate_renamed_attribute(old_name, new_name, mapping_func=None):
            if getattr(self, old_name, None) is not None:
                if mapping_func is None:

                    def identity(x):
                        return x

                    mapping_func = identity
                setattr(self, new_name, mapping_func(getattr(self, old_name)))
                warnings.warn(
                    f"{old_name} will be deprecated soon, use {new_name} instead, {new_name} is set to "
                    f"{getattr(self, new_name)}.",
                    FutureWarning,
                )

        deprecate_renamed_attribute(
            "optimize_with_onnxruntime_only",
            "enable_transformers_specific_optimizations",
            mapping_func=lambda x: not x,
        )

        deprecate_renamed_attribute("disable_gelu", "disable_bias_gelu_fusion")
        deprecate_renamed_attribute("disable_layer_norm", "disable_layer_norm_fusion")
        deprecate_renamed_attribute("disable_attention", "disable_attention_fusion")
        deprecate_renamed_attribute("disable_skip_layer_norm", "disable_skip_layer_norm_fusion")
        deprecate_renamed_attribute("disable_bias_skip_layer_norm", "disable_bias_skip_layer_norm_fusion")
        deprecate_renamed_attribute("disable_bias_gelu", "disable_bias_gelu_fusion")
        deprecate_renamed_attribute("disable_embed_layer_norm", "disable_embed_layer_norm_fusion")

    def create_fusion_options(self, model_type: str) -> FusionOptions:
        class Box:
            pass

        args = Box()
        args.model_type = model_type
        attribute_map = {
            "disable_gelu_fusion": "disable_gelu",
            "disable_layer_norm_fusion": "disable_layer_norm",
            "disable_attention_fusion": "disable_attention",
            "disable_skip_layer_norm_fusion": "disable_skip_layer_norm",
            "disable_bias_skip_layer_norm_fusion": "disable_bias_skip_layer_norm",
            "disable_bias_gelu_fusion": "disable_bias_gelu",
            "disable_embed_layer_norm_fusion": "disable_embed_layer_norm",
            "disable_group_norm_fusion": "disable_group_norm",
            "disable_packed_kv": "disable_packed_kv",
            "use_raw_attention_mask": "use_raw_attention_mask",
            "enable_gemm_fast_gelu_fusion": "enable_gemm_fast_gelu",
            "use_multi_head_attention": "use_multi_head_attention",
        }
        for attr_name, fusion_attr_name in attribute_map.items():
            setattr(args, fusion_attr_name, getattr(self, attr_name))

        for attr, value in self.__dict__.items():
            if hasattr(args, attr):
                continue
            setattr(args, attr, value)

        return FusionOptions.parse(args)


class AutoOptimizationConfig:
    """
    Factory to create common `OptimizationConfig`.
    """

    _LEVELS = {
        "O1": {
            "optimization_level": 1,
            "enable_transformers_specific_optimizations": False,
        },
        "O2": {
            "optimization_level": 2,
            "enable_transformers_specific_optimizations": True,
        },
        "O3": {
            "optimization_level": 2,
            "enable_transformers_specific_optimizations": True,
            "enable_gelu_approximation": True,
        },
        "O4": {
            "optimization_level": 2,
            "enable_transformers_specific_optimizations": True,
            "enable_gelu_approximation": True,
            "fp16": True,
        },
    }

    @classmethod
    def with_optimization_level(cls, optimization_level: str, for_gpu: bool = False, **kwargs) -> OptimizationConfig:
        """
        Creates an [`~OptimizationConfig`] with pre-defined arguments according to an optimization level.

        Args:
            optimization_level (`str`):
                The optimization level, the following values are allowed:
                - O1: Basic general optimizations
                - O2: Basic and extended general optimizations, transformers-specific fusions.
                - O3: Same as O2 with Fast Gelu approximation.
                - O4: Same as O3 with mixed precision.
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the requested optimization level.
        """
        if optimization_level not in cls._LEVELS:
            raise ValueError(
                f"optimization_level must be in {', '.join(cls._LEVELS.keys())}, got {optimization_level}"
            )

        if optimization_level == "O4":
            if for_gpu is False:
                logger.warning("Overridding for_gpu=False to for_gpu=True as half precision is available only on GPU.")
            for_gpu = True

        return OptimizationConfig(optimize_for_gpu=for_gpu, **cls._LEVELS[optimization_level], **kwargs)

    @classmethod
    def O1(cls, for_gpu: bool = False, **kwargs) -> OptimizationConfig:
        """
        Creates an O1 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O1 optimization level.
        """
        return cls.with_optimization_level("O1", for_gpu=for_gpu, **kwargs)

    @classmethod
    def O2(cls, for_gpu: bool = False, **kwargs) -> OptimizationConfig:
        """
        Creates an O2 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O2 optimization level.
        """
        return cls.with_optimization_level("O2", for_gpu=for_gpu, **kwargs)

    @classmethod
    def O3(cls, for_gpu: bool = False, **kwargs) -> OptimizationConfig:
        """
        Creates an O3 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O3 optimization level.
        """
        return cls.with_optimization_level("O3", for_gpu=for_gpu, **kwargs)

    @classmethod
    def O4(cls, for_gpu: bool = True, **kwargs) -> OptimizationConfig:
        """
        Creates an O4 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O4 optimization level.
        """
        return cls.with_optimization_level("O4", for_gpu=for_gpu, **kwargs)


class ORTConfig(BaseConfig):
    """
    ORTConfig is the configuration class handling all the ONNX Runtime parameters related to the ONNX IR model export,
    optimization and quantization parameters.

    Attributes:
        opset (`Optional[int]`, defaults to `None`):
            ONNX opset version to export the model with.
        use_external_data_format (`bool`, defaults to `False`):
            Allow exporting model >= than 2Gb.
        one_external_file (`bool`, defaults to `True`):
            When `use_external_data_format=True`, whether to save all tensors to one external file.
            If false, save each tensor to a file named with the tensor name.
            (Can not be set to `False` for the quantization)
        optimization (`Optional[OptimizationConfig]`, defaults to `None`):
            Specify a configuration to optimize ONNX Runtime model
        quantization (`Optional[QuantizationConfig]`, defaults to `None`):
            Specify a configuration to quantize ONNX Runtime model
    """

    CONFIG_NAME = "ort_config.json"
    FULL_CONFIGURATION_FILE = "ort_config.json"

    def __init__(
        self,
        opset: Optional[int] = None,
        use_external_data_format: bool = False,
        one_external_file: bool = True,
        optimization: Optional[OptimizationConfig] = None,
        quantization: Optional[QuantizationConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.opset = opset
        self.use_external_data_format = use_external_data_format
        self.one_external_file = one_external_file
        self.optimization = self.dataclass_to_dict(optimization)
        self.quantization = self.dataclass_to_dict(quantization)
        self.optimum_version = kwargs.pop("optimum_version", None)

    @staticmethod
    def dataclass_to_dict(config) -> dict:
        new_config = {}
        if config is None:
            return new_config
        if isinstance(config, dict):
            return config
        for k, v in asdict(config).items():
            if isinstance(v, Enum):
                v = v.name
            elif isinstance(v, list):
                v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
            new_config[k] = v
        return new_config
