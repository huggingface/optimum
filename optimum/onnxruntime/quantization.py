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

import logging
import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import onnx
from datasets import Dataset, load_dataset
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

from onnxruntime.quantization import (
    CalibrationDataReader,
    QDQQuantizer,
    QuantFormat,
    QuantizationMode,
)

from optimum.onnxruntime.configuration import NodeName, QuantizationConfig, CalibrationConfig, NodeType

LOGGER = logging.getLogger(__name__)


class ORTCalibrationDataReader(CalibrationDataReader):
    """

    """

    def __init__(self, dataset: Dataset, batch_size: int = 1):
        if dataset is None:
            raise ValueError("Provided dataset is None.")

        if batch_size <= 0:
            raise ValueError(f"Provided batch_size should be >= 1 (got: {batch_size}).")

        self.dataset = dataset
        self.batch_size = batch_size

        self._dataset_iter = iter(self.dataset)

    def get_next(self):
        featurized_samples = None
        try:
            if self.batch_size == 1:
                featurized_samples = {key: [value] for key, value in next(self._dataset_iter).items()}
            else:
                featurized_samples = defaultdict(list)
                for _ in range(self.batch_size):
                    sample = next(self._dataset_iter)

                    for name, value in sample.items():
                        featurized_samples[name] += [value]

        except StopIteration:
            pass
        finally:
            if featurized_samples is not None and len(featurized_samples) > 0:
                return featurized_samples
            else:
                return None


class ORTQuantizer(ABC):
    """

    """

    @staticmethod
    def from_pretrained(
        model_name_or_path: Union[str, os.PathLike],
        feature: str,
        opset: Optional[int] = None
    ) -> 'ORTQuantizer':
        """

        :param model_name_or_path:
        :param feature:
        :param opset:
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        model = model_class.from_pretrained(model_name_or_path)

        return ORTQuantizer(tokenizer, model, feature, opset)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        feature: str = "default",
        opset: Optional[int] = None
    ):
        """

        :param tokenizer:
        :param model:
        :param feature:
        :param opset:
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model

        self.feature = feature

        self._model_type, onnx_config_factory = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        self._onnx_config = onnx_config_factory(self.model.config)
        self.opset = self._onnx_config.default_onnx_opset if opset is None else opset

        self._calibrator = None

    def fit(
        self,
        dataset: Dataset,
        calibration_config: CalibrationConfig,
        onnx_model_path: Union[str, os.PathLike, Path],
        onnx_augmented_model_name: str = "augmented_model.onnx",
        batch_size: int = 1,
        use_external_data_format: bool = False,
        operators_to_quantize: Optional[List[NodeType]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """

        :param dataset
        :param calibration_config:
        :param onnx_model_path:
        :param onnx_augmented_model_name
        :param batch_size:
        :param use_external_data_format:
        :param operators_to_quantize:
        :return:
        """
        # If a dataset is provided, then we are in a static quantization mode
        LOGGER.info(
            f"Using static quantization schema ("
            f"dataset: {calibration_config.dataset_name}, method: {calibration_config.method}"
            f")"
        )

        self.partial_fit(
            dataset,
            calibration_config,
            onnx_model_path,
            onnx_augmented_model_name,
            batch_size,
            use_external_data_format,
            operators_to_quantize
        )
        return self.compute_ranges()

    def partial_fit(
        self,
        dataset: Dataset,
        calibration_config: CalibrationConfig,
        onnx_model_path: Union[str, os.PathLike],
        onnx_augmented_model_name: str = "augmented_model.onnx",
        batch_size: int = 1,
        use_external_data_format: bool = False,
        operators_to_quantize: Optional[List[NodeType]] = None
    ):
        """

        :param dataset
        :param batch_size
        :param calibration_config:
        :param onnx_model_path:
        :param onnx_augmented_model_name
        :param use_external_data_format:
        :param operators_to_quantize:
        :return:
        """
        if not isinstance(onnx_model_path, Path):
            onnx_model_path = Path(onnx_model_path)

        # Export the model to ONNX IR
        if not onnx_model_path.exists():
            export(self.tokenizer, self.model, self._onnx_config, self.opset, onnx_model_path)

            LOGGER.info(f"Exported model to ONNX at: {onnx_model_path.as_posix()}")

        # If no calibrator, then create one
        if calibration_config.method is not None:
            LOGGER.info(f"Creating calibrator: {calibration_config.method}({calibration_config})")
            self._calibrator = calibration_config.create_calibrator(
                onnx_model_path=onnx_model_path.as_posix(),
                use_external_data_format=use_external_data_format,
                augmented_model_name=onnx_augmented_model_name,
                operators_to_quantize=operators_to_quantize
            )

        LOGGER.info("Collecting tensors statistics...")
        self._calibrator.collect_data(
            ORTCalibrationDataReader(dataset, batch_size)
        )

    def compute_ranges(self) -> Dict[NodeName, Tuple[float, float]]:
        """

        :return:
        """
        if self._calibrator is None:
            raise ValueError(
                "Calibrator is None, please call `partial_fit` or `fit` method at least ones to compute ranges."
            )

        LOGGER.info("Computing calibration ranges")
        return self._calibrator.compute_range()

    def export(
        self,
        onnx_model_path: Union[str, os.PathLike],
        onnx_quantized_model_output_path: Union[str, os.PathLike],
        calibration_tensors_range: Dict[NodeName, Tuple[float, float]],
        quantization_config: QuantizationConfig,
        use_external_data_format: bool = False
    ) -> Path:
        """

        :param onnx_model_path:
        :param onnx_quantized_model_output_path:
        :param calibration_tensors_range:
        :param quantization_config:
        :param use_external_data_format:
        :return:
        """
        is_static = calibration_tensors_range is not None
        use_qdq = is_static and quantization_config.format == QuantFormat.QDQ

        if not is_static and quantization_config.mode != QuantizationMode.IntegerOps:
            LOGGER.warning(
                f"ONNX Runtime dynamic quantization mode should be QuantizationMode.IntegerOps "
                f"(got: {quantization_config.mode})."
            )

        LOGGER.info(
            f"Creating {'dynamic' if is_static else 'static'} quantizer: "
            f"{quantization_config}"
        )

        onnx_model = onnx.load(onnx_model_path)
        quantizer_factory = QDQQuantizer if use_qdq else ONNXQuantizer
        quantizer = quantizer_factory(
            model=onnx_model,
            static=is_static,
            per_channel=quantization_config.per_channel,
            mode=quantization_config.mode,
            weight_qType=quantization_config.weights_dtype,
            input_qType=quantization_config.activations_dtype,
            tensors_range=calibration_tensors_range,
            reduce_range=quantization_config.reduce_range,
            nodes_to_quantize=quantization_config.nodes_to_quantize,
            nodes_to_exclude=quantization_config.nodes_to_exclude,
            op_types_to_quantize=quantization_config.operators_to_quantize
        )

        LOGGER.info("Quantizing model...")
        quantizer.quantize_model()

        LOGGER.info(f"Saving quantized model at: {onnx_model_path} (external data format: {use_external_data_format})")
        quantizer.model.save_model_to_file(onnx_quantized_model_output_path, use_external_data_format)

        return Path(onnx_quantized_model_output_path)

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        seed: int = 2016
    ) -> Dataset:
        """
        Returns the calibration :class:`~datasets.arrow_dataset.Dataset` to use for the post-training static
        quantization calibration step.
        """
        if dataset_name is None:
            raise ValueError(
                "ORTQuantizer: Static quantization calibration step requires a dataset_name if no calib_dataset is "
                "provided."
            )
        if preprocess_function is None:
            raise ValueError(
                "ORTQuantizer: Processing function to apply after loading the dataset used for static quantization "
                "calibration step was not provided."
            )

        calib_dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
        )

        if num_samples is not None:
            num_samples = min(num_samples, len(calib_dataset))
            calib_dataset = calib_dataset.shuffle(seed=seed).select(range(num_samples))

        if preprocess_function is not None:
            processed_calib_dataset = calib_dataset.map(preprocess_function, batched=preprocess_batch)
        else:
            processed_calib_dataset = calib_dataset

        ignored_columns = list(set(processed_calib_dataset.column_names) - set(self._onnx_config.inputs.keys()))
        return processed_calib_dataset.remove_columns(ignored_columns)
