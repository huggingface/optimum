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
from typing import Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

import onnx
from onnxruntime.quantization import CalibrationDataReader, QDQQuantizer, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from optimum.onnxruntime import ORTQuantizableOperator
from optimum.onnxruntime.configuration import CalibrationConfig, NodeName, NodeType, QuantizationConfig
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor


LOGGER = logging.getLogger(__name__)


class ORTCalibrationDataReader(CalibrationDataReader):
    """ """

    __slots__ = ["batch_size", "dataset", "_dataset_iter"]

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
    Handles the ONNX Runtime quantization process for models shared on huggingface.co/models.
    """

    @staticmethod
    def from_pretrained(
        model_name_or_path: Union[str, os.PathLike], feature: str, opset: Optional[int] = None
    ) -> "ORTQuantizer":
        """
        Instantiate a `ORTQuantizer` from a pretrained pytorch model and tokenizer.

        Args:
            model_name_or_path (`Union[str, os.PathLike]`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            feature (`str`):
                Feature to use when exporting the model.
            opset (`int`, *optional*):
                ONNX opset version to export the model with.

        Returns:
            An instance of `ORTQuantizer`.
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
        opset: Optional[int] = None,
    ):
        """
        Args:
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer used to preprocess the data.
            model (`PreTrainedModel`):
                The model to optimize.
            feature (`str`, defaults to `"default"`):
                Feature to use when exporting the model.
            opset (`int`, *optional*):
                ONNX opset version to export the model with.
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
        operators_to_quantize: Optional[List[NodeType]] = None,
        batch_size: int = 1,
        use_external_data_format: bool = False,
        use_gpu: bool = False,
        force_symmetric_range: bool = False,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Perform the calibration step and collect the quantization ranges.

        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            calibration_config (`CalibrationConfig`):
                The configuration containing the parameters related to the calibration step.
            onnx_model_path (`Union[str, os.PathLike]`):
                The path used to save the model exported to an ONNX Intermediate Representation (IR).
            onnx_augmented_model_name (`Union[str, os.PathLike]`):
                The path used to save the augmented model used to collect the quantization ranges.
            operators_to_quantize (`list`, *optional*):
                List of the operators types to quantize.
            batch_size (`int`, defaults to 1):
                The batch size to use when collecting the quantization ranges values.
            use_external_data_format (`bool`, defaults to `False`):
                Whether uto se external data format to store model which size is >= 2Gb.
            use_gpu (`bool`, defaults to `False`):
                Whether to use the GPU when collecting the quantization ranges values.
            force_symmetric_range (`bool`, defaults to `False`):
                Whether to make the quantization ranges symmetric.

        Returns:
            The dictionary mapping the nodes name to their quantization ranges.
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
            operators_to_quantize,
            batch_size,
            use_external_data_format,
            use_gpu,
            force_symmetric_range,
        )
        return self.compute_ranges()

    def partial_fit(
        self,
        dataset: Dataset,
        calibration_config: CalibrationConfig,
        onnx_model_path: Union[str, os.PathLike],
        onnx_augmented_model_name: str = "augmented_model.onnx",
        operators_to_quantize: Optional[List[NodeType]] = None,
        batch_size: int = 1,
        use_external_data_format: bool = False,
        use_gpu: bool = False,
        force_symmetric_range: bool = False,
    ):
        """
        Perform the calibration step and collect the quantization ranges.

        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            calibration_config (`CalibrationConfig`):
                The configuration containing the parameters related to the calibration step.
            onnx_model_path (`Union[str, os.PathLike]`):
                The path used to save the model exported to an ONNX Intermediate Representation (IR).
            onnx_augmented_model_name (`Union[str, os.PathLike]`):
                The path used to save the augmented model used to collect the quantization ranges.
            operators_to_quantize (`list`, *optional*):
                List of the operators types to quantize.
            batch_size (`int`, defaults to 1):
                The batch size to use when collecting the quantization ranges values.
            use_external_data_format (`bool`, defaults to `False`):
                Whether uto se external data format to store model which size is >= 2Gb.
            use_gpu (`bool`, defaults to `False`):
                Whether to use the GPU when collecting the quantization ranges values.
            force_symmetric_range (`bool`, defaults to `False`):
                Whether to make the quantization ranges symmetric.

        Returns:
            The dictionary mapping the nodes name to their quantization ranges.
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
                operators_to_quantize=operators_to_quantize,
                force_symmetric_range=force_symmetric_range,
            )

        if use_gpu:
            self._calibrator.set_execution_providers(execution_providers=["CUDAExecutionProvider"])

        LOGGER.info("Collecting tensors statistics...")
        reader = ORTCalibrationDataReader(dataset, batch_size)
        self._calibrator.collect_data(reader)

    def compute_ranges(self) -> Dict[NodeName, Tuple[float, float]]:
        """
        Returns:
            The dictionary mapping the nodes name to their quantization ranges.
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
        quantization_config: QuantizationConfig,
        calibration_tensors_range: Optional[Dict[NodeName, Tuple[float, float]]] = None,
        use_external_data_format: bool = False,
        preprocessor: Optional[QuantizationPreprocessor] = None,
    ) -> Path:
        """
        Quantize a model given the optimization specifications defined in `quantization_config`.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                The path used to save the model exported to an ONNX Intermediate Representation (IR).
            onnx_quantized_model_output_path (`Union[str, os.PathLike]`):
                The path used to save the quantized model exported to an ONNX Intermediate Representation (IR).
            quantization_config (`QuantizationConfig`):
                The configuration containing the parameters related to quantization.
            calibration_tensors_range (`Dict[NodeName, Tuple[float, float]]`, *optional*):
                The dictionary mapping the nodes name to their quantization ranges, used and required only when applying
                static quantization.
            use_external_data_format (`bool`, defaults to `False`):
                Whether uto se external data format to store model which size is >= 2Gb.
            preprocessor (`QuantizationPreprocessor`, *optional*):
                The preprocessor to use to collect the nodes to include or exclude from quantization.

        Returns:
            The path of the resulting quantized model.
        """
        if not isinstance(onnx_model_path, Path):
            onnx_model_path = Path(onnx_model_path)

        # Export the model if it has not already been exported to ONNX IR (useful for dynamic quantization)
        if not onnx_model_path.exists():
            export(self.tokenizer, self.model, self._onnx_config, self.opset, onnx_model_path)

        use_qdq = quantization_config.is_static and quantization_config.format == QuantFormat.QDQ

        if not quantization_config.is_static:
            if quantization_config.mode != QuantizationMode.IntegerOps:
                LOGGER.warning(
                    f"ONNX Runtime dynamic quantization mode should be QuantizationMode.IntegerOps "
                    f"(got: {quantization_config.mode})."
                )
            if quantization_config.activations_dtype != QuantType.QUInt8:
                LOGGER.warning(
                    f"ONNX Runtime dynamic quantization activations data type should be QuantType.QUInt8 "
                    f"(got: {quantization_config.activations_dtype})."
                )

        LOGGER.info(
            f"Creating {'dynamic' if quantization_config.is_static else 'static'} quantizer: {quantization_config}"
        )

        if preprocessor is not None:
            LOGGER.info("Preprocessor detected, collecting nodes to include/exclude")
            nodes_to_quantize, nodes_to_exclude = preprocessor.collect()

            nodes_to_quantize.update(quantization_config.nodes_to_quantize)
            nodes_to_exclude.update(quantization_config.nodes_to_exclude)

            quantization_config.nodes_to_quantize = list(nodes_to_quantize)
            quantization_config.nodes_to_exclude = list(nodes_to_exclude)

        onnx_model = onnx.load(onnx_model_path)
        quantizer_factory = QDQQuantizer if use_qdq else ONNXQuantizer
        quantizer = quantizer_factory(
            model=onnx_model,
            static=quantization_config.is_static,
            per_channel=quantization_config.per_channel,
            mode=quantization_config.mode,
            weight_qType=quantization_config.weights_dtype,
            input_qType=quantization_config.activations_dtype,
            tensors_range=calibration_tensors_range,
            reduce_range=quantization_config.reduce_range,
            nodes_to_quantize=quantization_config.nodes_to_quantize,
            nodes_to_exclude=quantization_config.nodes_to_exclude,
            op_types_to_quantize=[
                operator.value if isinstance(operator, ORTQuantizableOperator) else operator
                for operator in quantization_config.operators_to_quantize
            ],
            extra_options={
                "WeightSymmetric": quantization_config.weights_symmetric,
                "ActivationSymmetric": quantization_config.activations_symmetric,
                "EnableSubgraph": False,
                "ForceSymmetric": quantization_config.activations_symmetric and quantization_config.weights_symmetric,
            },
        )

        LOGGER.info("Quantizing model...")
        quantizer.quantize_model()

        LOGGER.info(
            f"Saving quantized model at: {onnx_quantized_model_output_path} (external data format: "
            f"{use_external_data_format})"
        )
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
        seed: int = 2016,
    ) -> Dataset:
        """
        Create the calibration `datasets.Dataset` to use for the post-training static quantization calibration step

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                to load to use for the calibration step.
            num_samples (`int`, defaults to 100):
                The maximum number of samples composing the calibration dataset.
            dataset_config_name (`str`, *optional*):
                The name of the dataset configuration.
            dataset_split (`str`, *optional*):
                Which split of the dataset to use to perform the calibration step.
            preprocess_function (`Callable`, *optional*):
                Processing function to apply to each example after loading dataset.
            preprocess_batch (`int`, defaults to `True`):
                Whether the `preprocess_function` should be batched.
            seed (`int`, defaults to 2016):
                The random seed to use when shuffling the calibration dataset.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration
            step.
        """
        if dataset_name is None:
            raise ValueError(
                "ORTQuantizer: Static quantization calibration step requires a dataset_name if no calib_dataset is "
                "provided."
            )

        calib_dataset = load_dataset(dataset_name, name=dataset_config_name, split=dataset_split)

        if num_samples is not None:
            num_samples = min(num_samples, len(calib_dataset))
            calib_dataset = calib_dataset.shuffle(seed=seed).select(range(num_samples))

        if preprocess_function is not None:
            processed_calib_dataset = calib_dataset.map(preprocess_function, batched=preprocess_batch)
        else:
            processed_calib_dataset = calib_dataset

        return self.clean_calibration_dataset(processed_calib_dataset)

    def clean_calibration_dataset(self, dataset: Dataset) -> Dataset:
        ignored_columns = list(set(dataset.column_names) - set(self._onnx_config.inputs.keys()))

        return dataset.remove_columns(ignored_columns)
