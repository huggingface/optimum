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

import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from optimum.onnxruntime import ORTQuantizableOperator
from optimum.onnxruntime.configuration import CalibrationConfig, NodeName, NodeType, QuantizationConfig
from optimum.onnxruntime.modeling_ort import ORTModel
from optimum.onnxruntime.modeling_seq2seq import ORTModelForConditionalGeneration
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.onnxruntime.utils import ONNX_WEIGHTS_NAME
from optimum.quantization_base import OptimumQuantizer


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


class ORTQuantizer(OptimumQuantizer):
    """
    Handles the ONNX Runtime quantization process for models shared on huggingface.co/models.
    """

    def __init__(self, onnx_model_path: List[Path]):
        """
        Args:
            onnx_model_path (`Path`):
                Path to the onnx model files you want to quantize.
        """
        super().__init__()
        self.onnx_model_path = onnx_model_path
        self._calibrator = None

    @classmethod
    def from_pretrained(
        cls,
        model_or_path: Union[str, Path],
        file_name: Optional[str] = None,
    ) -> "ORTQuantizer":
        """
        Instantiate a `ORTQuantizer` from a pretrained pytorch model and preprocessor.

        Args:
            model_or_path (`Union[str, Path]`):
                Can be either:
                    - A path to a saved exported ONNX Intermediate Representation (IR) model, e.g., `./my_model_directory/.
                    - Or a `ORTModelForXX` class, e.g., `ORTModelForQuestionAnswering`.
            file_name(`Union[str, List[str]]`, *optional*):
                Overwrites the default model file name from `"model.onnx"` to `file_name`.
                This allows you to load different model files from the same repository or directory.
        Returns:
            An instance of `ORTQuantizer`.
        """
        # define the file name for the quantizable models
        if file_name is None:
            if isinstance(model_or_path, ORTModel):
                if isinstance(model_or_path, ORTModelForConditionalGeneration):
                    raise ValueError(
                        "ORTQuantizer does not support multi-file quantization. Please create separate ORTQuantizer instances for each model/file."
                    )
                model_file_name = model_or_path.latest_model_name
            else:
                model_file_name = ONNX_WEIGHTS_NAME
        else:
            model_file_name = file_name

        # create ORTQuantizer based on the provided input
        if isinstance(model_or_path, ORTModel):
            return cls(model_or_path.model_save_dir.joinpath(model_file_name))
        # load from local path
        elif os.path.isdir(model_or_path):
            if not isinstance(model_or_path, Path):
                model_or_path = Path(model_or_path)
            return cls(model_or_path.joinpath(model_file_name))
        else:
            raise ValueError(f"Unable to load model from {model_or_path}.")

    def calibrate(
        self,
        dataset: Dataset,
        calibration_config: CalibrationConfig,
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

        self.partial_calibrate(
            dataset,
            calibration_config,
            onnx_augmented_model_name,
            operators_to_quantize,
            batch_size,
            use_external_data_format,
            use_gpu,
            force_symmetric_range,
        )
        return self.compute_ranges()

    def partial_calibrate(
        self,
        dataset: Dataset,
        calibration_config: CalibrationConfig,
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
        # If no calibrator, then create one
        if calibration_config.method is not None:
            LOGGER.info(f"Creating calibrator: {calibration_config.method}({calibration_config})")
            self._calibrator = calibration_config.create_calibrator(
                onnx_model_path=self.onnx_model_path.as_posix(),
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

    def fit(
        self,
        output_path: Union[str, Path],
        quantization_config: QuantizationConfig,
        file_suffix: Optional[str] = "quantized",
        calibration_tensors_range: Optional[Dict[NodeName, Tuple[float, float]]] = None,
        use_external_data_format: bool = False,
        preprocessor: Optional[QuantizationPreprocessor] = None,
    ) -> Path:
        """
        Quantize a model given the optimization specifications defined in `quantization_config`.

        Args:
            output_path (`Union[str, Path]`):
                The path used to save the quantized model exported to an ONNX Intermediate Representation (IR).
            quantization_config (`QuantizationConfig`):
                The configuration containing the parameters related to quantization.
            file_suffix (`str`, *optional*, defaults to `"quantized"`):
                The file_suffix used to save the quantized model.
            calibration_tensors_range (`Dict[NodeName, Tuple[float, float]]`, *optional*):
                The dictionary mapping the nodes name to their quantization ranges, used and required only when applying
                static quantization.
            use_external_data_format (`bool`, defaults to `False`):
                Whether to use external data format to store model which size is >= 2Gb.
            preprocessor (`QuantizationPreprocessor`, *optional*):
                The preprocessor to use to collect the nodes to include or exclude from quantization.

        Returns:
            The path of the resulting quantized model.
        """
        use_qdq = quantization_config.is_static and quantization_config.format == QuantFormat.QDQ
        output_path = Path(output_path).joinpath(f"{self.onnx_model_path.stem}_{file_suffix}.onnx")

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
            f"Creating {'static' if quantization_config.is_static else 'dynamic'} quantizer: {quantization_config}"
        )

        if preprocessor is not None:
            LOGGER.info("Preprocessor detected, collecting nodes to include/exclude")
            nodes_to_quantize, nodes_to_exclude = preprocessor.collect(self.onnx_model_path)

            nodes_to_quantize.update(quantization_config.nodes_to_quantize)
            nodes_to_exclude.update(quantization_config.nodes_to_exclude)

            quantization_config.nodes_to_quantize = list(nodes_to_quantize)
            quantization_config.nodes_to_exclude = list(nodes_to_exclude)

        onnx_model = onnx.load(self.onnx_model_path)
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

        LOGGER.info(f"Saving quantized model at: {output_path} (external data format: " f"{use_external_data_format})")
        quantizer.model.save_model_to_file(output_path, use_external_data_format)

        return Path(output_path)

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
        model = onnx.load(self.onnx_model_path)
        model_inputs = {input.name for input in model.graph.input}
        ignored_columns = list(set(dataset.column_names) - model_inputs)

        return dataset.remove_columns(ignored_columns)
