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
"""Classes handling quantization with ONNX Runtime."""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import onnx
from datasets import Dataset, load_dataset
from packaging.version import Version, parse
from transformers import AutoConfig

from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer

from ..quantization_base import OptimumQuantizer
from ..utils.save_utils import maybe_save_preprocessors
from . import ORTQuantizableOperator
from .configuration import CalibrationConfig, NodeName, NodeType, ORTConfig, QuantizationConfig
from .modeling_decoder import ORTModelForCausalLM
from .modeling_ort import ORTModel
from .modeling_seq2seq import ORTModelForConditionalGeneration
from .preprocessors import QuantizationPreprocessor


if TYPE_CHECKING:
    from transformers import PretrainedConfig

LOGGER = logging.getLogger(__name__)


class ORTCalibrationDataReader(CalibrationDataReader):
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

        if featurized_samples is not None and len(featurized_samples) > 0:
            return featurized_samples
        return None


class ORTQuantizer(OptimumQuantizer):
    """
    Handles the ONNX Runtime quantization process for models shared on huggingface.co/models.
    """

    def __init__(self, onnx_model_path: Path, config: Optional["PretrainedConfig"] = None):
        """
        Args:
            onnx_model_path (`Path`):
                Path to the onnx model files you want to quantize.
            config (`Optional[PretrainedConfig]`, defaults to `None`):
                The configuration of the model.
        """
        super().__init__()
        self.onnx_model_path = onnx_model_path
        self.config = config
        if self.config is None:
            try:
                self.config = AutoConfig.from_pretrained(self.onnx_model_path.parent)
            except OSError:
                LOGGER.warning(
                    f"Could not load the config for {self.onnx_model_path} automatically, this might make "
                    "the quantized model harder to use because it will not be able to be loaded by an ORTModel without "
                    "having to specify the configuration explicitly."
                )
        self._calibrator = None

    @classmethod
    def from_pretrained(
        cls,
        model_or_path: Union["ORTModel", str, Path],
        file_name: Optional[str] = None,
    ) -> "ORTQuantizer":
        """
        Instantiates a `ORTQuantizer` from a an ONNX model file or an `ORTModel`.

        Args:
            model_or_path (`Union[ORTModel, str, Path]`):
                Can be either:
                    - A path to a saved exported ONNX Intermediate Representation (IR) model, e.g., `./my_model_directory/.
                    - Or an `ORTModelForXX` class, e.g., `ORTModelForQuestionAnswering`.
            file_name(`Optional[str]`, defaults to `None`):
                Overwrites the default model file name from `"model.onnx"` to `file_name`.
                This allows you to load different model files from the same repository or directory.
        Returns:
            An instance of `ORTQuantizer`.
        """
        ort_quantizer_error_message = "ORTQuantizer does not support multi-file quantization. Please create separate ORTQuantizer instances for each model/file, by passing the argument `file_name` to ORTQuantizer.from_pretrained()."

        if isinstance(model_or_path, str):
            model_or_path = Path(model_or_path)

        path = None
        if isinstance(model_or_path, ORTModelForConditionalGeneration):
            raise NotImplementedError(ort_quantizer_error_message)
        elif isinstance(model_or_path, ORTModelForCausalLM):
            if model_or_path.use_cache is False:
                path = Path(model_or_path.decoder_model_path)
            elif model_or_path.use_cache is True and model_or_path.use_merged is False:
                raise NotImplementedError(ort_quantizer_error_message)
            else:
                path = Path(model_or_path.decoder_model_path)
        elif isinstance(model_or_path, Path) and file_name is None:
            onnx_files = list(model_or_path.glob("*.onnx"))
            if len(onnx_files) == 0:
                raise FileNotFoundError(f"Could not find any ONNX model file in {model_or_path}")
            elif len(onnx_files) > 1:
                raise RuntimeError(
                    f"Found too many ONNX model files in {model_or_path}. {ort_quantizer_error_message}"
                )
            file_name = onnx_files[0].name

        if isinstance(model_or_path, ORTModel):
            if path is None:
                path = Path(model_or_path.model._model_path)
        elif os.path.isdir(model_or_path):
            path = Path(model_or_path) / file_name
        else:
            raise ValueError(f"Unable to load model from {model_or_path}.")
        return cls(path)

    def fit(
        self,
        dataset: Dataset,
        calibration_config: CalibrationConfig,
        onnx_augmented_model_name: Union[str, Path] = "augmented_model.onnx",
        operators_to_quantize: Optional[List[NodeType]] = None,
        batch_size: int = 1,
        use_external_data_format: bool = False,
        use_gpu: bool = False,
        force_symmetric_range: bool = False,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Performs the calibration step and computes the quantization ranges.

        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            calibration_config ([`~CalibrationConfig`]):
                The configuration containing the parameters related to the calibration step.
            onnx_augmented_model_name (`Union[str, Path]`, defaults to `"augmented_model.onnx"`):
                The path used to save the augmented model used to collect the quantization ranges.
            operators_to_quantize (`Optional[List[NodeType]]`, defaults to `None`):
                List of the operators types to quantize.
            batch_size (`int`, defaults to 1):
                The batch size to use when collecting the quantization ranges values.
            use_external_data_format (`bool`, defaults to `False`):
                Whether to use external data format to store model which size is >= 2Gb.
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
        onnx_augmented_model_name: Union[str, Path] = "augmented_model.onnx",
        operators_to_quantize: Optional[List[NodeType]] = None,
        batch_size: int = 1,
        use_external_data_format: bool = False,
        use_gpu: bool = False,
        force_symmetric_range: bool = False,
    ):
        """
        Performs the calibration step and collects the quantization ranges without computing them.

        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            calibration_config (`CalibrationConfig`):
                The configuration containing the parameters related to the calibration step.
            onnx_augmented_model_name (`Union[str, Path]`, defaults to `"augmented_model.onnx"`):
                The path used to save the augmented model used to collect the quantization ranges.
            operators_to_quantize (`Optional[List[NodeType]]`, defaults to `None`):
                List of the operators types to quantize.
            batch_size (`int`, defaults to 1):
                The batch size to use when collecting the quantization ranges values.
            use_external_data_format (`bool`, defaults to `False`):
                Whether uto se external data format to store model which size is >= 2Gb.
            use_gpu (`bool`, defaults to `False`):
                Whether to use the GPU when collecting the quantization ranges values.
            force_symmetric_range (`bool`, defaults to `False`):
                Whether to make the quantization ranges symmetric.
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
        Computes the quantization ranges.

        Returns:
            The dictionary mapping the nodes name to their quantization ranges.
        """
        if self._calibrator is None:
            raise ValueError(
                "Calibrator is None, please call `partial_fit` or `fit` method at least ones to compute ranges."
            )

        LOGGER.info("Computing calibration ranges")
        return self._calibrator.compute_range()

    def quantize(
        self,
        quantization_config: QuantizationConfig,
        save_dir: Union[str, Path],
        file_suffix: Optional[str] = "quantized",
        calibration_tensors_range: Optional[Dict[NodeName, Tuple[float, float]]] = None,
        use_external_data_format: bool = False,
        preprocessor: Optional[QuantizationPreprocessor] = None,
    ) -> Path:
        """
        Quantizes a model given the optimization specifications defined in `quantization_config`.

        Args:
            quantization_config (`QuantizationConfig`):
                The configuration containing the parameters related to quantization.
            save_dir (`Union[str, Path]`):
                The directory where the quantized model should be saved.
            file_suffix (`Optional[str]`, defaults to `"quantized"`):
                The file_suffix used to save the quantized model.
            calibration_tensors_range (`Optional[Dict[NodeName, Tuple[float, float]]]`, defaults to `None`):
                The dictionary mapping the nodes name to their quantization ranges, used and required only when applying static quantization.
            use_external_data_format (`bool`, defaults to `False`):
                Whether to use external data format to store model which size is >= 2Gb.
            preprocessor (`Optional[QuantizationPreprocessor]`, defaults to `None`):
                The preprocessor to use to collect the nodes to include or exclude from quantization.

        Returns:
            The path of the resulting quantized model.
        """
        use_qdq = quantization_config.is_static and quantization_config.format == QuantFormat.QDQ
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if quantization_config.is_static and calibration_tensors_range is None:
            raise ValueError(
                "Requested static quantization in the QuantizationConfig, but no calibration ranges were provided. Please run calibration first using the quantizer fit method, or use dynamic quantization."
            )
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

        has_subgraphs = False
        onnx_model = onnx.load(Path(self.onnx_model_path).as_posix())
        for node in onnx_model.graph.node:
            if node.op_type in ["If", "Loop", "Scan", "SequenceMap"]:
                has_subgraphs = True
                break

        if quantization_config.is_static and has_subgraphs:
            raise NotImplementedError("Static quantization is currently not supported for models with" " subgraphs.")

        quantizer_factory = QDQQuantizer if use_qdq else ONNXQuantizer

        if parse(ort_version) >= Version("1.13.0"):
            # The argument `input_qType` has been changed into `activation_qType` from ORT 1.13
            quantizer = quantizer_factory(
                model=onnx_model,
                static=quantization_config.is_static,
                per_channel=quantization_config.per_channel,
                mode=quantization_config.mode,
                weight_qType=quantization_config.weights_dtype,
                activation_qType=quantization_config.activations_dtype,
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
                    "EnableSubgraph": has_subgraphs,
                    "ForceSymmetric": quantization_config.activations_symmetric
                    and quantization_config.weights_symmetric,
                    "AddQDQPairToWeight": quantization_config.qdq_add_pair_to_weight,
                    "DedicatedQDQPair": quantization_config.qdq_dedicated_pair,
                    "QDQOpTypePerChannelSupportToAxis": quantization_config.qdq_op_type_per_channel_support_to_axis,
                },
            )
        else:
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
                    "ForceSymmetric": quantization_config.activations_symmetric
                    and quantization_config.weights_symmetric,
                    "AddQDQPairToWeight": quantization_config.qdq_add_pair_to_weight,
                    "DedicatedQDQPair": quantization_config.qdq_dedicated_pair,
                    "QDQOpTypePerChannelSupportToAxis": quantization_config.qdq_op_type_per_channel_support_to_axis,
                },
            )

        LOGGER.info("Quantizing model...")
        quantizer.quantize_model()

        suffix = f"_{file_suffix}" if file_suffix else ""
        quantized_model_path = save_dir.joinpath(f"{self.onnx_model_path.stem}{suffix}").with_suffix(".onnx")
        LOGGER.info(f"Saving quantized model at: {save_dir} (external data format: " f"{use_external_data_format})")
        quantizer.model.save_model_to_file(quantized_model_path.as_posix(), use_external_data_format)

        # Create and save the configuration summarizing all the parameters related to quantization
        ort_config = ORTConfig(quantization=quantization_config, use_external_data_format=use_external_data_format)
        ort_config.save_pretrained(save_dir)

        if self.config is not None:
            self.config.save_pretrained(save_dir)

        maybe_save_preprocessors(self.onnx_model_path.parent, save_dir)

        return Path(save_dir)

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        seed: int = 2016,
        use_auth_token: bool = False,
    ) -> Dataset:
        """
        Creates the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                to load to use for the calibration step.
            num_samples (`int`, defaults to 100):
                The maximum number of samples composing the calibration dataset.
            dataset_config_name (`Optional[str]`, defaults to `None`):
                The name of the dataset configuration.
            dataset_split (`Optional[str]`, defaults to `None`):
                Which split of the dataset to use to perform the calibration step.
            preprocess_function (`Optional[Callable]`, defaults to `None`):
                Processing function to apply to each example after loading dataset.
            preprocess_batch (`bool`, defaults to `True`):
                Whether the `preprocess_function` should be batched.
            seed (`int`, defaults to 2016):
                The random seed to use when shuffling the calibration dataset.
            use_auth_token (`bool`, defaults to `False`):
                Whether to use the token generated when running `transformers-cli login` (necessary for some datasets
                like ImageNet).
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration
            step.
        """
        if dataset_name is None:
            raise ValueError(
                "ORTQuantizer: Static quantization calibration step requires a dataset_name if no calib_dataset is "
                "provided."
            )

        calib_dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
            use_auth_token=use_auth_token,
        )

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
