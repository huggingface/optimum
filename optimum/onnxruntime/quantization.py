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
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

import numpy
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, default_data_collator
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from optimum.onnxruntime.utils import generate_identified_filename
from optimum.onnxruntime.configuration import ORTConfig


logger = logging.getLogger(__name__)


class ORTQuantizationMode(Enum):

    DYNAMIC = "dynamic"
    STATIC = "static"


SUPPORTED_QUANT_MODE = set([approach.value for approach in ORTQuantizationMode])

CALIB_METHOD = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy}

Q_FORMAT = {"operator": QuantFormat.QOperator, "qdq": QuantFormat.QDQ}

Q_TYPE = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}


class ORTCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calib_dataloader: DataLoader):
        self._iter = iter([{key: data[key].numpy() for key in data} for data in calib_dataloader])

    def get_next(self):
        return next(self._iter, None)


class ORTQuantizer:
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: Union[str, Path],
        ort_config: Union[str, ORTConfig],
        feature: str = "default",
        calib_dataset: Optional[Dataset] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        data_files: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            output_dir (:obj:`Union[str, Path]`):
                The output directory where the quantized model will be saved.
            ort_config (:obj:`Union[ORTConfig, str]`):
                Configuration file containing all the information related to the model quantization.
                Can be either:
                    - an instance of the class :class:`ORTConfig`,
                    - a string valid as input to :func:`ORTConfig.from_pretrained`.
            feature (:obj:`str`):
                Feature used when exporting the model.
            calib_dataset (:obj:`Dataset`, `optional`):
                Dataset to use for the calibration step.
            dataset_name (:obj:`str`, `optional`):
                Dataset repository name on the Hugging Face Hub or path to a local directory containing data files to
                load to use for the calibration step.
            dataset_config_name (:obj:`str`, `optional`):
                Name of the dataset configuration.
            data_files (:obj:`str`, `optional`):
                Path to source data files.
            preprocess_function (:obj:`Callable`, `optional`):
                Processing function to apply to each example after loading dataset.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(:obj:`str`, `optional`):
                The specific version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
        """
        config_kwargs_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None),
        ]
        config_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in config_kwargs_default}
        self.cache_dir = config_kwargs.get("cache_dir")
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
        self.model_path = self.output_dir.joinpath("model.onnx")
        self.quant_model_path = generate_identified_filename(self.model_path, "-quantized")
        if not isinstance(ort_config, ORTConfig):
            config_path = ort_config if ort_config is not None else model_path
            ort_config = ORTConfig.from_pretrained(config_path, **config_kwargs)
        self.ort_config = ort_config
        self.quantization_approach = ort_config.quantization_approach
        self.activation_type = Q_TYPE.get(ort_config.activation_type, QuantType.QUInt8)
        self.weight_type = Q_TYPE.get(ort_config.weight_type, QuantType.QUInt8)
        self.quant_format = Q_FORMAT.get(ort_config.quant_format, QuantFormat.QOperator)
        self.calibrate_method = CALIB_METHOD.get(ort_config.calibration_method, CalibrationMethod.MinMax)
        self.calib_dataset = calib_dataset
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.data_files = data_files
        self.preprocess_function = preprocess_function
        self.onnx_config = None
        self.feature = feature
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = FeaturesManager.get_model_from_feature(self.feature, self.model_name_or_path)

    def export(self) -> None:
        """
        Load and export a model to an ONNX Intermediate Representation (IR).
        """
        model_type, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            self.model, feature=self.feature
        )
        self.onnx_config = model_onnx_config(self.model.config)
        opset = self.onnx_config.default_onnx_opset if self.ort_config.opset is None else self.ort_config.opset
        _ = export(self.tokenizer, self.model, self.onnx_config, opset, self.model_path)

    def fit(self) -> None:
        """
        Load and export a model to an ONNX Intermediate Representation (IR) after applying the specified quantization
        approach.
        """
        self.export()
        if self.quantization_approach == ORTQuantizationMode.DYNAMIC.value:
            quantize_dynamic(
                self.model_path,
                self.quant_model_path,
                per_channel=self.ort_config.per_channel,
                reduce_range=self.ort_config.reduce_range,
                activation_type=self.activation_type,
                weight_type=self.weight_type,
                optimize_model=self.ort_config.optimize_model,
                use_external_data_format=self.ort_config.use_external_data_format,
            )
        elif self.quantization_approach == ORTQuantizationMode.STATIC.value:
            calib_dataset = self.calib_dataset if self.calib_dataset is not None else self.get_calib_dataset()
            calib_dataloader = self.get_calib_dataloader(calib_dataset)
            calib_data_reader = self.get_data_reader(calib_dataloader)
            quantize_static(
                self.model_path,
                self.quant_model_path,
                calib_data_reader,
                quant_format=self.quant_format,
                per_channel=self.ort_config.per_channel,
                reduce_range=self.ort_config.reduce_range,
                activation_type=self.activation_type,
                weight_type=self.weight_type,
                optimize_model=self.ort_config.optimize_model,
                use_external_data_format=self.ort_config.use_external_data_format,
                calibrate_method=self.calibrate_method,
            )
        else:
            raise ValueError(
                f"Unknown quantization approach: `quantization_approach` was set to {self.quantization_approach}. "
                f"Supported quantization approaches are " + ", ".join(SUPPORTED_QUANT_MODE)
            )

    def get_calib_dataset(self) -> Dataset:
        """
        Returns the calibration :class:`~datasets.arrow_dataset.Dataset` to use for the post-training static
        quantization calibration step.
        """
        if self.dataset_name is None:
            raise ValueError(
                "ORTQuantizer: Static quantization calibration step requires a dataset_name if no calib_dataset is "
                "provided."
            )
        if self.preprocess_function is None:
            raise ValueError(
                "ORTQuantizer: Processing function to apply after loading the dataset used for static quantization "
                "calibration step was not provided."
            )
        calib_dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config_name,
            data_files=self.data_files,
            split=self.ort_config.split,
            cache_dir=self.cache_dir,
        )
        calib_dataset = calib_dataset.map(self.preprocess_function, batched=True)
        return calib_dataset

    def get_calib_dataloader(self, calib_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the calibration :class:`~torch.utils.data.DataLoader`.
        Args:
            calib_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                If provided, will override :obj:`self.calib_dataset`.
        """
        if calib_dataset is None and self.calib_dataset is None:
            raise ValueError("ORTQuantizer: static quantization calibration step requires a calib_dataset.")

        calib_dataset = calib_dataset if calib_dataset is not None else self.calib_dataset

        if self.ort_config.max_samples is not None and len(calib_dataset) > self.ort_config.max_samples:
            calib_dataset = calib_dataset.select(range(self.ort_config.max_samples))

        ignored_columns = list(set(calib_dataset.column_names) - set(self.onnx_config.inputs.keys()))
        calib_dataset = calib_dataset.remove_columns(ignored_columns)

        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        sampler = RandomSampler(calib_dataset, generator=generator)

        return DataLoader(
            calib_dataset,
            batch_size=self.ort_config.calib_batch_size,
            sampler=sampler,
            collate_fn=default_data_collator,
        )

    @staticmethod
    def get_data_reader(calib_dataloader: DataLoader) -> ORTCalibrationDataReader:
        """
        Returns the calibration :class:`~optimum.onnxruntime.quantization.ORTCalibrationDataReader`.
        Args:
            calib_dataloader (:obj:`torch.utils.data.DataLoader`):
                Calibration dataloader to use for the post-training static quantization calibration step.
        """
        return ORTCalibrationDataReader(calib_dataloader)
