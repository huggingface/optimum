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

from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, default_data_collator
from transformers.onnx import OnnxConfig, export, validate_model_outputs

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    onnx_model,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from optimum.onnxruntime.utils import generate_identified_filename


class ORTQuantizationMode(Enum):

    DYNAMIC = "dynamic"
    STATIC = "static"


SUPPORTED_QUANT_MODE = set([approach.value for approach in ORTQuantizationMode])


class ORTCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calib_dataloader: DataLoader):
        self._iter = iter([{key: data[key].numpy() for key in data} for data in calib_dataloader])

    def get_next(self):
        return next(self._iter, None)


class ORTQuantizer:
    def __init__(
        self,
        model_name_or_path,
        output_dir,
        quantization_approach=None,
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        optimize_model=True,
        quant_format=QuantFormat.QOperator,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=False,
        calib_dataset=None,
        dataset_name=None,
        dataset_config_name=None,
        data_files=None,
        preprocess_function=None,
        batch_size=8,
        split="train",
        max_samples=80,
        cache_dir=None,
        config=None,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
        self.model_path = self.output_dir.joinpath("model.onnx")
        self.quant_model_path = generate_identified_filename(self.model_path, "-quantized")

        self.config = config
        self.approach = quantization_approach
        self.per_channel = per_channel
        self.reduce_range = reduce_range
        self.activation_type = activation_type
        self.weight_type = weight_type
        self.optimize_model = optimize_model
        self.use_external_data_format = use_external_data_format

        self.quant_format = quant_format
        self.calibrate_method = calibrate_method

        self.calib_dataset = calib_dataset
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.data_files = data_files
        self.preprocess_function = preprocess_function
        self.batch_size = batch_size
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir

        self.onnx_config = None
        self.feature = "default"
        self.opset = None

    def export(self):
        """
        Load and export a model to an ONNX Intermediate Representation (IR).
        """
        from transformers.onnx import export, validate_model_outputs
        from transformers.onnx.features import FeaturesManager

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = FeaturesManager.get_model_from_feature(self.feature, self.model_name_or_path)
        model_type, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=self.feature)
        self.onnx_config = model_onnx_config(model.config)
        self.opset = self.onnx_config.default_onnx_opset if self.opset is None else self.opset
        onnx_inputs, onnx_outputs = export(tokenizer, model, self.onnx_config, self.opset, self.model_path)

    def fit(self):

        self.export()

        if self.approach == ORTQuantizationMode.DYNAMIC.value:
            quantize_dynamic(
                self.model_path,
                self.quant_model_path,
                per_channel=self.per_channel,
                reduce_range=self.reduce_range,
                activation_type=self.activation_type,
                weight_type=self.weight_type,
                optimize_model=self.optimize_model,
                use_external_data_format=self.use_external_data_format,
            )

        elif self.approach == ORTQuantizationMode.STATIC.value:
            calib_dataset = self.calib_dataset if self.calib_dataset is not None else self.get_calib_dataset()
            calib_dataloader = self.get_calib_dataloader(calib_dataset)
            calib_data_reader = self.get_data_reader(calib_dataloader)

            quantize_static(
                self.model_path,
                self.quant_model_path,
                calib_data_reader,
                quant_format=self.quant_format,
                per_channel=self.per_channel,
                reduce_range=self.reduce_range,
                activation_type=self.activation_type,
                weight_type=self.weight_type,
                optimize_model=self.optimize_model,
                use_external_data_format=self.use_external_data_format,
                calibrate_method=self.calibrate_method,
            )

        else:
            raise ValueError(
                "Unknown quantization approach. Supported approach are " + ", ".join(SUPPORTED_QUANT_MODE)
            )

    def get_calib_dataset(self) -> Dataset:

        if self.dataset_name is None:
            raise ValueError(
                "ORTQuantizer: static quantization calibration step requires a dataset_name if no calib_dataset is "
                "provided."
            )

        calib_dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config_name,
            data_files=self.data_files,
            split=self.split,
            cache_dir=self.cache_dir,
        )
        calib_dataset = calib_dataset.map(self.preprocess_function, batched=True)

        return calib_dataset

    def get_calib_dataloader(self, calib_dataset: Optional[Dataset] = None) -> DataLoader:

        if calib_dataset is None and self.calib_dataset is None:
            raise ValueError("ORTQuantizer: static quantization calibration step requires a calib_dataset.")

        calib_dataset = calib_dataset if calib_dataset is not None else self.calib_dataset

        if self.max_samples is not None and len(calib_dataset) > self.max_samples:
            calib_dataset = calib_dataset.select(range(self.max_samples))

        ignored_columns = list(set(calib_dataset.column_names) - set(self.onnx_config.inputs.keys()))
        calib_dataset = calib_dataset.remove_columns(ignored_columns)

        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        sampler = RandomSampler(calib_dataset, generator=generator)

        return DataLoader(
            calib_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=default_data_collator,
        )

    @staticmethod
    def get_data_reader(calib_dataloader: DataLoader) -> ORTCalibrationDataReader:
        return ORTCalibrationDataReader(calib_dataloader)

