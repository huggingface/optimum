# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from torch.ao.quantization.quantize_fx import convert_fx
from transformers.optimization import AdamW

from ...utils.runs import QuantizationApproach
from .functions import calibrate, calibrate_qat, prepare_fx, prepare_qat_fx


if TYPE_CHECKING:
    from datasets import Dataset
    from torch.ao.quantization.quantize_fx import ObservedGraphModule
    from transformers import PreTrainedModel


def quantize(
    model: "PreTrainedModel",
    approach: Union[str, QuantizationApproach],
    qconfig_dict: Dict[str, Any],
    preprocess_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    calibration_dataset: Optional["Dataset"] = None,
    num_calibration_samples: int = -1,
    input_names: Optional[List[str]] = None,
) -> torch.nn.Module:
    """
    Handles the quantization of the model end-to-end, by taking care of all the intermediate steps.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to quantize.
        approach (`str` or [`~optimum.utils.runs.QuantizationApproach`]):
            The quantization approach.
        qconfig_dict (`Dict[str, Any]`):
            The dictionary specifying how each part of the model should be quantized.
            Please refer to [the PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.quantization.quantize_fx.prepare_fx.html#torch.quantization.quantize_fx.prepare_fx)
            for more details.
        preprocess_func (`Callable[[Dict[str, Any]], [Dict[str, Any]], *optional*):
            The preprocessing function to apply to the calibration dataset to make the data ready to be fed to the
            model.
        calibration_dataset (`datasets.Dataset`, *optional*):
            The calibration dataset to use for calibrating the model quantization parameters or QAT.
        num_calibration_samples (`int`, defaults to -1):
            The number of examples to use from calibration_dataset.
        input_names (`List[str]`, *optional*):
            The name of the inputs to keep when tracing the model for quantization.

    Returns:
        `torch.nn.Module`:
            The quantized version of the model.
    """
    prepare_fn = prepare_qat_fx if approach is QuantizationApproach.qat else prepare_fx
    # TODO: handle the possibility to provide other arguments (equalization qconfig etc)
    prepared_model = prepare_fn(model, qconfig_dict, input_names=input_names)

    calibration_dataloader = None
    if calibration_dataset is not None:
        if num_calibration_samples > 0:
            calibration_dataset = calibration_dataset.shuffle().select(range(num_calibration_samples))
        columns_to_keep = input_names if input_names else model.dummy_inputs.keys()
        calibration_dataset = calibration_dataset.map(
            preprocess_func, remove_columns=[n for n in calibration_dataset.column_names if n not in columns_to_keep]
        )
        calibration_dataset = calibration_dataset.with_format("torch")
        calibration_dataloader = torch.utils.data.DataLoader(calibration_dataset)

    calibration_info = {
        QuantizationApproach.dynamic: None,
        QuantizationApproach.static: (calibrate, (prepared_model, calibration_dataloader)),
        # TODO: give more flexibility on the AdamW parameters.
        QuantizationApproach.qat: (
            calibrate_qat,
            (prepared_model, calibration_dataloader, AdamW(prepared_model.parameters())),
        ),
    }

    if type(approach) is str:
        approach = QuantizationApproach(approach)

    if approach in [QuantizationApproach.static, QuantizationApproach.qat]:
        if calibration_dataset is None:
            raise ValueError(
                "You must provide a calibration dataset for post training static quantization or quantization aware training."
            )
        calibrate_fn, args = calibration_info[approach]
        calibrate_fn(*args)

    quantized = convert_fx(prepared_model)
    return quantized
