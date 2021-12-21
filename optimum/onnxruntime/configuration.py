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

from typing import Any, Dict, Tuple, Union

from ..configuration_utils import BaseConfig


class ORTConfig(BaseConfig):

    CONFIG_NAME = "ort_config.json"
    FULL_CONFIGURATION_FILE = "ort_config.json"

    def __init__(self, **kwargs):
        self.opset = kwargs.pop("opset", None)
        self.optimize_model = kwargs.pop("optimize_model", True)
        self.opt_level = kwargs.pop("opt_level", None)
        self.only_onnxruntime = kwargs.pop("only_onnxruntime", False)
        self.use_gpu = kwargs.pop("use_gpu", False)
        self.quantization_approach = kwargs.pop("quantization_approach", None)
        self.per_channel = kwargs.pop("per_channel", False)
        self.reduce_range = kwargs.pop("reduce_range", False)
        self.activation_type = kwargs.pop("activation_type", "uint8")
        self.weight_type = kwargs.pop("weight_type", "uint8")
        self.quant_format = kwargs.pop("quant_format", "operator")
        self.calibration_method = kwargs.pop("calibration_method", "minmax")
        self.split = kwargs.pop("split", "train")
        self.max_samples = kwargs.pop("max_samples", 80)
        self.calib_batch_size = kwargs.pop("calib_batch_size", 8)
        self.use_external_data_format = kwargs.pop("use_external_data_format", False)
