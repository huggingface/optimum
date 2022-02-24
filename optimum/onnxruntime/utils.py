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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from packaging.version import parse
from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.file_utils import is_torch_onnx_dict_inputs_support_available
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx.config import OnnxConfig
from transformers.onnx.convert import ensure_model_and_config_inputs_match
from transformers.utils import logging


logger = logging.get_logger(__name__)


def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)
