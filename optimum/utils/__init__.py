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
import importlib.util


CONFIG_NAME = "config.json"

_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
_pydantic_available = importlib.util.find_spec("pydantic") is not None
_accelerate_available = importlib.util.find_spec("accelerate") is not None


def is_onnxruntime_available():
    return _onnxruntime_available


def is_pydantic_available():
    return _pydantic_available


def is_accelerate_available():
    return _accelerate_available
