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
"""Utility functions, classes and constants for ONNX Runtime."""

import importlib.util
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers.utils import logging

import onnxruntime as ort

from ..exporters.onnx import OnnxConfig, OnnxConfigWithLoss


logger = logging.get_logger(__name__)

ONNX_WEIGHTS_NAME = "model.onnx"

ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model.onnx"


def _is_gpu_available():
    """
    Checks if a gpu is available.
    """
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers and torch.cuda.is_available():
        return True
    else:
        return False


def is_onnxruntime_training_available():
    """
    Checks if onnxruntime-training is available.
    """
    path_training_dependecy = os.path.join(ort.__path__[0], "training")
    if os.path.exists(path_training_dependecy):
        return True
    else:
        return False


def is_cupy_available():
    """
    Checks if onnxruntime-training is available.
    """
    return importlib.util.find_spec("cupy") is not None


class ORTConfigManager:
    """
    A class that contains all the information needed by ONNX Runtime optimization for a given model type.

    Attributes:
        _conf (`Dict[str]`):
            A dictionary mapping each supported model type to the corresponding ONNX Runtime model type.
    """

    # Contribution note: Please add new models in alphabetical order
    _conf = {
        "albert": "bert",
        "bart": "bart",
        "bert": "bert",
        "big_bird": "bert",
        "bigbird_pegasus": None,  # bug in `fusion_skiplayernorm.py`
        "camembert": "bert",
        "codegen": "gpt2",
        "deberta": "bert",
        "deberta-v2": "bert",
        "distilbert": "bert",
        "electra": "bert",
        "gpt2": "gpt2",
        "gpt_neo": "gpt2",
        "marian": "bart",
        "mbart": "bart",
        "mt5": "bart",
        "m2m_100": "bart",
        "roberta": "bert",
        "t5": "t5",
        "whisper": "whisper",
        "xlm-roberta": "bert",
    }

    @classmethod
    def get_model_ort_type(cls, model_type: str) -> str:
        cls.check_supported_model(model_type)
        return cls._conf[model_type]

    @classmethod
    def check_supported_model(cls, model_type: str):
        if model_type not in cls._conf:
            model_types = ", ".join(cls._conf.keys())
            raise KeyError(
                f"{model_type} model type is not supported yet. Only {model_types} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )

    @classmethod
    def check_optimization_supported_model(cls, model_type: str):
        supported_model_types_for_optimization = ["bert", "gpt2", "bart"]
        if (model_type not in cls._conf) or (cls._conf[model_type] not in supported_model_types_for_optimization):
            raise KeyError(
                f"ONNX Runtime doesn't support the graph optimization of {model_type} yet. Only {supported_model_types_for_optimization} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue in ONNX Runtime:https://github.com/microsoft/onnxruntime."
            )


def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


def wrap_onnx_config_for_loss(onnx_config: OnnxConfig) -> OnnxConfig:
    return OnnxConfigWithLoss(onnx_config)


def get_device_for_provider(provider: str) -> torch.device:
    """
    Gets the PyTorch device (CPU/CUDA) associated with an ONNX Runtime provider.
    """
    return (
        torch.device("cuda:0")
        if provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
        else torch.device("cpu")
    )


def get_provider_for_device(device: torch.device) -> str:
    """
    Gets the ONNX Runtime provider associated with the PyTorch device (CPU/CUDA).
    """
    return "CUDAExecutionProvider" if device.type.lower() == "cuda" else "CPUExecutionProvider"


def parse_device(device: Union[torch.device, str, int]) -> Tuple[torch.device, Dict]:
    """Gets the relevant torch.device from the passed device, and if relevant the provider options (e.g. to set the GPU id)."""
    if device == -1:
        device = torch.device("cpu")
    else:
        device = torch._C._nn._parse_to(device)[0]

    provider_options = {}

    if device.type == "cuda":
        if device.index == None:
            device = torch.device("cuda:0")

        provider_options["device_id"] = device.index

    return device, provider_options


def validate_provider_availability(provider: str):
    """
    Ensure the ONNX Runtime execution provider `provider` is available, and raise an error if it is not.

    Args:
        provider (str): Name of an ONNX Runtime execution provider.
    """
    if provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
        path_cuda_lib = os.path.join(ort.__path__[0], "capi", "libonnxruntime_providers_cuda.so")
        path_trt_lib = os.path.join(ort.__path__[0], "capi", "libonnxruntime_providers_tensorrt.so")
        path_dependecy_loading = os.path.join(ort.__path__[0], "capi", "_ld_preload.py")

        with open(path_dependecy_loading, "r") as f:
            file_string = f.read()

            if "ORT_CUDA" not in file_string or "ORT_TENSORRT" not in file_string:
                if os.path.isfile(path_cuda_lib) and os.path.isfile(path_trt_lib):
                    raise ImportError(
                        f"`onnxruntime-gpu` is installed, but GPU dependencies are not loaded. It is likely there is a conflicting install between `onnxruntime` and `onnxruntime-gpu`. Please install only `onnxruntime-gpu` in order to use {provider}."
                    )
                else:
                    raise ImportError(
                        f"Asked to use {provider}, but `onnxruntime-gpu` package was not found. Make sure to install `onnxruntime-gpu` package instead of `onnxruntime`."
                    )

            from onnxruntime.capi import _ld_preload

            if provider == "CUDAExecutionProvider":
                if os.environ.get("ORT_CUDA_UNAVAILABLE", "0") == "1":
                    raise ImportError(
                        "`onnxruntime-gpu` package is installed, but CUDA requirements could not be loaded. Make sure to meet the required dependencies: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html"
                    )
            if provider == "TensorrtExecutionProvider":
                if os.environ.get("ORT_TENSORRT_UNAVAILABLE", "0") == "1":
                    raise ImportError(
                        "`onnxruntime-gpu` package is installed, but TensorRT requirements could not be loaded. Make sure to meet the required dependencies following https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html and https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider ."
                    )

    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise ValueError(
            f"Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}."
        )


def check_io_binding(providers: List[str], use_io_binding: Optional[bool] = None) -> bool:
    """
    Whether to use IOBinding or not.
    """
    if providers[0] == "CUDAExecutionProvider" and use_io_binding is None:
        use_io_binding = True
    elif providers[0] != "CUDAExecutionProvider":
        if use_io_binding is True:
            logger.warning(
                "No need to enable IO Binding if the provider used is not CUDAExecutionProvider. IO Binding will be turned off."
            )
        use_io_binding = False

    return use_io_binding


class ORTQuantizableOperator(Enum):
    # Common ops
    Gather = "Gather"
    Transpose = "Transpose"
    EmbedLayerNormalizationQuant = "EmbedLayerNormalization"

    # QLinearOps
    Conv = "Conv"
    MatMul = "MatMul"
    Add = "Add"
    Mul = "Mul"
    Relu = "Relu"
    Clip = "Clip"
    LeakyRelu = "LeakyRelu"
    Sigmoid = "Sigmoid"
    MaxPool = "MaxPool"
    GlobalAveragePool = "GlobalAveragePool"
    Split = "Split"
    Pad = "Pad"
    Reshape = "Reshape"
    Squeeze = "Squeeze"
    Unsqueeze = "Unsqueeze"
    Resize = "Resize"
    AveragePool = "AveragePool"
    Concat = "Concat"
