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

import importlib
import os
import re
from enum import Enum
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from tqdm import tqdm
from transformers import EvalPrediction
from transformers.trainer_pt_utils import nested_concat
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import logging

import onnxruntime as ort

from ..exporters.onnx import OnnxConfig, OnnxConfigWithLoss


if TYPE_CHECKING:
    from datasets import Dataset

    from .modeling_ort import ORTModel


logger = logging.get_logger(__name__)

ONNX_WEIGHTS_NAME = "model.onnx"

ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model.onnx"
ONNX_DECODER_MERGED_NAME = "decoder_model_merged.onnx"

_ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}


def _is_gpu_available():
    """
    Checks if a gpu is available.
    """
    available_providers = ort.get_available_providers()
    if (
        "CUDAExecutionProvider" in available_providers or "ROCMExecutionProvider" in available_providers
    ) and torch.cuda.is_available():
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
    Checks if CuPy is available.
    """
    # Don't use _is_package_available as it doesn't work with CuPy installed
    # with `cupy-cuda*` and `cupy-rocm-*` package name (prebuilt wheels).
    return importlib.util.find_spec("cupy") is not None


class ORTConfigManager:
    """
    A class that contains all the information needed by ONNX Runtime optimization for a given model type.

    Attributes:
        _conf (`Dict[str]`):
            A dictionary mapping each supported model type to the corresponding ONNX Runtime model type.
    """

    # Contribution note: Please add new models in alphabetical order
    # TODO: for encoder-decoder models, validate if bert or gpt2 optimization is better
    _conf = {
        "albert": "bert",
        "bart": "bart",
        "bert": "bert",
        "big-bird": "bert",
        "bigbird-pegasus": "bart",
        "blenderbot": "bert",
        "bloom": "gpt2",
        "camembert": "bert",
        "codegen": "gpt2",
        "deberta": "bert",
        "deberta-v2": "bert",
        "dinov2": "vit",
        "distilbert": "bert",
        "electra": "bert",
        "gpt2": "gpt2",
        "gpt-bigcode": "gpt2",
        "gpt-neo": "gpt2",
        "gpt-neox": "gpt2",
        "gptj": "gpt2",
        "granite": "gpt2",
        "longt5": "bert",
        "llama": "gpt2",
        "marian": "bart",
        "mbart": "bart",
        "mistral": "gpt2",
        "mpnet": "bert",
        "mt5": "bart",
        "m2m-100": "bart",
        "nystromformer": "bert",
        "pegasus": "bert",
        "roberta": "bert",
        "segformer": "vit",
        "t5": "bert",
        "vit": "vit",
        "whisper": "bart",
        "xlm-roberta": "bert",
        "pix2struct": "vit",
    }

    @classmethod
    def get_model_ort_type(cls, model_type: str) -> str:
        model_type = model_type.replace("_", "-")
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
    def check_optimization_supported_model(cls, model_type: str, optimization_config):
        # as of 1.15.O: https://github.com/microsoft/onnxruntime/blob/v1.15.0/onnxruntime/python/tools/transformers/optimizer.py#L42
        supported_model_types_for_optimization = [
            "bart",
            "bert",
            "gpt2",
            "tnlr",
            "t5",
            "unet",
            "vae",
            "clip",
            "vit",
            "swin",
            "swinv2",
        ]
        model_type = model_type.replace("_", "-")
        if (model_type not in cls._conf) or (cls._conf[model_type] not in supported_model_types_for_optimization):
            raise NotImplementedError(
                f"ONNX Runtime doesn't support the graph optimization of {model_type} yet. Only {list(cls._conf.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue in ONNX Runtime: https://github.com/microsoft/onnxruntime."
            )


def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


def wrap_onnx_config_for_loss(onnx_config: OnnxConfig) -> OnnxConfig:
    return OnnxConfigWithLoss(onnx_config)


def get_device_for_provider(provider: str, provider_options: Dict) -> torch.device:
    """
    Gets the PyTorch device (CPU/CUDA) associated with an ONNX Runtime provider.
    """
    if provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider"]:
        return torch.device(f"cuda:{provider_options['device_id']}")
    else:
        return torch.device("cpu")


def get_provider_for_device(device: torch.device) -> str:
    """
    Gets the ONNX Runtime provider associated with the PyTorch device (CPU/CUDA).
    """
    if device.type.lower() == "cuda":
        if "ROCMExecutionProvider" in ort.get_available_providers():
            return "ROCMExecutionProvider"
        else:
            return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def parse_device(device: Union[torch.device, str, int]) -> Tuple[torch.device, Dict]:
    """Gets the relevant torch.device from the passed device, and if relevant the provider options (e.g. to set the GPU id)."""
    if device == -1:
        device = torch.device("cpu")
    else:
        device = torch._C._nn._parse_to(device)[0]

    provider_options = {}

    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")

        provider_options["device_id"] = device.index

    return device, provider_options


def validate_provider_availability(provider: str):
    """
    Ensure the ONNX Runtime execution provider `provider` is available, and raise an error if it is not.

    Args:
        provider (str): Name of an ONNX Runtime execution provider.
    """
    # Disable on Windows as reported in https://github.com/huggingface/optimum/issues/769.
    # Disable as well for ORT 1.16.0 that has changed changed the way _ld_preload.py is filled: https://github.com/huggingface/optimum/issues/1402.
    if (
        version.parse(ort.__version__) < version.parse("1.16.0")
        and os.name != "nt"
        and provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
    ):
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
                elif os.path.isfile(path_cuda_lib) and is_onnxruntime_training_available():
                    if provider == "TensorrtExecutionProvider":
                        raise ImportError(
                            f"Asked to use {provider}, but `onnxruntime-training` package doesn't support {provider}. Please use `CUDAExecutionProvider` instead."
                        )
                else:
                    raise ImportError(
                        f"Asked to use {provider}, but `onnxruntime-gpu` package was not found. Make sure to install `onnxruntime-gpu` package instead of `onnxruntime`."
                    )

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
    if use_io_binding is None and providers[0] == "CUDAExecutionProvider":
        use_io_binding = True
    elif providers[0] != "CPUExecutionProvider" and providers[0] != "CUDAExecutionProvider":
        if use_io_binding is True:
            logger.warning(
                "No need to enable IO Binding if the provider used is neither CPUExecutionProvider nor CUDAExecutionProvider. IO Binding will be turned off."
            )
        use_io_binding = False

    return use_io_binding


def get_ordered_input_names(input_names: List[str], func: Callable) -> List[str]:
    """
    Returns the input names from input_names keys ordered according to the signature of func. This is especially useful with the
    forward function when using IO Binding, as the input order of the ONNX and forward may be different.

    Method inspired from OnnxConfig.ordered_inputs.

    Args:
        input_names (`List[str]`):
            Names of the inputs of the ONNX model.
        func (`Callable`):
            Callable to remap the input_names order to.

    """
    signature_func = signature(func)
    _ordered_input_names = []
    for param in signature_func.parameters:
        param_regex = re.compile(rf"{param}(\.\d*)?")  # will for example match input_ids, past_key_values.0
        for name in input_names:
            if re.search(param_regex, name):
                _ordered_input_names.append(name)
    return _ordered_input_names


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


def evaluation_loop(
    model: "ORTModel",
    dataset: "Dataset",
    label_names: Optional[List[str]] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
):
    """
    Run evaluation and returns metrics and predictions.

    Args:
        model (`ORTModel`):
            The ONNXRuntime model to use for the evaluation step.
        dataset (`datasets.Dataset`):
            Dataset to use for the evaluation step.
        label_names (`List[str]`, `optional`):
            The list of keys in your dictionary of inputs that correspond to the labels.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take an `EvalPrediction` and
            return a dictionary string to metric values.
    """

    all_preds = None
    all_labels = None

    for inputs in tqdm(dataset, desc="Evaluation"):
        has_labels = all(inputs.get(k) is not None for k in label_names)
        if has_labels:
            labels = tuple(np.array([inputs.get(name)]) for name in label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        inputs = {key: np.array([inputs[key]]) for key in model.input_names if key in inputs}
        preds = model(**inputs)

        if len(preds) == 1:
            preds = preds[0]

        all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
        all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

    if compute_metrics is not None and all_preds is not None and all_labels is not None:
        metrics = compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    else:
        metrics = {}

    return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(dataset))


def np_to_pt_generators(np_object, device):
    if isinstance(np_object, np.random.RandomState):
        return torch.Generator(device=device).manual_seed(int(np_object.get_state()[1][0]))
    elif isinstance(np_object, np.random.Generator):
        return torch.Generator(device=device).manual_seed(int(np_object.bit_generator.state[1][0]))
    elif isinstance(np_object, list) and isinstance(np_object[0], (np.random.RandomState, np.random.Generator)):
        return [np_to_pt_generators(a, device) for a in np_object]
    elif isinstance(np_object, dict) and isinstance(
        next(iter(np_object.values())), (np.random.RandomState, np.random.Generator)
    ):
        return {k: np_to_pt_generators(v, device) for k, v in np_object.items()}
    else:
        return np_object


class DummyWhisperModel:
    def __init__(self):
        self.encoder = self.Encoder()

    class Encoder:
        def __init__(self):
            self.conv1 = self.Conv(stride=(1,))
            self.conv2 = self.Conv(stride=(2,))

        class Conv:
            def __init__(self, stride):
                self.stride = stride
