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

from enum import Enum
from typing import Dict, Tuple, Type, Union

import torch
from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from transformers.utils import logging

import onnx
import onnxruntime as ort

from ..onnx import OnnxConfigWithLoss, OnnxConfigWithPastAndLoss, OnnxSeq2SeqConfigWithPastAndLoss
from ..utils import NormalizedTextConfig, is_onnxruntime_gpu_available, is_onnxruntime_strict_available


logger = logging.get_logger(__name__)

ONNX_WEIGHTS_NAME = "model.onnx"
OPTIMIZED_ONNX_WEIGHTS_NAME = "optimized_model.onnx"
QUANTIZED_ONNX_WEIGHTS_NAME = "q8_model.onnx"

ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model.onnx"


def _is_gpu_available():
    """
    checks if a gpu is available.
    """
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers and torch.cuda.is_available():
        return True
    else:
        return False


BartLikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="encoder_attention_heads",
    hidden_size="d_model",
)
GPT2LikeNormalizedTextConfig = NormalizedTextConfig.with_args(num_attention_heads="n_head", hidden_size="n_embd")
T5LikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    num_attention_heads="num_heads",
    hidden_size="d_model",
)
WhisperLikeNormalizedTextConfig = NormalizedTextConfig.with_args(
    hidden_size="d_model",
)


class ORTConfigManager:
    """
    A class that contains all the information needed by ONNX Runtime optimization for a given model type.

    Attributes:
        _conf (`Dict[str, tuple]`):
            A dictionary mapping each supported model type to a tuple containing the number of attention heads
            and the hidden size model config attribute names as well as the corresponding ONNX Runtime model type.
    """

    # Contribution note: Please add new models in alphabetical order
    _conf = {
        "albert": (NormalizedTextConfig, "bert"),
        "bart": (BartLikeNormalizedTextConfig, "bart"),
        "bert": (NormalizedTextConfig, "bert"),
        "big_bird": (NormalizedTextConfig, "bert"),
        "bigbird_pegasus": (BartLikeNormalizedTextConfig, None),  # bug in `fusion_skiplayernorm.py`
        "camembert": (NormalizedTextConfig, "bert"),
        "codegen": (GPT2LikeNormalizedTextConfig, "gpt2"),
        "deberta": (NormalizedTextConfig, "bert"),
        "deberta-v2": (NormalizedTextConfig, "bert"),
        "distilbert": (NormalizedTextConfig.with_args(num_attention_heads="n_heads", hidden_size="dim"), "bert"),
        "electra": (NormalizedTextConfig, "bert"),
        "gpt2": (GPT2LikeNormalizedTextConfig, "gpt2"),
        "gpt_neo": (NormalizedTextConfig.with_args(num_attention_heads="num_heads"), "gpt2"),
        "marian": (BartLikeNormalizedTextConfig, "bart"),
        "mbart": (BartLikeNormalizedTextConfig, "bart"),
        "mt5": (T5LikeNormalizedTextConfig, "bart"),
        "m2m_100": (BartLikeNormalizedTextConfig, "bart"),
        "roberta": (NormalizedTextConfig, "bert"),
        "t5": (T5LikeNormalizedTextConfig, "t5"),
        "whisper": (WhisperLikeNormalizedTextConfig, "whisper"),
        "xlm-roberta": (NormalizedTextConfig, "bert"),
    }

    @classmethod
    def get_normalized_config_class(cls, model_type: str) -> Type:
        cls.check_supported_model(model_type)
        return cls._conf[model_type][0]

    @classmethod
    def get_model_ort_type(cls, model_type: str) -> str:
        cls.check_supported_model(model_type)
        return cls._conf[model_type][1]

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
        if (model_type not in cls._conf) or (cls._conf[model_type][1] not in supported_model_types_for_optimization):
            raise KeyError(
                f"ONNX Runtime doesn't support the graph optimization of {model_type} yet. Only {supported_model_types_for_optimization} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue in ONNX Runtime:https://github.com/microsoft/onnxruntime."
            )


def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


# TODO: shouldn't it be in optimum/onnx/graph_transformations.py?
def fix_atenops_to_gather(model_path):
    # Fix broken ATenOp nodes back to Gather nodes.
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    nodes = model.graph.node

    for node in nodes:
        if node.op_type in ["ATenOp", "ATen"]:
            logger.info(f"----Start fixing node: {node.name}----")
            op_num = node.name.split("_")[-1]
            new_node = onnx.helper.make_node(
                "Gather",
                name="Gather_" + op_num,
                inputs=[node.input[0], node.input[1]],
                outputs=node.output,
            )

            model.graph.node.remove(node)
            model.graph.node.insert(int(op_num), new_node)

    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def wrap_onnx_config_for_loss(onnx_config: OnnxConfig) -> OnnxConfig:
    if isinstance(onnx_config, OnnxSeq2SeqConfigWithPast):
        return OnnxSeq2SeqConfigWithPastAndLoss(onnx_config)
    elif isinstance(onnx_config, OnnxConfigWithPast):
        return OnnxConfigWithPastAndLoss(onnx_config)
    else:
        return OnnxConfigWithLoss(onnx_config)


def get_device_for_provider(provider: str) -> torch.device:
    """
    Gets the PyTorch device (CPU/CUDA) associated with an ONNX Runtime provider.
    """
    return (
        torch.device("cuda")
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
    Ensure the ONNX Runtime execution provider `provider` is available.

    Args:
        provider (str): Name of an ONNX Runtime execution provider.
    """

    if provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
        additional_message = ""
        if provider == "CUDAExecutionProvider":
            additional_message = " Additionally, refer to https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html for required dependencies."
        if provider == "TensorrtExecutionProvider":
            additional_message = " Additionally, refer to https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html and https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider for required dependencies."

        if is_onnxruntime_strict_available() == True and is_onnxruntime_gpu_available() == True:
            raise ValueError(
                f"Both `onnxruntime` and `onnxruntime-gpu` packages are installed. To use Optimum ONNX Runtime integration on GPU, only `onnxruntime-gpu` should be installed.{additional_message}"
            )
        elif is_onnxruntime_strict_available() == True and is_onnxruntime_gpu_available() == False:
            raise ValueError(
                f"The `onnxruntime` package was found, please install instead `onnxruntime-gpu` to use Optimum ONNX Runtime integration on GPU.{additional_message}"
            )
        elif is_onnxruntime_gpu_available() == False:
            raise ValueError(
                f"Please install the `onnxruntime-gpu` to use the Optimum ONNX Runtime integration on GPU.{additional_message}"
            )

    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise ValueError(
            f"Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}."
        )


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
