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

import torch
from transformers.onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from transformers.utils import logging

import onnx
import onnxruntime as ort

from ..onnx import OnnxConfigWithLoss, OnnxConfigWithPastAndLoss, OnnxSeq2SeqConfigWithPastAndLoss


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
        "albert": ("num_attention_heads", "hidden_size", "bert"),
        "bart": ("encoder_attention_heads", "d_model", "bart"),
        "bert": ("num_attention_heads", "hidden_size", "bert"),
        "big_bird": ("num_attention_heads", "hidden_size", "bert"),
        "camembert": ("num_attention_heads", "hidden_size", "bert"),
        "codegen": ("n_head", "n_embd", "gpt2"),
        "deberta": ("num_attention_heads", "hidden_size", "bert"),
        "deberta-v2": ("num_attention_heads", "hidden_size", "bert"),
        "distilbert": ("n_heads", "dim", "bert"),
        "electra": ("num_attention_heads", "hidden_size", "bert"),
        "gpt2": ("n_head", "n_embd", "gpt2"),
        "gpt_neo": ("num_heads", "hidden_size", "gpt2"),
        "mt5": ("num_heads", "d_model", "bart"),
        "marian": ("encoder_attention_heads", "d_model", "bart"),
        "roberta": ("num_attention_heads", "hidden_size", "bert"),
        "xlm-roberta": ("num_attention_heads", "hidden_size", "bert"),
    }

    @classmethod
    def get_num_heads_name(cls, model_type: str) -> str:
        num_heads = "num_attention_heads"
        try:
            num_heads = cls._conf[model_type][0]
        except KeyError:
            logger.warning(
                f"{model_type} is not supported yet. Only {list(cls._conf.keys())} are supported. The default value to "
                f"access the number of heads defined in the config is set to `{num_heads}`."
            )
        return num_heads

    @classmethod
    def get_hidden_size_name(cls, model_type: str) -> str:
        hidden_size = "hidden_size"
        try:
            hidden_size = cls._conf[model_type][1]
        except KeyError:
            logger.warning(
                f"{model_type} is not supported yet. Only {list(cls._conf.keys())} are supported. The default value to "
                f"access the hidden size defined in the config is set to `{hidden_size}`."
            )
        return hidden_size

    @classmethod
    def get_model_ort_type(cls, model_type: str) -> str:
        try:
            model_type = cls._conf[model_type][2]
        except KeyError:
            logger.warning(f"{model_type} is not supported yet. Only {list(cls._conf.keys())} are supported.")
        return model_type

    @classmethod
    def check_supported_model_or_raise(cls, model_type: str) -> bool:
        if model_type not in cls._conf:
            raise KeyError(
                f"{model_type} model type is not supported yet. Only {list(cls._conf.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )


def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


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
