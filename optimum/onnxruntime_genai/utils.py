from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import torch
from transformers.utils import logging
import subprocess

import onnx
import onnxruntime as ort
from onnx import ModelProto


logger = logging.get_logger(__name__)

ONNX_WEIGHTS_NAME="model.onnx"
OPTIMIZED_ONNX_WEIGHTS_NAME = "optimized_model.onnx"
QUANTIZED_ONNX_WEIGHTS_NAME = "q8_model.onnx"


def _is_gpu_available():
    """
    checks if a gpu is available.
    Note: Currently only CUDA and CPU are supported
    """

    IS_NVIDIA_SYSTEM = False

    try:
        subprocess.run(["nvidia-smi"], check=True)
        IS_NVIDIA_SYSTEM = True
    except Exception:
        pass

    if IS_NVIDIA_SYSTEM and torch.cuda.is_available():
        return True
    else:
        return False
