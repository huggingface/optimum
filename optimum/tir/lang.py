from enum import Enum
from typing import NamedTuple

from iree.compiler import InputType


class TirTarget(str, Enum):
    INTERPRETED_CPU = "vmvx"
    COMPILED_CPU = "llvm-cpu"
    COMPILED_GPU = "vulkan"
    COMPILED_CUDA = "cuda"
    COMPILED_ROCM = "rocm"


TirFrontendInfo = NamedTuple("TirFrontendInfo", [("name", str), ("dialect", InputType)])


class TirFrontend(Enum):
    PYTORCH = TirFrontendInfo("pytorch", InputType.TM_TENSOR)
    TENSORFLOW = TirFrontendInfo("tensorflow", InputType.MHLO)
    TFLITE = TirFrontendInfo("tflite", InputType.TOSA)
    JAX = TirFrontendInfo("jax", InputType.XLA)
