from enum import Enum
from typing import NamedTuple

from iree.compiler import InputType


class TirTarget(Enum):
    INTERPRETED_CPU = "vmvx"
    COMPILED_CPU = "llvm-cpu"
    COMPILED_GPU = "vulkan"
    COMPILED_CUDA = "cuda"
    COMPILED_ROCM = "rocm"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, TirTarget):
            return self.value == other.value
        else:
            return self == other


TirFrontendInfo = NamedTuple("TirFrontendInfo", [("name", str), ("dialect", InputType)])


class TirFrontend(Enum):
    PYTORCH = TirFrontendInfo("pytorch", InputType.TM_TENSOR)
    TENSORFLOW = TirFrontendInfo("tensorflow", InputType.MHLO)
    JAX = TirFrontendInfo("jax", InputType.XLA)
