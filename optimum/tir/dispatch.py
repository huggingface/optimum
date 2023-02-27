import os
from abc import ABC, abstractmethod
from functools import singledispatch
from logging import getLogger
from typing import Any, List, Optional, Callable, Union
from transformers import PreTrainedModel, TFPreTrainedModel, is_torch_available, is_tf_available
from . import TirFrontend, TirTarget, TirConfig


LOGGER = getLogger("TirDispatcher")


class TirDispatcher(ABC):

    """
    The registry is an associative storage which keeps track all the succeeded compilations.
    """
    __slots__ = ("_model", "_target", "_executor", "_cache", "_parameters")

    @staticmethod
    def for_frontend(model, target: TirTarget, config: TirConfig) -> "TirDispatcher":
        if is_torch_available():
            from torch.nn import Module
            from .pytorch import TorchDispatcher

            if isinstance(model, Module) or isinstance(model, PreTrainedModel):
                LOGGER.info(f"TirDispatcher initializing frontend for PyTorch.")
                return TorchDispatcher(model, target, config)
        elif is_tf_available():
            from .tensorflow import TensorflowDispatcher
            if isinstance(model, TFPreTrainedModel):
                LOGGER.info(f"TirDispatcher initializing frontend for TensorFlow.")
                return TensorflowDispatcher(model, target, config)

        LOGGER.error("At least torch or tensorflow needs to be installed.")
        raise ImportError(f"Unsupported model type {type(model)}. Only pytorch and tensorflow are supported")

    def __init__(self, model, target: TirTarget):
        self._model = model
        self._target = target
        self._executor = None
        self._validator = None
        self._cache = dict()

    def __len__(self):
        """
        Number of precompiled graphs in the cache.

        Returns:
            Number of precompiled graphs currently present in the cache.
        """
        return len(self._targets)

    def __call__(self, *args, **kwargs) -> Any:
        curated_args = self.validate_forward_inputs(*args, **kwargs)
        key = (len(curated_args), ) + tuple(curated_args[0].shape)

        if key in self._cache:
            LOGGER.debug(f"Cache hit for dispatch key: {key}.")
            dispatch = self._cache[key]
        else:
            LOGGER.debug(f"Cache miss for dispatch key: {key}.")
            exported_module = self.export_model_to_mlir(self._model, self._target, curated_args)
            inferred_compiler_args = self.compiler_args_for_target(exported_module)
            dispatch = self.compile_from_mlir(exported_module, self._target, None, inferred_compiler_args)
            self._cache[key] = dispatch

        # TODO : Remove this dict because it's TensorFlow only.
        return self._internal_call(dispatch, curated_args)

    def compiler_args_for_target(self, mlir_module) -> List[str]:
        extra_args = [
            "--compile-mode=std",
            "--cost-kind=latency",
            "--iree-opt-const-eval",
            "--iree-opt-const-expr-hoisting",
            "--iree-opt-strip-assertions",
            "--iree-opt-numeric-precision-reduction"
        ]

        if "OPTIMUM_TIR_MLIR_PRINT" in os.environ and os.environ["OPTIMUM_TIR_MLIR_PRINT"] in {"1", "ON"}:
            extra_args += [
                # "--mlir-print-elementsattrs-with-hex-if-larger=100000",
                # "--mlir-print-ir-before-all",
            ]

        if self._target == TirTarget.COMPILED_CPU:
            extra_args += [
                "--iree-llvm-target-cpu-features=host",
                "--iree-mhlo-demote-i64-to-i32=false",
                "--iree-stream-resource-index-bits=64",
                "--iree-vm-target-index-bits=64",
                # "--iree-llvm-target-float-abi=hard",
                # "--iree-llvm-slp-vectorization",
                # "--iree-llvm-loop-interleaving",
                # "--iree-llvm-loop-unrolling",
                # "--iree-llvm-loop-vectorization",
                # "--iree-flow-demote-i64-to-i32",
                # "--iree-flow-enable-linalg-detensorize",
            ]
        elif self._target is TirTarget.COMPILED_CUDA:
            extra_args += [
                "--iree-hal-cuda-disable-loop-nounroll-wa",
            ]

            # Enable tensor core?
            # "--iree-flow-demote-f32-to-f16"
        return extra_args

    @abstractmethod
    def export_model_to_mlir(self, model, target: TirTarget, examples: Optional = None, dynamic_axes: List[int] = None):
        raise NotImplementedError()

    @abstractmethod
    def compile_from_mlir(self, mlir, target: TirTarget, device: Optional[str] = None, compiler_args: List[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def validate_forward_inputs(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _internal_call(self, dispatch: Callable, curated_args):
        raise NotImplementedError()

