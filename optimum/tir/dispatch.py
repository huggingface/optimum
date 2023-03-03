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
    __slots__ = ("_model", "_target", "_config", "_executor", "_cache", "_parameters")

    @staticmethod
    def for_frontend(model, target: TirTarget, config: TirConfig) -> "TirDispatcher":
        if is_torch_available():
            from torch.nn import Module
            from .pytorch import TorchDispatcher

            if isinstance(model, Module) or isinstance(model, PreTrainedModel):
                LOGGER.info(f"TirDispatcher initializing frontend for PyTorch.")
                return TorchDispatcher(model, target, config)

        if is_tf_available():
            from tensorflow.types.experimental import ConcreteFunction, GenericFunction
            from .tensorflow import TensorflowDispatcher
            if isinstance(model, (str, TFPreTrainedModel, ConcreteFunction, GenericFunction)):
                LOGGER.info(f"TirDispatcher initializing frontend for TensorFlow.")
                return TensorflowDispatcher(model, target, config)

        LOGGER.error("At least torch or tensorflow needs to be installed.")
        raise ImportError(f"Unsupported model type {type(model)}. Only pytorch and tensorflow are supported")

    def __init__(self, model, target: TirTarget, config: TirConfig):
        self._model = model
        self._target = target
        self._config = config
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
            inferred_compiler_args = self._config.get_compiler_args()
            dispatch = self.compile_from_mlir(exported_module, self._target, None, inferred_compiler_args)
            self._cache[key] = dispatch

        # TODO : Remove this dict because it's TensorFlow only.
        return self._internal_call(dispatch, curated_args)

    @abstractmethod
    def export_model_to_mlir(self, model, target: TirTarget, examples: Optional = None, dynamic_axes: List[int] = None):
        raise NotImplementedError()

    @abstractmethod
    def compile_from_mlir(self, mlir, target: TirTarget, compiler_args: List[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def validate_forward_inputs(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _internal_call(self, dispatch: Callable, curated_args):
        raise NotImplementedError()

