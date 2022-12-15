from abc import ABC, abstractmethod
from os import PathLike
from typing import List, Optional, Union, Any

from iree._runtime import HalDevice
from transformers import BatchEncoding
from transformers.utils.logging import get_logger

from tir import TirFrontend, TirTarget


class TirDispatcher(ABC):
    LOGGER = get_logger("TirDispatcher")

    """
    The registry is an associative storage which keeps track all the succeeded compilations.
    """
    __slots__ = ("_model", "_target", "_executor", "_cache", "_parameters")

    @staticmethod
    def for_frontend(frontend: TirFrontend, model, target: TirTarget, use_tflite_for_tf: bool) -> "TirDispatcher":
        if frontend == TirFrontend.PYTORCH:
            from .pytorch import TorchDispatcher
            return TorchDispatcher(model, target)
        elif frontend in {TirFrontend.TENSORFLOW, TirFrontend.TFLITE}:
            from .tensorflow import TensorflowDispatcher
            return TensorflowDispatcher(model, target, use_tflite_for_tf)
        else:
            raise NotImplementedError()

    @abstractmethod
    def export_model_to_mlir(self, model, target: TirTarget, examples: Optional = None, dynamic_axes: List[int] = None):
        raise NotImplementedError()

    @abstractmethod
    def compile_from_mlir(
        self,
        mlir: Union[bytes, str],
        target: TirTarget,
        device: Optional[HalDevice] = None
    ):
        raise NotImplementedError()

    @abstractmethod
    def validate_forward_inputs(self, *args, **kwargs):
        raise NotImplementedError()

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
            TirDispatcher.LOGGER.debug(f"Cache hit for dispatch key: {key}.")
            dispatch = self._cache[key]
        else:
            TirDispatcher.LOGGER.debug(f"Cache miss for dispatch key: {key}.")
            exported_module = self.export_model_to_mlir(self._model, self._target, curated_args)
            dispatch = self.compile_from_mlir(exported_module, self._target)
            self._cache[key] = dispatch

        # TODO : Remove this dict because it's TensorFlow only.
        # return dispatch({"input_ids": curated_args[0], "attention_mask": curated_args[1]})
        return dispatch(*curated_args)


class TirEngine:

    LOGGER = get_logger("TirEngine")

    __slots__ = ("_model", "_target", "_frontend", "_dispatcher", "_export_tf_to_tflite")

    def __init__(
        self,
        model: Union["transformers.PreTrainedModel", "transformers.TFPreTrainedModel"],
        target: TirTarget = TirTarget.COMPILED_CPU,
        export_tf_to_tflite: bool = True
    ):
        self._model = model
        self._target = target
        self._dispatcher = None
        self._export_tf_to_tflite = export_tf_to_tflite

        if self._model.framework == "pt":
            TirEngine.LOGGER.info(f"TirEngine initializing frontend for PyTorch.")
            self._frontend = TirFrontend.PYTORCH
        else:
            if export_tf_to_tflite:
                TirEngine.LOGGER.info(f"TirEngine initializing frontend for TFLite.")
                self._frontend = TirFrontend.TFLITE
            else:
                TirEngine.LOGGER.info(f"TirEngine initializing frontend for Tensorflow.")
                self._frontend = TirFrontend.TENSORFLOW

    def __enter__(self) -> 'TirEngine':
        if self._dispatcher is None:
            TirEngine.LOGGER.debug("Creating empty compilation dispatcher.")
            self._dispatcher = TirDispatcher.for_frontend(
                self._frontend,
                self._model,
                self._target,
                self._export_tf_to_tflite
            )
        else:
            # this branch covers the case
            # with TirEngine.from_precompiled(...) as engine:
            #      engine.compile(...)   # Should not override existing cache
            TirEngine.LOGGER.debug(f"Reusing already provisioned dispatcher ({len(self._dispatcher)} targets).")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dispatcher is not None:
            TirEngine.LOGGER.debug("Destroying compilation dispatcher.")
            self._dispatcher = None

    def __call__(self, *args, **kwargs):
        if self._dispatcher is None:
            TirEngine.LOGGER.debug("Creating empty compilation dispatcher.")
            self._dispatcher = TirDispatcher.for_frontend(self._frontend, self._model, self._target)

        return self._dispatcher(*args, **kwargs)

    def compile(self, inputs: BatchEncoding, use_cache: bool = False):
        TirEngine.LOGGER.debug(f"Compiling model (use_cache = {use_cache}).")
        raise NotImplementedError()

    def save_precompiled(self, path: PathLike):
        TirEngine.LOGGER.debug(f"Saving precompiled engine to: {path}.")
        raise NotImplementedError()

    @staticmethod
    def from_precompiled(path: PathLike) -> "TirEngine":
        TirEngine.LOGGER.debug(f"Attempting to restore precompiled engine from: {path}.")
        raise NotImplementedError()



