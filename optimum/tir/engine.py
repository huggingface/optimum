from logging import getLogger
from os import PathLike
from typing import Union

from transformers import is_torch_available, is_tf_available, PreTrainedModel, TFPreTrainedModel
from tir import TirFrontend, TirTarget, TirDispatcher

LOGGER = getLogger("TirEngine")


class TirEngine:

    __slots__ = ("_model", "_target", "_frontend", "_dispatcher")

    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel, "torch.nn.Module"],
        target: TirTarget = TirTarget.COMPILED_CPU,
        export_tf_to_tflite: bool = True
    ):
        self._model = model
        self._target = target
        self._dispatcher = None

        if is_torch_available():
            from torch.nn import Module
            if isinstance(self._model, Module) or isinstance(self._model, PreTrainedModel):
                LOGGER.info(f"TirEngine initializing frontend for PyTorch.")
                self._frontend = TirFrontend.PYTORCH
        elif is_tf_available():
            if export_tf_to_tflite:
                LOGGER.info(f"TirEngine initializing frontend for TFLite.")
                self._frontend = TirFrontend.TFLITE
            else:
                LOGGER.info(f"TirEngine initializing frontend for Tensorflow.")
                self._frontend = TirFrontend.TENSORFLOW
        else:
            LOGGER.error("At least torch or tensorflow needs to be installed.")
            raise ImportError("At elast torch or tensorflow needs to be installed.")

    def __enter__(self) -> 'TirEngine':
        if self._dispatcher is None:
            LOGGER.debug("Creating empty compilation dispatcher.")
            self._dispatcher = TirDispatcher.for_frontend(
                self._frontend,
                self._model,
                self._target,
            )
        else:
            # this branch covers the case
            # with TirEngine.from_precompiled(...) as engine:
            #      engine.compile(...)   # Should not override existing cache
            LOGGER.debug(f"Reusing already provisioned dispatcher ({len(self._dispatcher)} targets).")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dispatcher is not None:
            LOGGER.debug("Destroying compilation dispatcher.")
            self._dispatcher = None

    def __call__(self, *args, **kwargs):
        if self._dispatcher is None:
            LOGGER.debug("Creating empty compilation dispatcher.")
            self._dispatcher = TirDispatcher.for_frontend(self._frontend, self._model, self._target, False)

        return self._dispatcher(*args, **kwargs)

    def compile(self, inputs, use_cache: bool = False):
        LOGGER.debug(f"Compiling model (use_cache = {use_cache}).")
        raise NotImplementedError()

    def save_precompiled(self, path: PathLike):
        LOGGER.debug(f"Saving precompiled engine to: {path}.")
        raise NotImplementedError()

    @staticmethod
    def from_precompiled(path: PathLike) -> "TirEngine":
        LOGGER.debug(f"Attempting to restore precompiled engine from: {path}.")
        raise NotImplementedError()



