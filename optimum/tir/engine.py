from logging import getLogger
from os import PathLike
from typing import Union, List, Optional

from transformers import PreTrainedModel, TFPreTrainedModel
from . import TirTarget, TirDispatcher, TirConfig

LOGGER = getLogger("TirEngine")


class TirEngine:

    __slots__ = ("_model", "_target", "_config", "_signatures", "_frontend", "_dispatcher")

    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel, "torch.nn.Module"],
        target: TirTarget,
        config: TirConfig,
        signatures: Optional[List[str]] = None
    ):
        self._model = model
        self._target = target
        self._config = config
        self._signatures = signatures
        self._dispatcher = None

    def __enter__(self) -> 'TirEngine':
        if self._dispatcher is None:
            LOGGER.debug("Creating empty compilation dispatcher.")
            self._dispatcher = TirDispatcher.for_frontend(self._model, self._target, self._config, self._signatures)
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
            self._dispatcher = TirDispatcher.for_frontend(self._model, self._target, self._config, self._signatures)

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



