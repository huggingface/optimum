from optimum.version import __version__

_TIR_HUB_USER_AGENT = f"huggingface/tir:{__version__}"

from .lang import TirConfig, TirConfigStore, TirFrontend, TirTarget
from .dispatch import TirDispatcher
from .engine import TirEngine
