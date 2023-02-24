import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


logger = logging.getLogger(__name__)


class OptimumQuantizer(ABC):
    @classmethod
    def from_pretrained(
        cls,
        model_or_path: Union[str, Path],
        file_name: Optional[str] = None,
    ):
        """Overwrite this method in subclass to define how to load your model from pretrained"""
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to load your model from pretrained for quantization"
        )

    @abstractmethod
    def quantize(self, save_dir: Union[str, Path], file_prefix: Optional[str] = None, **kwargs):
        """Overwrite this method in subclass to define how to quantize your model for quantization"""
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to quantize your model for quantization"
        )
