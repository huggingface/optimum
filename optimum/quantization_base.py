import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Union
from pathlib import Path

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
    def fit(self, output_path: Union[str, Path], file_prefix: Optional[str] = None, **kwargs):
        """Overwrite this method in subclass to define how fit your model for quantization"""
        raise NotImplementedError("Overwrite this method in subclass to define how fit your model for quantization")
