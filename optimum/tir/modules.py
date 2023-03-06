from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union

from iree.compiler.tools import InputType
from tir import TirFrontend


class TirExportableModule(ABC):

    @property
    @abstractmethod
    def module(self) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def framework(self) -> TirFrontend:
        """
        Indicate which framework to export from
        :return:
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def signatures(self) -> List[str]:
        """
        Name of the signatures (methods) to include in the generate binary artifact.
        :return: List of string referencing function's names to export.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def dialect(self) -> InputType:
        """
        The target format (dialect) when exporting the model. This can be for instance MHLO for TensorFlow,
        TM_TENSOR for PyTorch, etc.
        :return:
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def inputs(self) -> Any:
        """
        Provide some example inputs to be able to trace the model.
        :return:
        """
        raise NotImplementedError()

    # @property
    # @abstractmethod
    # def dynamic_axes(self) -> Optional[List[int]]:
    #     """
    #     List of integers (i.e. index) for which we would like to export with unknown (dynamic) axis.
    #     :return: List of integers or None if all the dims are static.
    #     """
    #     raise NotImplementedError()

