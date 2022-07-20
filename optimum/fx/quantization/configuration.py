# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import functools
import importlib
import inspect
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
import torch.ao.quantization.observer as observer
from transformers import __version__ as transformers_version

from ...configuration_utils import BaseConfig
from ...utils.runs import QuantizationApproach
from ...version import __version__


QConfigDict = Dict[str, Any]
PyTorchQuantizationUnit = Union[torch.ao.quantization.ObserverBase, torch.ao.quantization.FakeQuantize]


# TODO: merge that with what is being done for onnxruntime.
class CalibrationMethod(str, Enum):
    MinMax = "minmax"
    MovingAverage = "moving_average"
    Histogram = "histogram"


_METHOD_TO_OBSERVER_MAPPING = {
    (CalibrationMethod.MinMax, False): observer.MinMaxObserver,
    (CalibrationMethod.MinMax, True): observer.PerChannelMinMaxObserver,
    (CalibrationMethod.MovingAverage, False): observer.MovingAverageMinMaxObserver,
    (CalibrationMethod.MovingAverage, True): observer.MovingAveragePerChannelMinMaxObserver,
    (CalibrationMethod.Histogram, False): observer.HistogramObserver,
}
_OBSERVER_TO_METHOD_MAPPING = {v: k for k, v in _METHOD_TO_OBSERVER_MAPPING.items()}

# (symmetric, per_channel) -> torch.qscheme
_QSCHEME_MAPPING = {
    (False, False): torch.per_tensor_affine,
    (True, False): torch.per_tensor_symmetric,
    (False, True): torch.per_channel_affine,
    (True, True): torch.per_channel_symmetric,
}

_REVERSED_QSCHEME_MAPPING = {v: k for k, v in _QSCHEME_MAPPING.items()}


@dataclass
class QConfigUnit:
    """
    Represents the smallest quantization configuration unit.
    It specifies a quantization pattern, it is the optimum counterpart of `torch.ao.quantization.Observer`s and
    `torch.ao.quantization.FakeQuantize`.

    Attributes:
        dtype (`str` or `torch.dtype`, defaults to `torch.quint8`):
            The data type for the quantized values.
        symmetric (`bool`, defaults to `False`):
            Whether the quantization scheme is symmetric or not.
        per_channel (`bool`, defaults to `False`):
            Whether the quantization parameters are computed per channel or not, in which case they are computed per
            tensor.
        ch_axis (`int`, defaults to 0):
            The channel axis if per_channel is `True`.
        quant_min (`int`, *optional*):
            The minimum quantization value.
        quant_max (`int`, *optional*):
            The maximum quantization value.
        calibration_method (`CalibrationMethod`, defaults to `CalibrationMethod.MinMax`):
            The calibration method to use.
        averaging_constant (`float`, defaults to 0.01):
            Averaging constant for min/max, used when `calibration_method=CalibrationMethod.MovingAverage`.
        bins (`int`, defaults to 2048):
            The number of bins to use, used when `calibration_method=CalibrationMethod.Histogram`.
        upsample_rate (`float`, defaults to 128):
            The factor by which the histograms are upsampled, used `calibration_method=CalibrationMethod.Histogram`.
    """

    dtype: Union[str, torch.dtype] = torch.quint8
    symmetric: bool = False
    per_channel: bool = False
    ch_axis: int = 0
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    calibration_method: CalibrationMethod = CalibrationMethod.MinMax
    averaging_constant: float = 0.01
    bins: int = 2048
    upsample_rate: float = 128

    def __post_init__(self):
        # TODO: This is a constraint from PyTorch implementation of FakeQuantize initialization, remove that once possible.
        if (self.quant_min is not None and self.quant_max is None) or (
            self.quant_min is None and self.quant_max is not None
        ):
            raise ValueError(
                "You have to either provide neither quant_min and quant_max or both, but here only one of them was specified."
            )
        # if isinstance(self.dtype, str):
        #     self.dtype = getattr(torch, self.dtype)
        if not isinstance(self.calibration_method, CalibrationMethod):
            self.calibration_method = CalibrationMethod(self.calibration_method)

    def __eq__(self, other):
        """
        Custom re-implementation of __eq__ to make sure that relevant attributes are compared, depending on the
        quantization scheme, and the calibration method.
        For instance, if the calibration method is MinMax, then it does not make sense to compare the upsample_rate
        attributes of self and other, since it will be ignored when converted to PyTorch.

        """
        attributes_to_always_test = ["dtype", "symmetric", "per_channel", "calibration_method"]

        for name in attributes_to_always_test:
            if getattr(self, name) != getattr(other, name):
                return False

        self_quant_min = self.quant_min
        if self_quant_min is None:
            try:
                self_quant_min = torch.iinfo(self.dtype).min
            except TypeError:
                self_quant_min = torch.finfo(self.dtype).min
        other_quant_min = other.quant_min
        if other_quant_min is None:
            try:
                other_quant_min = torch.iinfo(other.dtype).min
            except TypeError:
                other_quant_min = torch.finfo(other.dtype).min

        if self_quant_min != other_quant_min:
            return False

        self_quant_max = self.quant_max
        if self_quant_max is None:
            try:
                self_quant_max = torch.iinfo(self.dtype).max
            except TypeError:
                self_quant_max = torch.finfo(self.dtype).max
        other_quant_max = other.quant_max
        if other_quant_max is None:
            try:
                other_quant_max = torch.iinfo(other.dtype).max
            except TypeError:
                other_quant_max = torch.finfo(other.dtype).max

        if self_quant_max != other_quant_max:
            return False

        if self.per_channel and (self.ch_axis != other.ch_axis):
            return False
        if self.calibration_method == "moving_average" and (self.averaging_constant != other.averaging_constant):
            return False
        if self.calibration_method == "histogram" and (
            self.bins != other.bins or self.upsample_rate != other.upsample_rate
        ):
            return False

        return True

    @classmethod
    def default(
        cls, backend: str = "fbgemm", for_weights: bool = False, for_activations: bool = False
    ) -> "QConfigUnit":
        """
        Args:
            backend (`str`, defaults to `"fbgemm"`):
                The backend that will be used, to be able to choose the best default values. It can either be `"fbgemm"`
                or `"qnnpack"`, please refer to [this section from the PyTorch documentation](https://pytorch.org/docs/stable/quantization.html#backend-hardware-support)
                for more details.
            for_weights (`bool`, defaults to `False`):
                Whether the default QConfigUnit is for the weights or not.
            for_activations (`bool`, defaults to `False`):
                Whether the default QConfigUnit is for the activations or not.

        Returns:
            [`optimum.fx.quantization.QConfigUnit`]:
                The default QConfigUnit for the provided backend, for either the weights or the activations.
        """
        if backend not in ["fbgemm", "qnnpack"]:
            raise ValueError(f'The backend must either be "fbgemm" or "qnnpack", but "{backend}" was provided.')
        if not for_weights and not for_activations:
            raise ValueError("Either for_weights or for_activations should be set to True.")
        elif for_weights and for_activations:
            raise ValueError("Both for_weights or for_activations are set to True, but only one of them should.")
        else:
            qconfig = torch.ao.quantization.get_default_qconfig(backend)
            return cls.from_pytorch(qconfig.weight() if for_weights else qconfig.activation())

    @classmethod
    def from_observer(cls, observer: torch.ao.quantization.ObserverBase) -> "QConfigUnit":
        """
        Args:
            observer (`torch.ao.quantization.ObserverBase`):
                The observer from which the QConfigUnit will be created.

        Returns:
            [`optimum.fx.quantization.QConfigUnit`]:
                A QConfigUnit instantiated from the observer.
        """
        qconfig_unit = cls()
        attributes = {name: getattr(observer, name) for name in dir(observer) if name in qconfig_unit.__dict__}
        calibration_method, per_channel = _OBSERVER_TO_METHOD_MAPPING.get(observer.__class__, ("other", False))
        if observer.__class__ in _OBSERVER_TO_METHOD_MAPPING:
            calibration_method, per_channel = _OBSERVER_TO_METHOD_MAPPING[observer.__class__]
            symmetric, _ = _REVERSED_QSCHEME_MAPPING[observer.qscheme]
            attributes["calibration_method"] = calibration_method
            attributes["per_channel"] = per_channel
            attributes["symmetric"] = symmetric
        return cls(**attributes)

    @classmethod
    def from_fake_quantize(cls, fake_quantize: torch.ao.quantization.FakeQuantizeBase) -> "QConfigUnit":
        """
        Args:
            fake_quantize (`torch.ao.quantization.FakeQuantizeBase`):
                The fake quantize from which the QConfigUnit will be created.

        Returns:
            [`optimum.fx.quantization.QConfigUnit`]:
                A QConfigUnit instantiated from the fake quantize.
        """
        qconfig_unit = cls()
        attributes = {
            name: getattr(fake_quantize, name) for name in dir(fake_quantize) if name in qconfig_unit.__dict__
        }
        qconfig = cls.from_observer(fake_quantize.activation_post_process)
        for name, attr in attributes.items():
            setattr(qconfig, name, attr)
        return qconfig

    @classmethod
    def from_pytorch(
        cls, pytorch_qconfig_unit: Union[Callable[[], PyTorchQuantizationUnit], PyTorchQuantizationUnit]
    ) -> "QConfigUnit":
        """
        Args:
            pytorch_qconfig_unit:
                The (observer or fake quantize) factory function, observer or fake quantize from which the QConfigUnit will be created.

        Returns:
            [`optimum.fx.quantization.QConfigUnit`]:
                A QConfigUnit instantiated from pytorch_qconfig_unit.
        """
        unit = pytorch_qconfig_unit
        if not isinstance(
            unit, (torch.ao.quantization.ObserverBase, torch.ao.quantization.FakeQuantizeBase)
        ) and callable(unit):
            # We cannot simply do: unit = pytorch_qconfig_unit() and then use it to create the QConfigUnit because
            # creating the actual PyTorch QConfig "specializes" some values, such as quant_max, effectively changing
            # what was in the QConfig factory function.
            unit = pytorch_qconfig_unit()
            if not isinstance(unit, (torch.ao.quantization.ObserverBase, torch.ao.quantization.FakeQuantizeBase)):
                for name, value in pytorch_qconfig_unit.p.keywords.items():
                    setattr(unit, name, value)

        if isinstance(unit, torch.ao.quantization.ObserverBase):
            return cls.from_observer(unit)
        return cls.from_fake_quantize(unit)

    @staticmethod
    def _observers_mapping(
        calibration_method: Optional[CalibrationMethod] = None,
        per_channel: Optional[bool] = None,
        observer_class: Optional[Type] = None,
    ) -> Union[Type, Tuple[CalibrationMethod, bool]]:
        """
        Maps observer classes to their corresponding (calibration_method, per_channel) pair.
        2 cases:
            1. observer_class is provided. In this case, the corresponding (calibration_method, per_channel) pair is returned.
            2. A (calibration_method, per_channel) pair is provided. In this case an observer class is returned.
        """
        if observer_class is not None:
            result = _OBSERVER_TO_METHOD_MAPPING[observer_class]
        elif calibration_method is not None and per_channel is not None:
            result = _METHOD_TO_OBSERVER_MAPPING[(calibration_method, per_channel)]
        else:
            raise ValueError(
                f"Could not map with calibration_method = {calibration_method}, per_channel = {per_channel} and observer_class = {observer_class}"
            )
        return result

    def get_observer_kwargs(self, observer_class: Type) -> Dict[str, Any]:
        """
        Retrieves the proper arguments from self in order to create an instance of observer_class.

        Args:
            observer_class (`Type`):
                The class of the observer that will be created.

        Returns:
            `Dict[str, Any]`:
                The keyword arguments to provide when instantiating the observer.
        """
        allowed_parameters = set(inspect.signature(observer_class.__init__).parameters.keys())
        kwargs = {name: attr for name, attr in self.__dict__.items() if name in allowed_parameters}
        kwargs["qscheme"] = _QSCHEME_MAPPING[(self.symmetric, self.per_channel)]
        return kwargs

    def get_observer_info(self) -> Tuple[Type, Dict[str, Any]]:
        """
        Retrieves both the class and the keyword arguments of the observer / fake quantize to create.
        """
        observer_class = QConfigUnit._observers_mapping(
            calibration_method=self.calibration_method, per_channel=self.per_channel
        )
        if observer_class is None:
            raise KeyError(
                f"Could not find an observer matching calibration_method = {self.calibration_method} with per_channel = {self.per_channel}."
            )
        observer_kwargs = self.get_observer_kwargs(observer_class)
        return observer_class, observer_kwargs

    def as_observer(
        self, as_factory: bool = True
    ) -> Union[Callable[[], torch.ao.quantization.ObserverBase], torch.ao.quantization.ObserverBase]:
        """
        Args:
            as_factory (`bool`, defaults to `True`):
                Whether the factory function for instantiating observers should be returned instead of returning a
                concrete instance of the observer.

        Returns:
            The factory or an instance of the observer.
        """
        observer_class, observer_kwargs = self.get_observer_info()
        return observer_class.with_args(**observer_kwargs) if as_factory else observer_class(**observer_kwargs)

    def as_fake_quantize(
        self, as_factory: bool = True
    ) -> Union[Callable[[], torch.ao.quantization.FakeQuantizeBase], torch.ao.quantization.FakeQuantizeBase]:
        """
        Args:
            as_factory (`bool`, defaults to `True`):
                Whether the factory function for instantiating fake quantizes should be returned instead of returning a
                concrete instance of the fake quantize.

        Returns:
            The factory or an instance of the fake quantize.
        """
        observer_class, observer_kwargs = self.get_observer_info()
        quant_min = observer_kwargs.pop("quant_min")
        if quant_min is None:
            quant_min = observer_class(dtype=self.dtype).quant_min
        quant_max = observer_kwargs.pop("quant_max")
        if quant_max is None:
            quant_max = observer_class(dtype=self.dtype).quant_max
        fake_quantize_cls = torch.ao.quantization.FakeQuantize
        if self.calibration_method == "moving_average" and self.per_channel and self.ch_axis == 0:
            fake_quantize_cls = torch.ao.quantization.FusedMovingAvgObsFakeQuantize
        partial_fn = fake_quantize_cls.with_args(
            observer=observer_class,
            quant_min=quant_min,
            quant_max=quant_max,
            **observer_kwargs,
        )
        return partial_fn if as_factory else partial_fn()


@dataclass
class QConfig:
    """
    Represents the quantization config to apply to either weights or activations.
    It is the optimum counterpart of `torch.ao.quantization.QConfig`.

    Attributes:
        activation ([`~optimum.fx.quantization.QConfigUnit`], *optional*):
            The [`~optimum.fx.quantization.QConfigUnit`] specifying the configuration for the activations.
        weight ([`~optimum.fx.quantization.QConfigUnit`], *optional*):
            The [`~optimum.fx.quantization.QConfigUnit`] specifying the configuration for the weights.
    """

    activation: Optional[QConfigUnit] = None
    weight: Optional[QConfigUnit] = None

    def __post_init__(self):
        if isinstance(self.activation, dict):
            self.activation = QConfigUnit(**self.activation)
            self.weight = QConfigUnit(**self.weight)
        if self.activation is None:
            self.activation = QConfigUnit()
        if self.weight is None:
            self.weight = QConfigUnit()

    @classmethod
    def default(cls, backend: str = "fbgemm"):
        return cls(
            activation=QConfigUnit.default(backend=backend, for_activations=True),
            weight=QConfigUnit.default(backend=backend, for_weights=True),
        )

    @classmethod
    def from_pytorch(cls, pytorch_qconfig: torch.ao.quantization.QConfig) -> "QConfig":
        """
        Args:
            pytorch_qconfig:
                The PyTorch QConfig from which the QConfig will be created.

        Returns:
            [`optimum.fx.quantization.QConfig`]:
                A QConfig instantiated from pytorch_qconfig.
        """
        return cls(
            activation=QConfigUnit.from_pytorch(pytorch_qconfig.activation),
            weight=QConfigUnit.from_pytorch(pytorch_qconfig.weight),
        )

    def to_pytorch(self, quantization_approach: QuantizationApproach) -> torch.ao.quantization.QConfig:
        """
        Args:
            quantization approach ([`optimum.utils.runs.QuantizationApproach`]):
                The quantization approach for which you want the PyTorch config.

        Returns:
            `torch.ao.quantization.QConfig`:
                A PyTorch QConfig instantiated with the proper attributes for the quantization approach.
        """
        if quantization_approach is QuantizationApproach.qat:
            qconfig = torch.ao.quantization.QConfig(
                activation=self.activation.as_fake_quantize(),
                weight=self.weight.as_fake_quantize(),
            )
        else:
            qconfig = torch.ao.quantization.QConfig(
                activation=self.activation.as_observer(),
                weight=self.weight.as_observer(),
            )
        return qconfig


def handle_tuples(validate_func):
    @functools.wraps(validate_func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], (list, tuple)):
            for x in args[0]:
                validate_func(x, *args[1:], **kwargs)
        else:
            validate_func(*args, **kwargs)

    return wrapper


class QuantizationConfig(BaseConfig):
    """
    Represents the general quantization configuration.
    It specifies which object types and modules should be quantized by providing an [`optimum.fx.quantization.QConfig`] for each.
    It is the optimum counterpart of PyTorch qconfig_dict.
    """

    CONFIG_NAME = "fx_quantization_config.json"
    FULL_CONFIGURATION_FILE = "fx_quantization_config.json"

    def __init__(self, **kwargs):
        def process(attr):
            def map_fn(item):
                if isinstance(item, dict):
                    return QConfig(**item)
                elif isinstance(item, str) and item.startswith("torch"):
                    module_name, attr_name = item.rsplit(".", maxsplit=1)
                    mod = importlib.import_module(module_name)
                    return getattr(mod, attr_name)

                return item

            return QuantizationConfig._map(map_fn, attr, dicts_are_leafs=False)

        self._global = process(kwargs.pop("global", None))
        self.object_type = process(kwargs.pop("object_type", {}))
        self.module_name = process(kwargs.pop("module_name", {}))
        self.module_name_regex = process(kwargs.pop("module_name_regex", {}))
        self.module_name_object_type_order = process(kwargs.pop("module_name_object_type_order", {}))

        # Make sure that the loaded attributes are stored as dictionaries.
        self._transform_attributes(to_dict=True, inplace=True)

    @classmethod
    def default(cls, backend: str = "fbgemm") -> "QuantizationConfig":
        """
        Args:
            backend (`str`, defaults to `"fbgemm"`):
                The backend that will be used, to be able to choose the best default values. It can either be `"fbgemm"`
                or `"qnnpack"`, please refer to [this section from the PyTorch documentation](https://pytorch.org/docs/stable/quantization.html#backend-hardware-support)
                for more details.

        Returns:
            [`optimum.fx.quantization.QuantizationConfig`]:
                The default QuantizationConfig for the provided backend.
        """
        if backend not in ["fbgemm", "qnnpack"]:
            raise ValueError(f'The backend must either be "fbgemm" or "qnnpack", but "{backend}" was provided.')
        return cls.from_pytorch(torch.ao.quantization.get_default_qconfig_dict(backend))

    @staticmethod
    def _map(func, item, dicts_are_leafs=True):
        if item is None:
            return None
        elif not isinstance(item, torch.ao.quantization.QConfig) and isinstance(item, (list, tuple)):
            return type(item)([QuantizationConfig._map(func, x, dicts_are_leafs=dicts_are_leafs) for x in item])
        elif not dicts_are_leafs and isinstance(item, dict):
            return {k: QuantizationConfig._map(func, v, dicts_are_leafs=dicts_are_leafs) for k, v in item.items()}
        else:
            return func(item)

    def _list_to_dict(self, list_):
        if isinstance(list_, dict):
            return list_
        return {t[0] if len(t) == 2 else t[:-1]: t for t in list_}

    def _dict_to_list(self, dict_):
        if isinstance(dict_, list):
            return dict_
        return list(dict_.values())

    def _transform_attributes(self, to_list=False, to_dict=False, inplace=False):
        """
        Transforms the underlying attributes (object_type, module_name, etc):
            - Dict -> List if to_list is True
            - List -> Dict if to_dict is True
        It is useful to go from the internal representation (dictionaries) to the user-facing representation (list,
        matching what is done in PyTorch).
        """
        if to_list and to_dict:
            raise ValueError('Only one of "to_list" and "to_dict" can be set to True.')
        transform_function = self._list_to_dict if to_dict else self._dict_to_list
        attribute_names = ["object_type", "module_name", "module_name_regex", "module_name_object_type_order"]
        transformed_attributes = {k: transform_function(getattr(self, k)) for k in attribute_names}
        if inplace:
            self.__dict__.update(transformed_attributes)
        return transformed_attributes

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output.update(self._transform_attributes(to_list=True))
        output["global"] = output.pop("_global")

        def qconfig_to_dict(item):
            if isinstance(item, QConfig):
                return asdict(item)
            return item

        # Find a better way (embedded in QConfigUnit?)
        def torch_dtype_to_string(item):
            if isinstance(item, torch.dtype):
                return str(item).split(".")[1]
            return item

        def torch_class_or_function_to_string(item):
            try:
                mod = inspect.getmodule(item)
                if mod.__name__.startswith("torch"):
                    return ".".join((mod.__name__, item.__name__))
            except BaseException:
                pass
            return item

        output = self._map(qconfig_to_dict, output, dicts_are_leafs=False)
        output = self._map(torch_dtype_to_string, output, dicts_are_leafs=False)
        output = self._map(torch_class_or_function_to_string, output, dicts_are_leafs=False)

        # Transformers version when serializing the model
        output["transformers_version"] = transformers_version
        output["optimum_version"] = __version__

        return output

    @classmethod
    def from_pytorch(cls, qconfig_dict: Dict[str, Any]) -> "QuantizationConfig":
        """
        Args:
            qconfig_dict (`Dict[str, Any]`):
                The PyTorch qconfig dict to use to create the QuantizationConfig.

        Returns:
            [`optimum.fx.quantization.QuantizationConfig`]:
                A QuantizationConfig instantiated from qconfig_dict.
        """
        clone = copy.deepcopy(qconfig_dict)
        clone["global"] = clone.pop("", None)

        def map_fn(item):
            if isinstance(item, torch.ao.quantization.QConfig):
                return QConfig.from_pytorch(item)
            return item

        clone = cls._map(map_fn, clone, dicts_are_leafs=False)
        return cls(**clone)

    def to_pytorch(self, quantization_approach: QuantizationApproach):
        """
        Args:
            quantization approach ([`optimum.utils.runs.QuantizationApproach`]):
                The quantization approach for which you want the PyTorch config.

        Returns:
            `Dict[str, Any]`:
                A qconfig dict that can be used with the PyTorch API for the specified quantization approach.
        """

        def to_pytorch(item):
            def cast_fn(x):
                if isinstance(x, QConfig):
                    return x.to_pytorch(quantization_approach)
                return x

            return self._map(cast_fn, item)

        attributes = {
            attr_name: to_pytorch(attr) for attr_name, attr in self._transform_attributes(to_list=True).items()
        }
        return {"": to_pytorch(self.global_), **attributes}

    def summary(self):
        """
        Prints the summary of the QuantizationConfig: what is being quantized, and how.
        """

        summary_list = ["*** Summary ***"]
        # Global
        global_qconfig_str = f"\t{self.global_}" if self.global_ is not None else "None"
        summary_list += ["Global:", global_qconfig_str, ""]

        # Object type
        if self.object_type:
            summary_list.append("Object type:")
            for object_type, t in self.object_type.items():
                summary_list.append(f"\t{object_type} => \n\t{str(t[1])}")
                summary_list.append("")

        # Module name
        if self.module_name:
            summary_list.append("Module name:")
            for module_name, t in self.module_name.items():
                summary_list.append(f"\t{module_name} => \n\t{str(t[1])}")
                summary_list.append("")

        # Module name regex
        if self.module_name:
            summary_list.append("Module name regex:")
            for module_name_regex, t in self.module_name_regex.items():
                summary_list.append(f'\t"{module_name_regex}" => \n\t{str(t[1])}')
                summary_list.append("")

        # Module name regex
        if self.module_name_object_type_order:
            summary_list.append("Module name object type order:")
            for key, t in self.module_name_regex.items():
                summary_list.append(f'\t"{key}" => \n\t{str(t[-1])}')
                summary_list.append("")

        print("\n".join(summary_list))

    @property
    def global_(self):
        return self._global

    @staticmethod
    @handle_tuples
    def _validate_qconfig_type(qconfig):
        if qconfig is not None and not isinstance(qconfig, QConfig):
            raise TypeError(
                f"The specified qconfig must either be None or a QConfig, but an object of type {type(qconfig)} was provided here."
            )

    @staticmethod
    @handle_tuples
    def _validate_object_type_type(object_type):
        if not isinstance(object_type, (type, str)) and not callable(object_type):
            raise TypeError(
                f"The object type must either be a class or a function, but an object of type {type(object_type)} was provided here."
            )

    @staticmethod
    @handle_tuples
    def _validate_module_name(module_name, attr_name):
        if not isinstance(module_name, str):
            raise TypeError(
                f"The {attr_name} must be a str, but an object of type {type(module_name)} was provided here."
            )

    @staticmethod
    @handle_tuples
    def _validate_index(index):
        if not isinstance(index, int):
            raise TypeError(f"The index must be an int, but an object of type {type(index)} was provided here.")

    @global_.setter
    def global_(self, qconfig: Optional[QConfig]):
        """
        Sets the quantization config to use globally.

        Args:
            qconfig ([`optimum.fx.quantization.QConfig`], *optional*):
                The quantization config to apply globally.
        """
        self._validate_qconfig_type(qconfig)
        self._global = qconfig

    def add_object_type(
        self, object_type: Union[str, Type, Callable, Tuple[Union[str, Type, Callable]]], qconfig: Optional[QConfig]
    ):
        """
        Sets the quantization config to use for a given object type.

        Args:
            object_type (`str`, `Type`, `Callable`, or tuple of those types):
                The object type to which the config will be applied.

            qconfig ([`optimum.fx.quantization.QConfig`], *optional*):
                The quantization config.
        """
        self._validate_object_type_type(object_type)
        self._validate_qconfig_type(qconfig)
        self.object_type[object_type] = (object_type, qconfig)

    def remove_object_type(self, object_type: Union[torch.nn.Module, Callable]):
        """
        Removes the quantization config to use for a given object type.
        Does nothing if no quantization config was provided for the object type before.

        Args:
            object_type (`str`, `Type`, `Callable`, or tuple of those types):
                The object type for which the config should be removed.
        """
        self._validate_object_type_type(object_type)
        self.object_type.pop(object_type, None)

    def add_module_name(self, module_name: str, qconfig: Optional[QConfig]):
        """
        Sets the quantization config to use for a module name.

        Args:
            module_name (`str`):
                The module name to which the config will be applied.

            qconfig ([`optimum.fx.quantization.QConfig`], *optional*):
                The quantization config.
        """
        self._validate_module_name(module_name, "module_name")
        self._validate_qconfig_type(qconfig)
        self.module_name[module_name] = (module_name, qconfig)

    def remove_module_name(self, module_name: str):
        """
        Removes the quantization config to use for a given module name.
        Does nothing if no quantization config was provided for the module name before.

        Args:
            module_name (`str`):
                The module name for which the config should be removed.
        """
        self._validate_module_name(module_name, "module_name")
        self.module_name.pop(module_name, None)

    def add_module_name_regex(self, module_name_regex: str, qconfig: Optional[QConfig]):
        """
        Sets the quantization config to use for a module name regex.
        Every module with a name matching the pattern will use the config.

        Args:
            module_name_regex (`str`):
                The module name regex specifying to which module the config will be applied.

            qconfig ([`optimum.fx.quantization.QConfig`], *optional*):
                The quantization config.
        """
        self._validate_module_name(module_name_regex, "module_name_regex")
        self._validate_qconfig_type(qconfig)
        self.module_name_regex[module_name_regex] = (module_name_regex, qconfig)

    def remove_module_name_regex(self, module_name_regex: str):
        """
        Removes the quantization config to use for a given module name regex.
        Does nothing if no quantization config was provided for the module name regex before.

        Args:
            module_name_regex (`str`):
                The module name regex for which the config should be removed.
        """
        self._validate_module_name(module_name_regex, "module_name_regex")
        self.module_name_regex.pop(module_name_regex, None)

    def add_module_name_object_type_order(
        self,
        module_name_regex: str,
        object_type: Union[torch.nn.Module, Callable],
        index: int,
        qconfig: Optional[QConfig],
    ):
        """
        Sets the quantization config to use for a (module_name_regex, object type, index) tuple.

        Args:
            module_name_regex (`str`):
                The module name regex specifying to which module the config will be applied.

            object_type (`str`, `Type`, `Callable`, or tuple of those types):
                The object type of the submodule inside the matched module by module_name_regex.

            index (`int`):
                The index of the submodule.

            qconfig ([`optimum.fx.quantization.QConfig`], *optional*):
                The quantization config.
        """
        self._validate_module_name(module_name_regex, "module_name_regex")
        self._validate_object_type_type(object_type)
        self._validate_index(index)
        self._validate_qconfig_type(qconfig)
        self.module_name_object_type_order[(module_name_regex, object_type, index)] = (
            module_name_regex,
            object_type,
            index,
            qconfig,
        )

    def remove_module_name_object_type_order(
        self,
        module_name_regex: str,
        object_type: Union[str, Type, Callable, Tuple[Union[str, Type, Callable]]],
        index: str,
    ):
        """
        Removes the quantization config to use for a (module_name_regex, object type, index) tuple.
        Does nothing if no quantization config was provided for the (module_name_regex, object_type, index) tuple before.

        Args:
            module_name_regex (`str`):
                The module name regex specifying to which module the config will be applied.

            object_type (`str`, `Type`, `Callable`, or tuple of those types):
                The object type of the submodule inside the matched module by module_name_regex.

            index (`int`):
                The index of the submodule.
        """
        self._validate_object_type_type(object_type)
        self._validate_module_name(module_name_regex, "module_name_regex")
        self._validate_index(index)
        self.module_name_object_type_order.pop((module_name_regex, object_type, index), -1)

    def get_quantizable_nodes(self, model):
        raise NotImplementedError
