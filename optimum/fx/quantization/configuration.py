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
import inspect
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union

import torch
import torch.ao.quantization.observer as observer
from transformers import __version__ as transformers_version

from ...configuration_utils import BaseConfig
from ...utils.runs import QuantizationApproach
from ...version import __version__


QConfigDict = Dict[str, Any]


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
    (CalibrationMethod.Histogram, True): observer.HistogramObserver,
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
    dtype: Union[str, torch.dtype] = torch.qint8
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
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

    @classmethod
    def default(cls, backend: str, for_weights: bool = False, for_activations: bool = False) -> "QConfigUnit":
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
    def from_fake_quantize(cls, fake_quantize: torch.ao.quantization.FakeQuantize) -> "QConfigUnit":
        qconfig_unit = cls()
        attributes = {name: getattr(observer, name) for name in dir(observer) if name in qconfig_unit.__dict__}
        if fake_quantize.observer in _OBSERVER_TO_METHOD_MAPPING:
            calibration_method, per_channel = _OBSERVER_TO_METHOD_MAPPING[fake_quantize.observer]
            symmetric, _ = _REVERSED_QSCHEME_MAPPING[fake_quantize.observer(**fake_quantize.observer_kwargs).qscheme]
            attributes["calibration_method"] = calibration_method
            attributes["per_channel"] = per_channel
            attributes["symmetric"] = symmetric
        else:
            warnings.warn(f"The observer class {fake_quantize.observer} is not supported, feel free to ignore")
        # if per_channel != attributes["per_channel"]:
        #     warnings.warn("The FakeQuantize per channel attribute does not match the inferred one, using the one from FakeQuantize")
        #     per_channel = attributes["per_channel"]
        # Ignoring per_channel since it was already retrieved.
        return cls(**attributes)

    @classmethod
    def from_observer(cls, observer: torch.ao.quantization.ObserverBase) -> "QConfigUnit":
        qconfig_unit = cls()
        attributes = {name: getattr(observer, name) for name in dir(observer) if name in qconfig_unit.__dict__}
        calibration_method, per_channel = _OBSERVER_TO_METHOD_MAPPING.get(observer.__class__, ("other", False))
        if observer.__class__ in _OBSERVER_TO_METHOD_MAPPING:
            calibration_method, per_channel = _OBSERVER_TO_METHOD_MAPPING[observer.__class__]
            symmetric, _ = _REVERSED_QSCHEME_MAPPING[observer.qscheme]
            attributes["calibration_method"] = calibration_method
            attributes["per_channel"] = per_channel
            attributes["symmetric"] = symmetric
        # if per_channel != attributes["per_channel"]:
        #     warnings.warn("The FakeQuantize per channel attribute does not match the inferred one, using the one from FakeQuantize")
        #     per_channel = attributes["per_channel"]
        # Ignoring per_channel since it was already retrieved.
        return cls(**attributes)

    @classmethod
    def from_pytorch(
        cls, pytorch_qconfig_unit: Union[torch.ao.quantization.ObserverBase, torch.ao.quantization.FakeQuantize]
    ) -> "QConfigUnit":
        unit = pytorch_qconfig_unit()
        if isinstance(unit, torch.ao.quantization.ObserverBase):
            return cls.from_observer(unit)
        return cls.from_fake_quantize(unit)

    @staticmethod
    def _observers_mapping(
        calibration_method: Optional[CalibrationMethod] = None,
        per_channel: Optional[bool] = None,
        observer_class: Optional[Type] = None,
    ):
        if observer_class is not None:
            result = _OBSERVER_TO_METHOD_MAPPING[observer_class]
        elif calibration_method is not None and per_channel is not None:
            result = _METHOD_TO_OBSERVER_MAPPING[(calibration_method, per_channel)]
        else:
            raise ValueError(
                f"Could not map with calibration_method = {calibration_method}, per_channel = {per_channel} and observer_class = {observer_class}"
            )
        return result

    @staticmethod
    def create_observer_class(calibration_method: CalibrationMethod, per_channel: bool) -> observer.ObserverBase:
        observer_class = QConfigUnit._observers_mapping(calibration_method=calibration_method, per_channel=per_channel)
        if observer_class is None:
            raise KeyError(
                f"Could not find an observer matching calibration_method = {calibration_method} with per_channel = {per_channel}."
            )
        return observer_class

    @staticmethod
    def create_calibration_method_and_per_channel(observer_class: Type) -> Tuple[CalibrationMethod, bool]:
        calibration_method, per_channel = QConfigUnit._observers_mapping(observer_class=observer_class)

    @staticmethod
    def get_observer_class_allowed_parameters(observer_class: Type) -> Set[str]:
        return set(inspect.signature(observer_class.__init__).parameters.keys())

    def get_observer_kwargs(self, observer_class: Type) -> Dict[str, Any]:
        allowed_parameters = self.get_observer_class_allowed_parameters(observer_class)
        kwargs = {name: attr for name, attr in self.__dict__.items() if name in allowed_parameters}
        kwargs["qscheme"] = _QSCHEME_MAPPING[(self.symmetric, self.per_channel)]
        return kwargs

    def get_observer_info(self):
        observer_class = self.create_observer_class(self.calibration_method, self.per_channel)
        observer_kwargs = self.get_observer_kwargs(observer_class)
        return observer_class, observer_kwargs

    def as_observer(self, as_factory: bool = True):
        observer_class, observer_kwargs = self.get_observer_info()
        return observer_class.with_args(**observer_kwargs) if as_factory else observer_class(**observer_kwargs)

    def as_fake_quantize(self):
        observer_class, observer_kwargs = self.get_observer_info()
        quant_min = observer_kwargs.pop("quant_min")
        if quant_min is None:
            quant_min = observer_class(dtype=self.dtype).quant_min
        quant_max = observer_kwargs.pop("quant_max")
        if quant_max is None:
            quant_max = observer_class(dtype=self.dtype).quant_max
        return torch.ao.quantization.FakeQuantize(
            observer=observer_class, quant_min=quant_min, quant_max=quant_max, **observer_kwargs
        )


@dataclass
class QConfig:
    # TODO: specify good defaults here
    activation: QConfigUnit = QConfigUnit()
    weight: QConfigUnit = QConfigUnit()

    def __post_init__(self):
        if isinstance(self.activation, dict):
            self.activation = QConfigUnit(**self.activation)
            self.weight = QConfigUnit(**self.weight)

    @classmethod
    def from_pytorch(cls, pytorch_qconfig: torch.ao.quantization.QConfig) -> "QConfig":
        return cls(
            activation=QConfigUnit.from_pytorch(pytorch_qconfig.activation),
            weight=QConfigUnit.from_pytorch(pytorch_qconfig.weight),
        )

    def to_pytorch(self, quantization_approach: QuantizationApproach) -> torch.ao.quantization.QConfig:
        if quantization_approach is QuantizationApproach.qat:
            qconfig = torch.ao.quantization.QConfig(
                activation=self.activation.as_fake_quantize(),
                weight=self.weight.as_fake_quantize(),
            )
        else:
            qconfig = torch.ao.quantization.QConfig(
                activation=self.activation.as_observer(),
                weight=self.weight.as_observer.as_observer(),
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
    CONFIG_NAME = "fx_quantization_config.json"
    FULL_CONFIGURATION_FILE = "fx_quantization_config.json"

    def __init__(self, **kwargs):
        def process(attr):
            def map_fn(item):
                if isinstance(item, dict):
                    return QConfig(**item)
                return item

            return QuantizationConfig._map(map_fn, attr, dicts_are_leafs=False)

        self._global = process(kwargs.pop("global", None))
        self.object_type = process(kwargs.pop("object_type", {}))
        self.module_name = process(kwargs.pop("module_name", {}))
        self.module_name_regex = process(kwargs.pop("module_name_regex", {}))
        self.module_name_object_type = process(kwargs.pop("module_name_object_type", {}))

        # Make sure that the loaded attributes are stored as dictionaries.
        self.transform_attributes(to_dict=True, inplace=True)

    @classmethod
    def default(cls, backend: str):
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
        return {t[:-1]: t for t in list_}

    def _dict_to_list(self, dict_):
        if isinstance(dict_, list):
            return dict_
        return list(dict_.values())

    def transform_attributes(self, to_list=False, to_dict=False, inplace=False):
        if to_list and to_dict:
            raise ValueError('Only one of "to_list" and "to_dict" can be set to True.')
        transform_function = self._list_to_dict if to_dict else self._dict_to_list
        attribute_names = ["object_type", "module_name", "module_name_regex", "module_name_object_type"]
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
        output.update(self.transform_attributes(to_list=True))
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

        output = self._map(qconfig_to_dict, output, dicts_are_leafs=False)
        output = self._map(torch_dtype_to_string, output, dicts_are_leafs=False)

        # Transformers version when serializing the model
        output["transformers_version"] = transformers_version
        output["optimum_version"] = __version__

        return output

    @classmethod
    def from_pytorch(cls, qconfig_dict: Dict[str, Any]) -> "QuantizationConfig":
        clone = copy.deepcopy(qconfig_dict)
        clone["global"] = clone.pop("", None)

        def map_fn(item):
            if isinstance(item, torch.ao.quantization.QConfig):
                return QConfig.from_pytorch(item)
            return item

        clone = cls._map(map_fn, clone, dicts_are_leafs=False)
        return cls(**clone)

    def to_pytorch(self, quantization_approach: QuantizationApproach):
        def to_pytorch(item):
            return self._map(lambda x: x.to_pytorch(quantization_approach), item)

        attributes = {
            attr_name: to_pytorch(attr) for attr_name, attr in self.transform_attributes(to_list=True).items()
        }
        return {"": to_pytorch(self.global_), **attributes}

    def summary(self):
        summary_list = ["*** Summary ***"]
        # Global
        global_qconfig_str = f"\t{self.global_}" if self.global_ is not None else "None"
        summary_list += ["Global:", global_qconfig_str]

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
        if self.module_name_object_type:
            summary_list.append("Module name object type:")
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
        if not isinstance(object_type, (type, callable, str)):
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

    # def add(self, *args):
    #     add_method = None
    #     if isinstance(args[0], str):
    #         if args[0] in ["", "global"]:
    #             self.global_ = args[1]
    #             return
    #         elif args[0].startswith("regex:"):
    #             args = (args[0].replace("regex:", ""),) + args[1:]
    #             add_method = self.add_module_name_regex
    #         else:
    #             add_method = self.add_module_name
    #     else:
    #         if len(args) == 2:
    #             add_method = self.add_object_type
    #         else:
    #             add_method = self.add_module_name_object_type

    #     if add_method is None:
    #         raise RuntimeError(f"Could not infer on the adding method from the following args: {args}")

    #     add_method(*args)

    # def remove(self, *args):
    #     remove_method = None
    #     if len(args) == 1:
    #         if args[0] in ["", "global"]:
    #             self.global_ = None
    #             return
    #         elif isinstance(args[0], str):
    #             if args[0] in self.module_name:
    #                 remove_method = self.remove_module_name
    #             else:
    #                 remove_method = self.remove_module_name_regex
    #         else:
    #             remove_method = self.remove_object_type
    #     else:
    #         remove_method = self.remove_module_name_object_type

    #     remove_method(*args)

    @global_.setter
    def global_(self, value: Optional[QConfig]):
        self._validate_qconfig_type(value)
        self._global = value

    def add_object_type(self, object_type: Union[Type, Callable], qconfig: Optional[QConfig]):
        self._validate_object_type_type(object_type)
        self._validate_qconfig_type(qconfig)
        self.object_type[object_type] = (object_type, qconfig)

    def remove_object_type(self, object_type: Union[torch.nn.Module, Callable]):
        self._validate_object_type_type(object_type)
        self.object_type.pop(object_type, None)

    def add_module_name(self, module_name: str, qconfig: Optional[QConfig]):
        self._validate_module_name(module_name, "module_name")
        self._validate_qconfig_type(qconfig)
        self.module_name[module_name] = (module_name, qconfig)

    def remove_module_name(self, module_name: str):
        self._validate_module_name(module_name, "module_name")
        self.module_name.pop(module_name, None)

    def add_module_name_regex(self, module_name_regex: str, qconfig: Optional[QConfig]):
        self._validate_module_name(module_name_regex, "module_name_regex")
        self._validate_qconfig_type(qconfig)
        self.module_name_regex[module_name_regex] = (module_name_regex, qconfig)

    def remove_module_name_regex(self, module_name_regex: str):
        self._validate_module_name(module_name_regex, "module_name_regex")
        self.module_name_regex.pop(module_name_regex, None)

    def add_module_name_object_type(
        self,
        object_type: Union[torch.nn.Module, Callable],
        module_name_regex: str,
        index: int,
        qconfig: Optional[QConfig],
    ):
        self._validate_object_type_type(object_type)
        self._validate_module_name(module_name_regex, "module_name_regex")
        self._validate_qconfig_type(qconfig)
        self.module_name_object_type[(object_type, module_name_regex, index)] = (
            object_type,
            module_name_regex,
            index,
            qconfig,
        )

    def remove_module_name_object_type(
        self, object_type: Union[torch.nn.Module, Callable], module_name_regex: str, index: str
    ):
        self._validate_object_type_type(object_type)
        self._validate_module_name(module_name_regex, "module_name_regex")
        self.module_name_object_type.pop((object_type, module_name_regex, index), -1)
