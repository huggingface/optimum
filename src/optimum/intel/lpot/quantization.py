#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import os
from collections import OrderedDict
from enum import Enum
from typing import Callable, ClassVar, Optional, Union

from lpot.adaptor.adaptor import adaptor_registry, FRAMEWORKS
from lpot.adaptor.pytorch import PyTorch_FXAdaptor
from lpot.utils import logger
from lpot.utils.utility import dump_elapsed_time

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from transformers.utils.fx import (
    is_model_supported_for_symbolic_tracing,
    prepare_for_retracing,
    restore_after_retracing_,
    retracing_ready,
    symbolic_trace,
    _SUPPORTED_MODELS_FOR_DYNAMIC_AXES
)


class LpotQuantizationMode(Enum):

    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"


@adaptor_registry
class HFPyTorch_FXAdaptor(PyTorch_FXAdaptor):

    def trace(self, model):
        _, dynamic_axes_are_supported = is_model_supported_for_symbolic_tracing(model)
        if not dynamic_axes_are_supported:
            supported_model_names = ", ".join(
                (cls.__name__ for cls in _SUPPORTED_MODELS_FOR_DYNAMIC_AXES)
            )
            raise NotImplementedError(
                f"Dynamic axes are not supported for {model.__class__.__name__} yet, supported models: {supported_model_names}"
            )
        # TODO: take care of the way to pass input_names, and num_choices if needed.
        return symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask", "token_type_ids", "labels"],
            batch_size=-1,
            sequence_length=-1
        )

    @dump_elapsed_time("Pass query framework capability")
    def query_fw_capability(self, model):
        """This is a helper function to get all quantizable ops from model.

        Args:
            model (object): input model which is LPOT model

        Returns:
            q_capability (dictionary): tuning capability for each op from model.
        """
        self.pre_optimized_model = model
        tmp_model = model.model
        if isinstance(self, PyTorch_FXAdaptor):
            try:
                tmp_model = copy.deepcopy(model.model)
            except Exception as e:                              # pragma: no cover
                logger.warning("Deepcopy failed: {}, inplace=True now!".format(repr(e)))

            from torch.fx import GraphModule
            from torch.quantization.quantize_fx import (
                QuantizationTracer,
                _fuse_fx,
            )

            if model.kwargs is not None and \
                    model.kwargs.__contains__('prepare_custom_config_dict'):
                prepare_custom_config_dict = model.kwargs['prepare_custom_config_dict']
            else:
                prepare_custom_config_dict = {}
            skipped_module_names = prepare_custom_config_dict.get(\
                                                "non_traceable_module_name", [])
            skipped_module_classes = prepare_custom_config_dict.get(\
                                                "non_traceable_module_class", [])

            traced = self.trace(tmp_model)
            traced, attributes = prepare_for_retracing(traced)
            tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)
            graph_module = GraphModule(traced, tracer.trace(traced))
            restore_after_retracing_(graph_module, attributes)

            tmp_model = _fuse_fx(graph_module, prepare_custom_config_dict)

        quantizable_ops = []
        self._get_quantizable_ops_recursively(tmp_model, '', quantizable_ops)
        capability = self.query_handler.get_quantization_capability()['dynamic'] \
            if self.approach == "post_training_dynamic_quant" else \
            self.query_handler.get_quantization_capability()['int8']

        q_capability = {}
        q_capability['optypewise'] = OrderedDict()
        q_capability['opwise'] = OrderedDict()

        for q_op in quantizable_ops:
            q_capability['opwise'][q_op] = copy.deepcopy(capability[q_op[1]]) \
                if q_op[1] in capability.keys() else copy.deepcopy(capability['default'])
            if q_op[1] not in q_capability['optypewise'].keys():
                q_capability['optypewise'][q_op[1]] = copy.deepcopy(capability[q_op[1]]) \
                    if q_op[1] in capability.keys() else copy.deepcopy(capability['default'])

        return q_capability

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        from torch.quantization.quantize_fx import QuantizationTracer, _prepare_fx
        from transformers.utils.fx import HFTracer

        orig__module_getattr = QuantizationTracer._module_getattr
        QuantizationTracer._module_getattr = HFTracer._module_getattr

        orig__prepare_fx = _prepare_fx
        torch.quantization.quantize_fx._prepare_fx = retracing_ready(_prepare_fx)

        model.model = self.trace(model.model)
        import pdb; pdb.set_trace()
        res = super().quantize(tune_cfg, model, dataloader, q_func=q_func)

        QuantizationTracer._module_getattr = orig__module_getattr
        torch.quantization.quantize_fx._prepare_fx = orig__prepare_fx

        return res

    def _pre_hook_for_qat(self):
        from torch.quantization.quantize_fx import _prepare_fx
        orig = _prepare_fx
        torch.quantization.quantize_fx._prepare_fx = retracing_ready(_prepare_fx)
        res = super()._pre_hook_for_qat()
        torch.quantization.quantize_fx._prepare_fx = orig
        return res


class LpotQuantizer:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(
            self,
            config_path: str,
            model: Union[PreTrainedModel, nn.Module],
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            eval_func: Optional[Callable] = None,
            train_func: Optional[Callable] = None,
            calib_dataloader: Optional[DataLoader] = None,
    ):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
            model (:obj:`Union[PreTrainedModel, nn.Module]`):
                FP32 model specified for low precision tuning.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `optional`):
                Tokenizer used to preprocess the data.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function provided by user.
            train_func (:obj:`Callable`, `optional`):
                Training function provided by user.
            calib_dataloader (:obj:`Callable`, `optional`):
                DataLoader for calibration.
        Returns:
            quantizer: LpotQuantizer object.
        """
        from lpot.conf.config import Conf

        self.config_path = config_path
        self.config = Conf(config_path).usr_cfg
        self.approach = self.config.quantization.approach
        self.model = model
        self.tokenizer = tokenizer
        self._eval_func = eval_func
        self._train_func = train_func
        if calib_dataloader is not None:
            calib_dataloader = LpotDataLoader.from_pytorch_dataloader(calib_dataloader)
        self._calib_dataloader = calib_dataloader

    @property
    def eval_func(self):
        return self._eval_func

    @property
    def train_func(self):
        return self._train_func

    @property
    def calib_dataloader(self):
        return self._calib_dataloader

    @eval_func.setter
    def eval_func(self, func: Callable):
        self._eval_func = func

    @train_func.setter
    def train_func(self, func: Callable):
        self._train_func = func

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader: DataLoader):
        self._calib_dataloader = LpotDataLoader.from_pytorch_dataloader(dataloader)

    def init_quantizer(self):
        if self.config.model.framework == "pytorch_fx":
            import lpot
            from optimum.intel.lpot.utils import _cfgs_to_fx_cfgs, _get_quantizable_ops_recursively
            # TODO : Change this to apply quantization on other part of the model other that Linears
            lpot.adaptor.pytorch._cfgs_to_fx_cfgs = _cfgs_to_fx_cfgs
            lpot.adaptor.pytorch.PyTorch_FXAdaptor._get_quantizable_ops_recursively = _get_quantizable_ops_recursively

        from lpot.experimental import Quantization, common

        # TODO: make things cleaner regarding the adaptor.
        # Currently patching the adaptor class in the FRAMEWORKS registry works fine but getting things to work without
        # patching would be better as HFPytorch_FXAdaptor is also registered in FRAMEWORKS.

        # from lpot.conf.config import Quantization_Conf
        # conf = Quantization_Conf(self.config_path)
        # if conf.framework == "pytorch_fx":
        #     conf.framework = "hfpytorch_fx"
        # quantizer = Quantization(conf)
        # For now:
        # quantizer = Quantization(self.config_path)
        # if quantizer.framework == "pytorch_fx":
        #     quantizer.framework = "hfpytorch_fx"
        #     quantizer.cfg.model.framework = "hfpytorch_fx"

        # pytorch_fx_adaptor = FRAMEWORKS["pytorch_fx"]
        FRAMEWORKS["pytorch_fx"] = FRAMEWORKS["hfpytorch_fx"]
        quantizer = Quantization(self.config_path)
        # FRAMEWORKS["pytorch_fx"] = pytorch_fx_adaptor

        quantizer.model = common.Model(self.model)

        if self._eval_func is None:
            raise ValueError("eval_func must be provided for quantization.")

        quantizer.eval_func = self._eval_func
        return quantizer

    @staticmethod
    def adaptor_calib():
        from lpot.adaptor.pytorch import PyTorchAdaptor

        from .utils import model_calibration
        PyTorchAdaptor.model_calibration = model_calibration

    def fit_dynamic(self):
        quantizer = self.init_quantizer()
        model = quantizer()
        return model

    def fit_static(self):
        quantizer = self.init_quantizer()

        if self._calib_dataloader is None:
            raise ValueError("calib_dataloader must be provided for post-training quantization.")

        quantizer.calib_dataloader = self._calib_dataloader

        model = quantizer()
        return model

    def fit_aware_training(self):
        quantizer = self.init_quantizer()

        if self._train_func is None:
            raise ValueError("train_func must be provided for quantization aware training.")

        quantizer.q_func = self._train_func

        model = quantizer()
        return model

    @classmethod
    def from_config(
            cls,
            config_name_or_path: Union[str, os.PathLike],
            config_name: str,
            model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            **quantizer_kwargs
    ):
        """
        Instantiate a LpotQuantizer object from a configuration file which can either be hosted on huggingface.co or
        from a local directory path.

        Args:
            config_name_or_path (:obj:`Union[str, os.PathLike]`):
                Repository name in the Hub or path to a local directory containing the configuration file.
            config_name (:obj:`str`):
                Name of the configuration file.
            model_name_or_path (:obj:Union[str, os.PathLike]`, `optional`):
                Used to instantiate the model and tokenizer if specified, otherwise config_name_or_path is used.
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            quantizer_kwargs (:obj:`Dict`, `optional`):
                quantizer_kwargs will be passed to the LpotQuantizer object during initialization.
        Returns:
            quantizer: LpotQuantizer object.
        """

        from transformers import AutoTokenizer

        from .config import LpotConfig

        q8_config = LpotConfig.from_pretrained(
            config_name_or_path,
            config_name,
            cache_dir=cache_dir,
        )
        model_name_or_path = model_name_or_path if model_name_or_path is not None else config_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
        quantizer_kwargs.update({'tokenizer': tokenizer})
        quantizer = cls(q8_config.path, model, **quantizer_kwargs)
        return quantizer


class LpotQuantizerForQuestionAnswering(LpotQuantizer):
    from transformers import AutoModelForQuestionAnswering
    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class LpotQuantizerForSequenceClassification(LpotQuantizer):
    from transformers import AutoModelForSequenceClassification
    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


SUPPORTED_QUANT_APPROACH = {
    LpotQuantizationMode.STATIC.value,
    LpotQuantizationMode.DYNAMIC.value,
    LpotQuantizationMode.AWARE_TRAINING.value
}


def quantization_approach(config):
    """
    Extract quantization approach from YAML configuration file.

    Args:
        config: YAML configuration file used to control the tuning behavior.
    Returns:
        approach: Name of the quantization approach.
    """

    from lpot.conf.config import Conf
    from lpot.conf.dotdict import deep_get

    conf = Conf(config)
    approach = deep_get(conf.usr_cfg, "quantization.approach")

    if approach not in SUPPORTED_QUANT_APPROACH:
        raise ValueError("Unknown quantization approach. Supported approach are " + ", ".join(SUPPORTED_QUANT_APPROACH))

    return approach


def quantize_dynamic(model, config,  eval_func):
    """
    Apply LPOT dynamic quantization.

    Args:
        model: FP32 model specified for low precision tuning.
        config: YAML configuration file used to control the tuning behavior.
        eval_func: Evaluation function provided by user.
    Returns:
        model: Quantized model.
    """

    from lpot.experimental import Quantization, common

    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model


def quantize_static(model, config, eval_func, calib_dataloader):
    """
    Apply LPOT post-training quantization.

    Args:
        model: FP32 model specified for low precision tuning.
        config: YAML configuration file used to control the tuning behavior.
        eval_func: Evaluation function provided by user.
        calib_dataloader: DataLoader for calibration.
    Returns:
        model: Quantized model.
    """

    from lpot.adaptor.pytorch import PyTorchAdaptor
    from lpot.experimental import Quantization, common

    from .utils import model_calibration

    PyTorchAdaptor.model_calibration = model_calibration

    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.calib_dataloader = calib_dataloader
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model


def quantize_aware_training(model, config, eval_func, train_func):
    """
    Apply LPOT quantization aware training.

    Args:
        model: FP32 model specified for low precision tuning.
        config: YAML configuration file used to control the entire tuning behavior.
        eval_func: Evaluation function provided by user.
        train_func: Training function provided by user.
    Returns:
        model: Quantized model.
    """

    from lpot.adaptor.pytorch import PyTorchAdaptor
    from lpot.experimental import Quantization, common

    from .utils import model_calibration

    PyTorchAdaptor.model_calibration = model_calibration

    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.q_func = train_func
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model
