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

import os
from enum import Enum
from optimum.intel.neural_compressor.utils import IncDataLoader
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Callable, ClassVar, Optional, Union

from optimum.intel.neural_compressor.config import IncConfig


class IncQuantizationMode(Enum):

    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"


class IncQuantizer:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(
            self,
            config_path_or_obj: Union[str, IncConfig],
            model: Union[PreTrainedModel, nn.Module],
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            eval_func: Optional[Callable] = None,
            train_func: Optional[Callable] = None,
            calib_dataloader: Optional[DataLoader] = None,
    ):
        """
        Args:
            config_path_or_obj (:obj:`Union[str, IncConfig]` ):
                Path to the YAML configuration file used to control the tuning behavior or
                a class of IncConfig.
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
            quantizer: IncQuantizer object.
        """
        from neural_compressor.conf.config import Quantization_Conf

        self.config = \
            config_path_or_obj.config if isinstance(config_path_or_obj, IncConfig) else \
            Quantization_Conf(config_path_or_obj)
        self.approach = self.config.usr_cfg.quantization.approach
        self.model = model
        self.tokenizer = tokenizer
        self._eval_func = eval_func
        self._train_func = train_func
        if calib_dataloader is not None:
            calib_dataloader = IncDataLoader.from_pytorch_dataloader(calib_dataloader)
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
        self._calib_dataloader = IncDataLoader.from_pytorch_dataloader(dataloader)

    def init_quantizer(self):
        if self.config.usr_cfg.model.framework == "pytorch_fx":
            import neural_compressor
            from optimum.intel.neural_compressor.utils import _cfgs_to_fx_cfgs, _get_quantizable_ops_recursively
            # TODO : Change this to apply quantization on other part of the model other that Linears
            neural_compressor.adaptor.pytorch._cfgs_to_fx_cfgs = _cfgs_to_fx_cfgs
            neural_compressor.adaptor.pytorch.PyTorch_FXAdaptor._get_quantizable_ops_recursively = _get_quantizable_ops_recursively

        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization(self.config)
        quantizer.model = common.Model(self.model)

        if self._eval_func is None:
            raise ValueError("eval_func must be provided for quantization.")

        quantizer.eval_func = self._eval_func
        return quantizer

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
        Instantiate a IncQuantizer object from a configuration file which can either be hosted on huggingface.co or
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
                quantizer_kwargs will be passed to the IncQuantizer object during initialization.
        Returns:
            quantizer: IncQuantizer object.
        """

        from transformers import AutoTokenizer
        from .config import IncConfig

        q8_config = IncConfig.from_pretrained(
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
        quantizer = cls(q8_config, model, **quantizer_kwargs)
        return quantizer


class IncQuantizerForQuestionAnswering(IncQuantizer):
    from transformers import AutoModelForQuestionAnswering
    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncQuantizerForSequenceClassification(IncQuantizer):
    from transformers import AutoModelForSequenceClassification
    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


SUPPORTED_QUANT_APPROACH = {
    IncQuantizationMode.STATIC.value,
    IncQuantizationMode.DYNAMIC.value,
    IncQuantizationMode.AWARE_TRAINING.value
}


def quantization_approach(config):
    """
    Extract quantization approach from YAML configuration file.
    Args:
        config: YAML configuration file used to control the tuning behavior.
    Returns:
        approach: Name of the quantization approach.
    """

    from neural_compressor.conf.config import Quantization_Conf
    from neural_compressor.conf.dotdict import deep_get

    conf = Quantization_Conf(config)
    approach = deep_get(conf.usr_cfg, "quantization.approach")

    if approach not in SUPPORTED_QUANT_APPROACH:
        raise ValueError("Unknown quantization approach. Supported approach are " + ", ".join(SUPPORTED_QUANT_APPROACH))

    return approach


def quantize_dynamic(model, config_path_or_obj,  eval_func):
    """
    Apply neural_compressor dynamic quantization.

    Args:
        model: FP32 model specified for low precision tuning.
        config_path_or_obj: YAML configuration file used to control the tuning behavior or
                            a class of IncConfig.
        eval_func: Evaluation function provided by user.
    Returns:
        model: Quantized model.
    """

    from neural_compressor.experimental import Quantization, common

    config = config_path_or_obj.config if isinstance(config_path_or_obj, IncConfig) else \
             config_path_or_obj
    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model


def quantize_static(model, config_path_or_obj, eval_func, calib_dataloader):
    """
    Apply neural_compressor post-training quantization.

    Args:
        model: FP32 model specified for low precision tuning.
        config_path_or_obj: YAML configuration file used to control the tuning behavior or
                            a class of IncConfig.
        eval_func: Evaluation function provided by user.
        calib_dataloader: IncDataLoader for calibration.
    Returns:
        model: Quantized model.
    """

    from neural_compressor.experimental import Quantization, common

    config = config_path_or_obj.config if isinstance(config_path_or_obj, IncConfig) else config_path_or_obj
    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.calib_dataloader = calib_dataloader
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model


def quantize_aware_training(model, config_path_or_obj, eval_func, train_func):
    """
    Apply neural_compressor quantization aware training.

    Args:
        model: FP32 model specified for low precision tuning.
        config_path_or_obj: YAML configuration file used to control the entire tuning behavior or
                            a class of IncConfig.
        eval_func: Evaluation function provided by user.
        train_func: Training function provided by user.
    Returns:
        model: Quantized model.
    """

    from neural_compressor.experimental import Quantization, common

    config = config_path_or_obj.config if isinstance(config_path_or_obj, IncConfig) else config_path_or_obj
    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.q_func = train_func
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model

