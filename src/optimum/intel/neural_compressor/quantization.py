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
import torch
from enum import Enum
from optimum.intel.neural_compressor.config import IncConfig
from optimum.intel.neural_compressor.utils import IncDataLoader
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Callable, ClassVar, Dict, Optional, Union


class IncQuantizationMode(Enum):

    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"


SUPPORTED_QUANT_MODE = set([approach.value for approach in IncQuantizationMode])


class IncQuantizer:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(
            self,
            config_path_or_obj: Union[str, IncConfig],
            model: Union[PreTrainedModel, torch.nn.Module],
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            eval_func: Optional[Callable] = None,
            train_func: Optional[Callable] = None,
            calib_dataloader: Optional[DataLoader] = None,
    ):
        """
        Args:
            config_path_or_obj (:obj:`Union[str, IncConfig]` ):
                Path to the YAML configuration file or an IncConfig object, used to control the tuning behavior.
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
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
                Repository name in the Hugging Face Hub or path to a local directory containing the configuration file.
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


def apply_quantization_from_config(q_config: Dict, model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply Intel Neural Compressor (INC) quantization steps on the given model.

    Args:
        q_config (:obj:`Dict`):
            Dictionary containing all quantization information such as approach, dtype, scheme and granularity.
        model (:obj:`torch.nn.Module`):
            Model to quantize.
    Returns:
        q_model (:obj:`torch.nn.Module`):
            Quantized model.
    """
    import copy
    from torch.quantization import add_observer_, convert
    from neural_compressor.adaptor.pytorch import _cfg_to_qconfig, _propagate_qconfig

    approach = q_config.get("approach")
    framework = q_config.get("framework")

    if approach not in SUPPORTED_QUANT_MODE:
        raise ValueError("Unknown quantization approach. Supported approach are " +
                         ", ".join(SUPPORTED_QUANT_MODE.keys()))

    if approach == IncQuantizationMode.DYNAMIC.value:
        q_mapping = torch.quantization.quantization_mappings.get_default_dynamic_quant_module_mappings()
        white_list = torch.quantization.quantization_mappings.get_default_dynamic_quant_module_mappings()
        op_cfgs = _cfg_to_qconfig(q_config, approach)
    else:
        q_mapping = torch.quantization.quantization_mappings.get_default_static_quant_module_mappings()
        white_list = torch.quantization.quantization_mappings.get_default_qconfig_propagation_list() - \
                     {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}
        op_cfgs = _cfg_to_qconfig(q_config)

    q_model = copy.deepcopy(model)
    q_model.eval()

    if framework == "pytorch_fx":
        from optimum.intel.neural_compressor.utils import _cfgs_to_fx_cfgs
        from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx

        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, approach)

        if approach == IncQuantizationMode.AWARE_TRAINING.value:
            q_model.train()
            q_model = prepare_qat_fx(q_model, fx_op_cfgs)
        else:
            q_model = prepare_fx(q_model, fx_op_cfgs)
        q_model = convert_fx(q_model)
        return q_model

    _propagate_qconfig(q_model, op_cfgs, white_list=white_list, approach=approach)

    if approach != IncQuantizationMode.DYNAMIC.value:
        add_observer_(q_model)
    q_model = convert(q_model, mapping=q_mapping, inplace=True)

    return q_model


class IncQuantizedModel:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the"
            f"`{self.__class__.__name__}.from_pretrained(model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path: str,
            config_name: str,
            q_model_name: Optional[str] = None,
            state_dict: Optional[Dict[str, torch.Tensor]] = None,
            batch_size: Optional[int] = None,
            sequence_length: Optional[int] = None,
            **kwargs
    ) -> torch.nn.Module:
        """
        Instantiate a quantized pytorch model from a given Intel Neural Compressor (INC) configuration file.
        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hub or path to a local directory hosting the model.
            config_name (:obj:`str`):
                Name of the configuration file.
            q_model_name (:obj:`str`, `optional`):
                Name of the state dictionary located in model_name_or_path used to load the quantized model. If
                state_dict is specified, the latter will not be used.
            state_dict (:obj:`Dict[str, torch.Tensor]`, `optional`):
                State dictionary of the quantized model, if not specified q_model_name will be used to load the
                state dictionary.
        Returns:
            q_model: Quantized model.
        """

        if state_dict is None and q_model_name is None:
            raise RuntimeError("`IncQuantizedModel` requires either a `state_dict` or `q_model_name` argument")

        from .config import DeployIncConfig

        cache_dir = kwargs.get("cache_dir", None)

        deploy_config = DeployIncConfig.from_pretrained(
            model_name_or_path,
            config_name,
            cache_dir=cache_dir,
        )

        model_kwargs = kwargs

        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(
            model_name_or_path,
            state_dict=state_dict,
            **model_kwargs
        )

        if deploy_config.get_config("framework") == "pytorch_fx":
            from transformers.utils.fx import symbolic_trace

            if batch_size is None or sequence_length is None:
                raise ValueError("Need batch_size and sequence_length for tracing the model with torch fx.")

            model = symbolic_trace(
                model,
                input_names=["input_ids", "attention_mask", "token_type_ids", "labels"],
                batch_size=batch_size,
                sequence_length=sequence_length
            )

        q_model = apply_quantization_from_config(deploy_config.config, model)

        if state_dict is None:
            revision = None
            if len(model_name_or_path.split("@")) == 2:
                model_name_or_path, revision = model_name_or_path.split("@")

            if os.path.isdir(model_name_or_path) and q_model_name in os.listdir(model_name_or_path):
                state_dict_path = os.path.join(model_name_or_path, q_model_name)
            else:
                from huggingface_hub import hf_hub_download
                import requests
                try:
                    state_dict_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename=q_model_name,
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                except requests.exceptions.RequestException:
                    raise ValueError(f"{q_model_name} NOT FOUND in HuggingFace Hub")

            state_dict = torch.load(state_dict_path)

        q_model.load_state_dict(state_dict, strict=False)

        return q_model


class IncQuantizedModelForQuestionAnswering(IncQuantizedModel):
    from transformers import AutoModelForQuestionAnswering
    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncQuantizedModelForSequenceClassification(IncQuantizedModel):
    from transformers import AutoModelForSequenceClassification
    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


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

    if approach not in SUPPORTED_QUANT_MODE:
        raise ValueError("Unknown quantization approach. Supported approach are " + ", ".join(SUPPORTED_QUANT_MODE))

    return approach


def quantize_dynamic(model, config_path_or_obj,  eval_func):
    """
    Apply Intel Neural Compressor (INC) dynamic quantization.

    Args:
        model:
            FP32 model specified for low precision tuning.
        config_path_or_obj:
            Path to the YAML configuration file or an IncConfig object, used to control the tuning behavior.
        eval_func:
            Evaluation function provided by user.
    Returns:
        model:
            Quantized model.
    """

    from neural_compressor.experimental import Quantization, common

    config = config_path_or_obj.config if isinstance(config_path_or_obj, IncConfig) else config_path_or_obj
    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model


def quantize_static(model, config_path_or_obj, eval_func, calib_dataloader):
    """
    Apply Intel Neural Compressor (INC) post-training quantization.

    Args:
        model:
            FP32 model specified for low precision tuning.
        config_path_or_obj:
            Path to the YAML configuration file or an IncConfig object, used to control the tuning behavior.
        eval_func:
            Evaluation function provided by user.
        calib_dataloader:
            IncDataLoader for calibration.
    Returns:
        model:
            Quantized model.
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
    Apply Intel Neural Compressor (INC) quantization aware training.

    Args:
        model:
            FP32 model specified for low precision tuning.
        config_path_or_obj:
            Path to the YAML configuration file or an IncConfig object, used to control the tuning behavior.
        eval_func:
            Evaluation function provided by user.
        train_func:
            Training function provided by user.
    Returns:
        model:
            Quantized model.
    """

    from neural_compressor.experimental import Quantization, common

    config = config_path_or_obj.config if isinstance(config_path_or_obj, IncConfig) else config_path_or_obj
    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.q_func = train_func
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model

