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
import logging
import torch
from enum import Enum
from optimum.intel.neural_compressor.config import IncConfig, IncOptimizedConfig
from optimum.intel.neural_compressor.utils import IncDataLoader
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


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
                Path to the YAML configuration file or an instance of the class :class:`IncConfig`, used to control the
                tuning behavior.
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                FP32 model specified for low precision tuning.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `optional`):
                Tokenizer used to preprocess the data.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function for quantization aware training approach.
            calib_dataloader (:obj:`DataLoader`, `optional`):
                DataLoader for post-training quantization calibration.

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
            model_name_or_path: str,
            inc_config: Optional[Union[IncConfig, str]] = None,
            config_name: str = None,
            **kwargs
    ):
        """
        Instantiate a IncQuantizer object from a configuration file which can either be hosted on huggingface.co or
        from a local directory path.

        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            inc_config (:obj:`Union[IncConfig, str]`, `optional`):
                Configuration file containing all the information related to the model quantization.
                Can be either:
                    - an instance of the class :class:`IncConfig`,
                    - a string valid as input to :func:`IncConfig.from_pretrained`.
            config_name (:obj:`str`, `optional`):
                Name of the configuration file.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(:obj:`str`, `optional`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            calib_dataloader (:obj:`DataLoader`, `optional`):
                DataLoader for post-training quantization calibration.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function for quantization aware training approach.
        Returns:
            quantizer: IncQuantizer object.
        """
        from transformers import AutoTokenizer

        config_kwargs_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None)
        ]
        config_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in config_kwargs_default}
        quantizer_kwargs_names = ["eval_func", "train_func", "calib_dataloader"]
        quantizer_kwargs = {name: kwargs.pop(name, None) for name in quantizer_kwargs_names}

        if not isinstance(inc_config, IncConfig):
            config_path = inc_config if inc_config is not None else model_name_or_path
            inc_config = IncConfig.from_pretrained(
                config_path,
                config_file_name=config_name,
                **config_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(model_name_or_path, **kwargs)
        quantizer_kwargs["tokenizer"] = tokenizer
        quantizer = cls(inc_config, model, **quantizer_kwargs)
        return quantizer


class IncQuantizerForQuestionAnswering(IncQuantizer):
    from transformers import AutoModelForQuestionAnswering
    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncQuantizerForSequenceClassification(IncQuantizer):
    from transformers import AutoModelForSequenceClassification
    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class IncQuantizerForTokenClassification(IncQuantizer):
    from transformers import AutoModelForTokenClassification
    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


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
            inc_config: Union[IncOptimizedConfig, str] = None,
            q_model_name: Optional[str] = None,
            input_names: Optional[List[str]] = None,
            batch_size: Optional[int] = None,
            sequence_length: Optional[Union[int, List[int], Tuple[int]]] = None,
            **kwargs
    ) -> torch.nn.Module:
        """
        Instantiate a quantized pytorch model from a given Intel Neural Compressor (INC) configuration file.
        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            inc_config (:obj:`Union[IncOptimizedConfig, str]`, `optional`):
                Configuration file containing all the information related to the model quantization.
                Can be either:
                    - an instance of the class :class:`IncOptimizedConfig`,
                    - a string valid as input to :func:`IncOptimizedConfig.from_pretrained`.
            q_model_name (:obj:`str`, `optional`):
                Name of the state dictionary located in model_name_or_path used to load the quantized model. If
                state_dict is specified, the latter will not be used.
            input_names (:obj:`List[str]`, `optional`):
                List of names of the inputs used when tracing the model. If unset, model.dummy_inputs().keys() are used
                instead.
            batch_size (:obj:`int`, `optional`):
                Batch size of the traced model inputs.
            sequence_length (:obj:`Union[int, List[int], Tuple[int]]`, `optional`):
                Sequence length of the traced model inputs. For sequence-to-sequence models with different sequence
                lengths between the encoder and the decoder inputs, this must be :obj:`[encoder_sequence_length,
                decoder_sequence_length]`.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(:obj:`str`, `optional`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            state_dict (:obj:`Dict[str, torch.Tensor]`, `optional`):
                State dictionary of the quantized model, if not specified q_model_name will be used to load the
                state dictionary.
        Returns:
            q_model: Quantized model.
        """
        download_kwarg_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None)
        ]
        download_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in download_kwarg_default}
        state_dict = kwargs.get("state_dict", None)

        if not isinstance(inc_config, IncOptimizedConfig):
            config_path = inc_config if inc_config is not None else model_name_or_path
            inc_config = IncOptimizedConfig.from_pretrained(config_path, **download_kwargs)

        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(model_name_or_path, **kwargs)

        if inc_config.get_config("framework") == "pytorch_fx":
            from transformers.utils.fx import symbolic_trace
            if batch_size is None or sequence_length is None:
                raise ValueError("Need batch_size and sequence_length for tracing the model with torch fx.")

            model = symbolic_trace(
                model,
                input_names=input_names,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )

        q_model = apply_quantization_from_config(inc_config.config, model)

        if state_dict is None:
            from transformers.file_utils import cached_path, hf_bucket_url
            from optimum.intel.neural_compressor.utils import WEIGHTS_NAME

            q_model_name = q_model_name if q_model_name is not None else WEIGHTS_NAME
            revision = download_kwargs.pop("revision", None)
            if os.path.isdir(model_name_or_path):
                state_dict_path = os.path.join(model_name_or_path, q_model_name)
            elif os.path.isfile(model_name_or_path):
                state_dict_path = model_name_or_path
            else:
                state_dict_path = hf_bucket_url(model_name_or_path, filename=q_model_name, revision=revision)

            try:
                state_dict_path = cached_path(
                    state_dict_path,
                    **download_kwargs,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load config for '{model_name_or_path}'. Make sure that:\n\n - '{model_name_or_path}' is a "
                    f"correct model identifier listed on 'https://huggingface.co/models'\n\n - or "
                    f"'{model_name_or_path}' is a correct path to a directory containing a {q_model_name} file\n\n"
                )

                if revision is not None:
                    msg += f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id)  " \
                           f"thatexists for this model name as listed on its model page on " \
                           f"'https://huggingface.co/models'\n\n"

                raise EnvironmentError(msg)

            state_dict = torch.load(state_dict_path)

        q_model.load_state_dict(state_dict, strict=False)

        return q_model


class IncQuantizedModelForQuestionAnswering(IncQuantizedModel):
    from transformers import AutoModelForQuestionAnswering
    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncQuantizedModelForSequenceClassification(IncQuantizedModel):
    from transformers import AutoModelForSequenceClassification
    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class IncQuantizedModelForTokenClassification(IncQuantizedModel):
    from transformers import AutoModelForTokenClassification
    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


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
            Path to the YAML configuration file or an instance of the class :class:`IncConfig`, used to control the
            tuning behavior.
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
            Path to the YAML configuration file or an instance of the class :class:`IncConfig`, used to control the
            tuning behavior.
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
            Path to the YAML configuration file or an instance of the class :class:`IncConfig`, used to control the
            tuning behavior.
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

