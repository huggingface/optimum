import yaml
import os
import requests
from enum import Enum
from functools import reduce
from huggingface_hub import hf_hub_download
from typing import Any, Callable, ClassVar, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch import nn
from torch.utils.data import DataLoader


class LpotQuantizationMode(Enum):

    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"


class LpotQuantizationConfig:

    def __init__(
            self,
            config_path: str,
            save_path: Optional[str] = None,
            overwrite: Optional[bool] = False,
    ):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
            save_path (:obj:`str`, `optional`):
                Path used to save the configuration file.
            overwrite (:obj:`bool`, `optional`):
                Whether or not overwrite the configuration file when the latter is modified and saved.
        Returns:
            config: LpotQuantizationConfig object.
        """

        self.path = config_path
        self.config = self._read_config()
        self.save_path = save_path
        self.overwrite = overwrite

    def _read_config(self):
        with open(self.path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        return config

    def get_config(self, keys: str):
        return reduce(lambda d, key: d.get(key) if d else None, keys.split("."), self.config)

    def set_config(self, keys: str, value: Any):
        d = self.config
        keys = keys.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
        self._save_pretrained()

    def _save_pretrained(self):
        if self.save_path is None and not self.overwrite:
            raise ValueError("Needs either path or overwrite set to True.")

        self.path = self.save_path if self.save_path is not None else self.path
        with open(self.path, "w") as f:
            yaml.dump(self.config, f)

    @classmethod
    def from_pretrained(
            cls,
            config_name_or_path: Union[str, os.PathLike],
            config_name: str,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            **config_kwargs
    ):
        """
        Instantiate a LpotQuantizationConfig object from a configuration file which can either be hosted on
        huggingface.co or from a local directory path.

        Args:
            config_name_or_path (:obj:`Union[str, os.PathLike]`):
                Repository name in the Hub or path to a local directory containing the configuration file.
            config_name (:obj:`str`):
                Name of the configuration file.
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            config_kwargs (:obj:`Dict`, `optional`):
                config_kwargs will be passed to the LpotQuantizationConfig object during initialization.
        Returns:
            config: LpotQuantizationConfig object.
        """

        revision = None
        if len(config_name_or_path.split("@")) == 2:
            config_name_or_path, revision = config_name_or_path.split("@")

        if os.path.isdir(config_name_or_path) and config_name in os.listdir(config_name_or_path):
            config_file = os.path.join(config_name_or_path, config_name)
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=config_name_or_path,
                    filename=config_name,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            except requests.exceptions.RequestException:
                raise ValueError(f"{config_name} NOT FOUND in HuggingFace Hub")

        config = cls(config_file, **config_kwargs)
        return config


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
        self._calib_dataloader = dataloader

    def init_quantizer(self):
        from lpot.experimental import Quantization, common

        quantizer = Quantization(self.config_path)
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
        self.adaptor_calib()
        quantizer = self.init_quantizer()

        if self._calib_dataloader is None:
            raise ValueError("calib_dataloader must be provided for post-training quantization.")

        quantizer.calib_dataloader = self._calib_dataloader

        model = quantizer()
        return model

    def fit_aware_training(self):
        self.adaptor_calib()
        quantizer = self.init_quantizer()

        if self._train_func is None:
            raise ValueError("train_func must be provided for quantization aware training .")

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

        q8_config = LpotQuantizationConfig.from_pretrained(
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

    from lpot.experimental import Quantization, common
    from lpot.adaptor.pytorch import PyTorchAdaptor
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

    from lpot.experimental import Quantization, common
    from lpot.adaptor.pytorch import PyTorchAdaptor
    from .utils import model_calibration

    PyTorchAdaptor.model_calibration = model_calibration

    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.q_func = train_func
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model

