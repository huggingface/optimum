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

import logging
import os
from enum import Enum
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from optimum.intel.neural_compressor.config import IncOptimizedConfig, IncDistillationConfig
from optimum.intel.neural_compressor.utils import IncDataLoader


logger = logging.getLogger(__name__)


class IncDistillationMode(Enum):
    DISTILLATION = "distillation"


SUPPORTED_DISTILLATION_MODE = set([approach.value for approach in IncDistillationMode])


class IncDistillation:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        teacher_model: Union[PreTrainedModel, torch.nn.Module],
        config_path_or_obj: Union[str, IncDistillationConfig],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                Student model.
            teacher_model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                Teacher model.
            config_path_or_obj (:obj:`Union[str, IncDistillationConfig]` ):
                Path to the YAML configuration file or an instance of the class :class:`IncDistillationConfig`, used to
                control the tuning behavior.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `optional`):
                Tokenizer used to preprocess the data.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function which will be combined with distillation.
        Returns:
            distiller: IncDistillation object.
        """
        from neural_compressor.conf.config import Distillation_Conf

        self.config = (
            config_path_or_obj.config
            if isinstance(config_path_or_obj, IncDistillationConfig)
            else Distillation_Conf(config_path_or_obj)
        )
        self.model = model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self._eval_func = eval_func
        self._train_func = train_func

    @property
    def eval_func(self):
        return self._eval_func

    @property
    def train_func(self):
        return self._train_func

    @eval_func.setter
    def eval_func(self, func: Callable):
        self._eval_func = func

    @train_func.setter
    def train_func(self, func: Callable):
        self._train_func = func

    def fit(self):
        from neural_compressor.experimental import Distillation, common

        distiller = Distillation(self.config)
        distiller.model = common.Model(self.model)
        distiller.teacher_model = common.Model(self.teacher_model)

        if self._eval_func is None:
            raise ValueError("eval_func must be provided for distillation.")

        if self._train_func is None:
            raise ValueError("train_func must be provided for distillation.")

        distiller.train_func = self._train_func
        distiller.eval_func = self._eval_func

        return distiller

    @classmethod
    def from_config(
        cls,
        model_name_or_path: str,
        inc_config: Optional[Union[IncDistillationConfig, str]] = None,
        config_name: str = None,
        **kwargs
    ):
        """
        Instantiate an IncDistillation object from a configuration file which can either be hosted on huggingface.co or
        from a local directory path.

        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            inc_config (:obj:`Union[IncDistillationConfig, str]`, `optional`):
                Configuration file containing all the information related to the distillation.
                Can be either:
                    - an instance of the class :class:`IncDistillationConfig`,
                    - a string valid as input to :func:`IncDistillationConfig.from_pretrained`.
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
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function which will be combined with distillation.
        Returns:
            distiller: IncDistillation object.
        """
        from transformers import AutoTokenizer

        config_kwargs_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None),
        ]
        config_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in config_kwargs_default}
        distiller_kwargs_names = ["eval_func", "train_func"]
        distiller_kwargs = {name: kwargs.pop(name, None) for name in distiller_kwargs_names}

        if not isinstance(inc_config, IncDistillationConfig):
            config_path = inc_config if inc_config is not None else model_name_or_path
            inc_config = IncDistillationConfig.from_pretrained(
                config_path,
                config_file_name=config_name,
                **config_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(model_name_or_path, **kwargs)
        distiller_kwargs["tokenizer"] = tokenizer
        distiller = cls(model, inc_config, **distiller_kwargs)
        return distiller


class IncDistillationForQuestionAnswering(IncDistillation):
    from transformers import AutoModelForQuestionAnswering

    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncDistillationForSequenceClassification(IncDistillation):
    from transformers import AutoModelForSequenceClassification

    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class IncDistillationForTokenClassification(IncDistillation):
    from transformers import AutoModelForTokenClassification

    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


class IncDistillationForMultipleChoice(IncDistillation):
    from transformers import AutoModelForMultipleChoice

    TRANSFORMERS_AUTO_CLASS = AutoModelForMultipleChoice


class IncDistillationForSeq2SeqLM(IncDistillation):
    from transformers import AutoModelForSeq2SeqLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForSeq2SeqLM


class IncDistillationForCausalLM(IncDistillation):
    from transformers import AutoModelForCausalLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForCausalLM


class IncDistillationForMaskedLM(IncDistillation):
    from transformers import AutoModelForMaskedLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForMaskedLM


class IncDistillationForXLNetLM(IncDistillation):
    from transformers import XLNetLMHeadModel

    TRANSFORMERS_AUTO_CLASS = XLNetLMHeadModel
