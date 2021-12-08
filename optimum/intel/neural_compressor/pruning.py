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

from optimum.intel.neural_compressor.config import IncOptimizedConfig, IncPruningConfig
from optimum.intel.neural_compressor.utils import IncDataLoader


logger = logging.getLogger(__name__)


class IncPruningMode(Enum):
    MAGNITUDE = "basic_magnitude"


SUPPORTED_PRUNING_MODE = set([approach.value for approach in IncPruningMode])


class IncPruner:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        config_path_or_obj: Union[str, IncPruningConfig],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                Model to prune.
            config_path_or_obj (:obj:`Union[str, IncPruningConfig]` ):
                Path to the YAML configuration file or an instance of the class :class:`IncPruningConfig`, used to
                control the tuning behavior.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `optional`):
                Tokenizer used to preprocess the data.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function which will be combined with pruning.
        Returns:
            pruner: IncPruner object.
        """
        from neural_compressor.conf.config import Pruning_Conf

        self.config = (
            config_path_or_obj.config
            if isinstance(config_path_or_obj, IncPruningConfig)
            else Pruning_Conf(config_path_or_obj)
        )
        self.model = model
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
        from neural_compressor.experimental import Pruning, common

        pruner = Pruning(self.config)
        pruner.model = common.Model(self.model)

        if self._eval_func is None:
            raise ValueError("eval_func must be provided for pruning.")

        if self._train_func is None:
            raise ValueError("train_func must be provided for pruning.")

        pruner.pruning_func = self._train_func
        pruner.eval_func = self._eval_func

        return pruner

    @classmethod
    def from_config(
        cls,
        model_name_or_path: str,
        inc_config: Optional[Union[IncPruningConfig, str]] = None,
        config_name: str = None,
        **kwargs
    ):
        """
        Instantiate an IncPruner object from a configuration file which can either be hosted on huggingface.co or
        from a local directory path.

        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            inc_config (:obj:`Union[IncPruningConfig, str]`, `optional`):
                Configuration file containing all the information related to the pruning strategy.
                Can be either:
                    - an instance of the class :class:`IncPruningConfig`,
                    - a string valid as input to :func:`IncPruningConfig.from_pretrained`.
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
                Training function which will be combined with pruning.
        Returns:
            pruner: IncPruner object.
        """
        from transformers import AutoTokenizer

        config_kwargs_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None),
        ]
        config_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in config_kwargs_default}
        pruner_kwargs_names = ["eval_func", "train_func"]
        pruner_kwargs = {name: kwargs.pop(name, None) for name in pruner_kwargs_names}

        if not isinstance(inc_config, IncPruningConfig):
            config_path = inc_config if inc_config is not None else model_name_or_path
            inc_config = IncPruningConfig.from_pretrained(
                config_path,
                config_file_name=config_name,
                **config_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(model_name_or_path, **kwargs)
        pruner_kwargs["tokenizer"] = tokenizer
        pruner = cls(model, inc_config, **pruner_kwargs)
        return pruner


class IncPrunerForQuestionAnswering(IncPruner):
    from transformers import AutoModelForQuestionAnswering

    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncPrunerForSequenceClassification(IncPruner):
    from transformers import AutoModelForSequenceClassification

    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class IncPrunerForTokenClassification(IncPruner):
    from transformers import AutoModelForTokenClassification

    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


class IncPrunerForMultipleChoice(IncPruner):
    from transformers import AutoModelForMultipleChoice

    TRANSFORMERS_AUTO_CLASS = AutoModelForMultipleChoice


class IncPrunerForSeq2SeqLM(IncPruner):
    from transformers import AutoModelForSeq2SeqLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForSeq2SeqLM


class IncPrunerForCausalLM(IncPruner):
    from transformers import AutoModelForCausalLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForCausalLM


class IncPrunerForMaskedLM(IncPruner):
    from transformers import AutoModelForMaskedLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForMaskedLM


class IncPrunerForXLNetLM(IncPruner):
    from transformers import XLNetLMHeadModel

    TRANSFORMERS_AUTO_CLASS = XLNetLMHeadModel
