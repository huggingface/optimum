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
from functools import reduce
from typing import Any, Optional, Union

import requests
import yaml


logger = logging.getLogger(__name__)


class IncConfig:
    def __init__(self, config_path: str):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
        Returns:
            config: IncConfig object.
        """
        from neural_compressor.conf.config import Conf

        self.path = config_path
        self.config = Conf(config_path)
        self.usr_cfg = self.config.usr_cfg

    def get_config(self, keys: str):
        return reduce(lambda d, key: d.get(key) if d else None, keys.split("."), self.usr_cfg)

    def set_config(self, keys: str, value: Any):
        d = self.usr_cfg
        keys = keys.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def set_tolerance(self, perf_tol: Union[int, float]):
        if not isinstance(perf_tol, (int, float)):
            raise TypeError(f"Supported type for performance tolerance are int and float, got {type(perf_tol)}")
        if "absolute" in self.get_config("tuning.accuracy_criterion"):
            self.set_config("tuning.accuracy_criterion.absolute", perf_tol)
        else:
            if not -1 < perf_tol < 1:
                raise ValueError("Relative performance tolerance must not be <=-1 or >=1.")
            self.set_config("tuning.accuracy_criterion.relative", perf_tol)

    @classmethod
    def from_pretrained(cls, config_name_or_path: str, config_file_name: Optional[str] = None, **kwargs):
        """
        Instantiate an IncConfig object from a configuration file which can either be hosted on
        huggingface.co or from a local directory path.

        Args:
            config_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory containing the configuration file.
            config_file_name (:obj:`str`, `optional`):
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
        Returns:
            config: IncConfig object.
        """

        from transformers.file_utils import cached_path, hf_bucket_url

        from optimum.intel.neural_compressor.utils import CONFIG_NAME

        cache_dir = kwargs.get("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        resume_download = kwargs.get("resume_download", False)
        revision = kwargs.get("revision", None)

        config_file_name = config_file_name if config_file_name is not None else CONFIG_NAME
        if os.path.isdir(config_name_or_path):
            config_file = os.path.join(config_name_or_path, config_file_name)
        elif os.path.isfile(config_name_or_path):
            config_file = config_name_or_path
        else:
            config_file = hf_bucket_url(config_name_or_path, filename=config_file_name, revision=revision)

        try:
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
            )
        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load config for '{config_name_or_path}'. Make sure that:\n\n"
                f"-'{config_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"-or '{config_name_or_path}' is a correct path to a directory containing a {config_file_name} file\n\n"
            )

            if revision is not None:
                msg += (
                    f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that "
                    f"exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"
                )

            raise EnvironmentError(msg)

        config = cls(resolved_config_file)

        return config


class IncOptimizedConfig(IncConfig):
    def __init__(self, config_path: str):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
        Returns:
            config: IncOptimizedConfig object.
        """

        self.path = config_path
        self.config = self._read_config()
        self.usr_cfg = self.config

    def _read_config(self):
        with open(self.path, "r") as f:
            try:
                config = yaml.load(f, Loader=yaml.Loader)
            except yaml.YAMLError as err:
                logger.error(err)

        return config


class IncQuantizationConfig(IncConfig):
    def __init__(self, config_path: str):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
        Returns:
            config: IncQuantizationConfig object.
        """
        from neural_compressor.conf.config import Quantization_Conf

        self.path = config_path
        self.config = Quantization_Conf(config_path)
        self.usr_cfg = self.config.usr_cfg


class IncPruningConfig(IncConfig):
    def __init__(self, config_path: str):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
        Returns:
            config: IncPruningConfig object.
        """
        from neural_compressor.conf.config import Pruning_Conf

        self.path = config_path
        self.config = Pruning_Conf(config_path)
        self.usr_cfg = self.config.usr_cfg
