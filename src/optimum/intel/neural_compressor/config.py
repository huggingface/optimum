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
import requests
import yaml
from functools import reduce
from huggingface_hub import hf_hub_download
from typing import Any, Optional, Union
from neural_compressor.conf.config import Quantization_Conf


class IncConfig:

    def __init__(
            self,
            config_path: str,
    ):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
        Returns:
            config: IncConfig object.
        """
        self.path = config_path
        self.config = Quantization_Conf(config_path)

    def get_config(self, keys: str):
        return reduce(lambda d, key: d.get(key) if d else None, keys.split("."),
                      self.config.usr_cfg)

    def set_config(self, keys: str, value: Any):
        d = self.config.usr_cfg
        keys = keys.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    @classmethod
    def from_pretrained(
            cls,
            config_name_or_path: Union[str, os.PathLike],
            config_file_name: str,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            **config_kwargs
    ):
        """
        Instantiate an IncConfig object from a configuration file which can either be hosted on
        huggingface.co or from a local directory path.

        Args:
            config_name_or_path (:obj:`Union[str, os.PathLike]`):
                Repository name in the Hub or path to a local directory containing the configuration file.
            config_file_name (:obj:`str`):
                Name of the configuration file.
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            config_kwargs (:obj:`Dict`, `optional`):
                config_kwargs will be passed to the IncConfig object during initialization.
        Returns:
            config: IncConfig object.
        """

        revision = None
        if len(config_name_or_path.split("@")) == 2:
            config_name_or_path, revision = config_name_or_path.split("@")

        if os.path.isdir(config_name_or_path) and config_file_name in os.listdir(config_name_or_path):
            config_file = os.path.join(config_name_or_path, config_file_name)
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=config_name_or_path,
                    filename=config_file_name,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            except requests.exceptions.RequestException:
                raise ValueError(f"{config_file_name} NOT FOUND in HuggingFace Hub")

        config = cls(config_file, **config_kwargs)
        return config
