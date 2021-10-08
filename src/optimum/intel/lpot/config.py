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


class LpotConfig:

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
            config: LpotConfig object.
        """

        self.path = config_path
        self.config = self._read_config()
        self.save_path = save_path
        self.overwrite = overwrite

    def _read_config(self):
        with open(self.path, 'r') as f:
            try:
                config = yaml.load(f, Loader=yaml.Loader)
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
        Instantiate a LpotConfig object from a configuration file which can either be hosted on
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
                config_kwargs will be passed to the LpotConfig object during initialization.
        Returns:
            config: LpotConfig object.
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

