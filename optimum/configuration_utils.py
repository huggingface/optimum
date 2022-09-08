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
""" Configuration base class. """

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from packaging import version
from transformers import PretrainedConfig
from transformers import __version__ as transformers_version
from transformers.utils import is_offline_mode

from huggingface_hub import hf_hub_download

from .utils import logging
from .version import __version__


logger = logging.get_logger(__name__)


class BaseConfig(PretrainedConfig):
    """
    Base class for configuration classes that need to respect the same API than PretrainedConfig but with a different
    configuration file name.
    """

    CONFIG_NAME = "config.json"
    FULL_CONFIGURATION_FILE = "config.json"

    @classmethod
    def _re_configuration_file(cls):
        return re.compile(rf"{cls.FULL_CONFIGURATION_FILE.split('.')[0]}(.*)\.json")

    # Adapted from transformers.configuration_utils.PretrainedConfig.save_pretrained
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, self.CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Configuration pushed to the hub in this commit: {url}")

    # Adapted from transformers.configuration_utils.PretrainedConfig.get_configuration_file
    @classmethod
    def get_configuration_file(cls, configuration_files: List[str]) -> str:
        """
        Get the configuration file to use for this version of transformers.
        Args:
            configuration_files (`List[str]`): The list of available configuration files.
        Returns:
            `str`: The configuration file to use.
        """
        configuration_files_map = {}
        _re_configuration_file = cls._re_configuration_file()
        for file_name in configuration_files:
            search = _re_configuration_file.search(file_name)
            if search is not None:
                v = search.groups()[0]
                configuration_files_map[v] = file_name
        available_versions = sorted(configuration_files_map.keys())

        configuration_file = cls.CONFIG_NAME
        optimum_version = version.parse(__version__)
        for v in available_versions:
            if version.parse(v) <= optimum_version:
                configuration_file = configuration_files_map[v]
            else:
                # No point going further since the versions are sorted.
                break

        return configuration_file

    # Adapted from transformers.configuration_utils.PretrainedConfig.get_config_dict
    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.
        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.
        """
        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            configuration_file = cls.get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    # Adapted from transformers.configuration_utils.PretrainedConfig._get_config_dict
    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PretrainedConfig` using ``from_dict``.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        configuration_file = kwargs.pop("_configuration_file", cls.CONFIG_NAME)
        is_local = os.path.isdir(pretrained_model_name_or_path)

        if os.path.isfile(pretrained_model_name_or_path):
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        elif os.path.isdir(pretrained_model_name_or_path):
            resolved_config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
            if not os.path.isfile(resolved_config_file):
                raise EnvironmentError(
                    f"Could not locate {configuration_file} inside {pretrained_model_name_or_path}."
                )
        else:
            try:
                resolved_config_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=configuration_file,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    local_files_only=local_files_only,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it "
                    "from 'https://huggingface.co/models', make sure you don't have a local directory with the same "
                    f"name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {configuration_file} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )
        if is_local:
            logger.info(f"Loading configuration file {resolved_config_file}")
        else:
            logger.info(f"Loading configuration file from cache at {resolved_config_file}")

        return config_dict, kwargs

    # Adapted from transformers.configuration_utils.PretrainedConfig.from_dict
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    # Adapted from transformers.configuration_utils.PretrainedConfig.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = transformers_version
        output["optimum_version"] = __version__

        self.dict_torch_dtype_to_str(output)

        return output
