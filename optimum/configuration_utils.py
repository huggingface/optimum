# coding=utf-8
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
from typing import Any, Dict, List, Tuple, Union

from packaging import version
from transformers import PretrainedConfig
from transformers import __version__ as transformers_version_str

from .utils import logging
from .version import __version__


# TODO: remove once transformers release version is way above 4.22.
_transformers_version = version.parse(transformers_version_str)
_transformers_version_threshold = (4, 22)
_transformers_version_is_below_threshold = (
    _transformers_version.major,
    _transformers_version.minor,
) < _transformers_version_threshold

if _transformers_version_is_below_threshold:
    from transformers.utils import cached_path, hf_bucket_url
else:
    from transformers.dynamic_module_utils import custom_object_save
    from transformers.utils import cached_file, download_url, extract_commit_hash, is_remote_url


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
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        # TODO: remove conditon once transformers release version is way above 4.22.
        if not _transformers_version_is_below_threshold:
            os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            # TODO: remove once transformers release version is way above 4.22.
            if _transformers_version_is_below_threshold:
                repo = self._create_or_get_repo(save_directory, **kwargs)
            else:
                repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
                repo_id, token = self._create_repo(repo_id, **kwargs)
                files_timestamps = self._get_files_timestamps(save_directory)

        # TODO: remove once transformers release version is way above 4.22.
        if _transformers_version_is_below_threshold:
            os.makedirs(save_directory, exist_ok=True)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, self.CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            # TODO: remove once transformers release version is way above 4.22.
            if _transformers_version_is_below_threshold:
                url = self._push_to_hub(repo, commit_message=commit_message)
                logger.info(f"Configuration pushed to the hub in this commit: {url}")
            else:
                self._upload_modified_files(
                    save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token
                )

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

        # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
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
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            # Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        # TODO: remove once transformers release version is way above 4.22.
        elif _transformers_version_is_below_threshold and os.path.isdir(pretrained_model_name_or_path):
            configuration_file = kwargs.pop("_configuration_file", cls.CONFIG_NAME)
            resolved_config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
            if not os.path.isfile(resolved_config_file):
                raise EnvironmentError(
                    f"Could not locate {configuration_file} inside {pretrained_model_name_or_path}."
                )
        # TODO: remove condition once transformers release version is way above 4.22.
        elif not _transformers_version_is_below_threshold and is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path
            resolved_config_file = download_url(pretrained_model_name_or_path)
        else:
            configuration_file = kwargs.pop("_configuration_file", cls.CONFIG_NAME)

            try:
                # TODO: remove once transformers release version is way above 4.22.
                if _transformers_version_is_below_threshold:
                    config_file = hf_bucket_url(
                        pretrained_model_name_or_path,
                        filename=configuration_file,
                        revision=revision,
                        subfolder=subfolder if len(subfolder) > 0 else None,
                        mirror=None,
                    )
                    # Load from URL or cache if already cached
                    resolved_config_file = cached_path(
                        config_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
                else:
                    # Load from local folder or from cache or download from model Hub and cache
                    resolved_config_file = cached_file(
                        pretrained_model_name_or_path,
                        configuration_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _commit_hash=commit_hash,
                    )
                    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
            # TODO: remove once transformers release version is way above 4.22.
            if _transformers_version_is_below_threshold:
                config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        return config_dict, kwargs

    # Adapted from transformers.configuration_utils.PretrainedConfig.from_dict
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
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
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]

        # Transformers version when serializing the model
        output["transformers_version"] = transformers_version_str
        output["optimum_version"] = __version__

        self.dict_torch_dtype_to_str(output)

        return output
