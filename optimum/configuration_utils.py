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

import copy
import json
import os
import re
from typing import Any, Dict, Optional, Tuple, Union

from packaging import version
from transformers import PretrainedConfig

from trasnformers.file_utils import cached_path, get_list_of_files, hf_bucket_url, is_offline_mode, is_remote_url

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
    _RE_CONFIGURATION_FILE = re.compile(rf"{FULL_CONFIGURATION_FILE}\.(.*)\.json")

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

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

    @classmethod
    def get_configuration_file(
        cls,
        path_or_repo: Union[str, os.PathLike],
        revision: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        local_files_only: bool = False,
    ) -> str:
        """
        Get the configuration file to use for this version of transformers.

        Args:
            path_or_repo (:obj:`str` or :obj:`os.PathLike`):
                Can be either the id of a repo on huggingface.co or a path to a `directory`.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            local_files_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only rely on local files and not to attempt to download any files.

        Returns:
            :obj:`str`: The configuration file to use.
        """
        # Inspect all files from the repo/folder.
        all_files = get_list_of_files(
            path_or_repo, revision=revision, use_auth_token=use_auth_token, local_files_only=local_files_only
        )
        configuration_files_map = {}
        for file_name in all_files:
            search = cls._RE_CONFIGURATION_FILE.search(file_name)
            if search is not None:
                v = search.groups()[0]
                configuration_files_map[v] = file_name
        available_versions = sorted(configuration_files_map.keys())

        # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
        configuration_file = cls.FULL_CONFIGURATION_FILE
        transformers_version = version.parse(__version__)
        for v in available_versions:
            if version.parse(v) <= transformers_version:
                configuration_file = configuration_files_map[v]
            else:
                # No point going further since the versions are sorted.
                break

        return configuration_file

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PretrainedConfig` using ``from_dict``.



        Parameters:
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
        if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            configuration_file = cls.get_configuration_file(
                pretrained_model_name_or_path,
                revision=revision,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )

            if os.path.isdir(pretrained_model_name_or_path):
                config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
            else:
                config_file = hf_bucket_url(
                    pretrained_model_name_or_path, filename=configuration_file, revision=revision, mirror=None
                )

        try:
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
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)

        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n"
                f"  (make sure '{pretrained_model_name_or_path}' is not a path to a local directory with something else, in that case)\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {self.CONFIG_NAME} file\n\n"
            )

            if revision is not None:
                msg += f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"

            raise EnvironmentError(msg)

        except (json.JSONDecodeError, UnicodeDecodeError):
            msg = (
                f"Couldn't reach server at '{config_file}' to download configuration file or "
                "configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_config_file}."
            )
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info(f"loading configuration file {config_file}")
        else:
            logger.info(f"loading configuration file {config_file} from cache at {resolved_config_file}")

        return config_dict, kwargs

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
        output["optimum_version"] = __version__

        self.dict_torch_dtype_to_str(output)

        return output
