# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base class for optimized model inference wrapping."""

import logging
import os
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from huggingface_hub import HfApi
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoConfig, PretrainedConfig, add_start_docstrings
from transformers.utils import http_user_agent

from .exporters import TasksManager
from .utils import CONFIG_NAME


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel


logger = logging.getLogger(__name__)

FROM_PRETRAINED_START_DOCSTRING = r"""
    Instantiate a pretrained model from a pre-trained model configuration.

    Args:
        model_id (`Union[str, Path]`):
            Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing a model saved using [`~OptimizedModel.save_pretrained`],
                    e.g., `./my_model_directory/`.
        export (`bool`, defaults to `False`):
            Defines whether the provided `model_id` needs to be exported to the targeted format.
        force_download (`bool`, defaults to `True`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
            Deprecated. Please use the `token` argument instead.
        token (`Optional[Union[bool,str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
        cache_dir (`Optional[str]`, defaults to `None`):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        config (`Optional[transformers.PretrainedConfig]`, defaults to `None`):
            The model configuration.
        local_files_only (`Optional[bool]`, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for custom code defined on the Hub in their own modeling. This option should only be set
            to `True` for repositories you trust and in which you have read the code, as it will execute code present on
            the Hub on your local machine.
        revision (`Optional[str]`, defaults to `None`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
"""


# workaround to enable compatibility between optimum models and transformers pipelines
class PreTrainedModel(ABC):  # noqa: F811
    pass


class OptimizedModel(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "optimized_model"
    config_name = CONFIG_NAME

    def __init__(self, model: Union["PreTrainedModel", "TFPreTrainedModel"], config: PretrainedConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.preprocessors = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model, needs to be overwritten.
        """
        raise NotImplementedError

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.

        Args:
            save_directory (`Union[str, os.PathLike]`):
                Directory to which to save. Will be created if it doesn't exist.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                <Tip warning={true}>

                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.

                </Tip>
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        self._save_config(save_directory)
        for preprocessor in self.preprocessors:
            preprocessor.save_pretrained(save_directory)
        self._save_pretrained(save_directory)

        if push_to_hub:
            return self.push_to_hub(save_directory, **kwargs)

    @abstractmethod
    def _save_pretrained(self, save_directory):
        """
        Saves a model weights into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        raise NotImplementedError

    def _save_config(self, save_directory):
        """
        Saves a model configuration into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        self.config.save_pretrained(save_directory)

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> str:
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        hf_api = HfApi(user_agent=http_user_agent(), token=token)

        hf_api.create_repo(
            repo_id=repository_id,
            private=private,
            exist_ok=True,
            token=token,
        )

        for path, subdirs, files in os.walk(save_directory):
            for name in files:
                local_file_path = os.path.join(path, name)
                _, hub_file_path = os.path.split(local_file_path)
                # FIXME: when huggingface_hub fixes the return of upload_file
                try:
                    hf_api.upload_file(
                        token=token,
                        repo_id=f"{repository_id}",
                        path_or_fileobj=os.path.join(os.getcwd(), local_file_path),
                        path_in_repo=hub_file_path,
                    )
                except KeyError:
                    pass
                except NameError:
                    pass

    def git_config_username_and_email(self, git_user: str = None, git_email: str = None):
        """
        Sets git user name and email (only in the current repo)
        """
        try:
            if git_user is not None:
                subprocess.run(
                    ["git", "config", "--global", "user.name", git_user],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                )
            if git_email is not None:
                subprocess.run(
                    ["git", "config", "--global", "user.email", git_email],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    @classmethod
    def _load_config(
        cls,
        config_name_or_path: Union[str, os.PathLike],
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        force_download: bool = False,
        subfolder: str = "",
        trust_remote_code: bool = False,
    ) -> PretrainedConfig:
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=config_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                subfolder=subfolder,
                trust_remote_code=trust_remote_code,
            )
        except OSError as e:
            # if config not found in subfolder, search for it at the top level
            if subfolder != "":
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=config_name_or_path,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    trust_remote_code=trust_remote_code,
                )
                logger.info(
                    f"config.json not found in the specified subfolder {subfolder}. Using the top level config.json."
                )
            else:
                raise OSError(e)
        return config

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        **kwargs,
    ) -> "OptimizedModel":
        """Overwrite this method in subclass to define how to load your model from pretrained"""
        raise NotImplementedError("Overwrite this method in subclass to define how to load your model from pretrained")

    @classmethod
    def _from_transformers(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OptimizedModel":
        """Overwrite this method in subclass to define how to load your model from vanilla transformers model"""
        raise NotImplementedError(
            "`_from_transformers` method will be deprecated in a future release. Please override `_export` instead"
            "to define how to load your model from vanilla transformers model"
        )

    @classmethod
    def _export(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OptimizedModel":
        """Overwrite this method in subclass to define how to load your model from vanilla hugging face model"""
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to load your model from vanilla hugging face model"
        )

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        config: Optional[PretrainedConfig] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "OptimizedModel":
        """
        Returns:
            `OptimizedModel`: The loaded optimized model.
        """

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if isinstance(model_id, Path):
            model_id = model_id.as_posix()

        from_transformers = kwargs.pop("from_transformers", None)
        if from_transformers is not None:
            logger.warning(
                "The argument `from_transformers` is deprecated, and will be removed in optimum 2.0.  Use `export` instead"
            )
            export = from_transformers

        if len(model_id.split("@")) == 2:
            logger.warning(
                f"Specifying the `revision` as @{model_id.split('@')[1]} is deprecated and will be removed in v1.23, please use the `revision` argument instead."
            )
            if revision is not None:
                logger.warning(
                    f"The argument `revision` was set to {revision} but will be ignored for {model_id.split('@')[1]}"
                )
            model_id, revision = model_id.split("@")

        all_files, _ = TasksManager.get_model_files(
            model_id,
            subfolder=subfolder,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )

        config_folder = subfolder
        if cls.config_name not in all_files:
            logger.info(
                f"{cls.config_name} not found in the specified subfolder {subfolder}. Using the top level {cls.config_name}."
            )
            config_folder = ""

        library_name = TasksManager.infer_library_from_model(
            model_id, subfolder=config_folder, revision=revision, cache_dir=cache_dir, token=token
        )

        if library_name == "timm":
            config = PretrainedConfig.from_pretrained(
                model_id, subfolder=config_folder, revision=revision, cache_dir=cache_dir, token=token
            )

        if config is None:
            if os.path.isdir(os.path.join(model_id, config_folder)) and cls.config_name == CONFIG_NAME:
                if CONFIG_NAME in os.listdir(os.path.join(model_id, config_folder)):
                    config = AutoConfig.from_pretrained(
                        os.path.join(model_id, config_folder), trust_remote_code=trust_remote_code
                    )
                else:
                    raise OSError(f"config.json not found in {model_id} local folder")
            else:
                config = cls._load_config(
                    model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    force_download=force_download,
                    subfolder=config_folder,
                    trust_remote_code=trust_remote_code,
                )
        elif isinstance(config, (str, os.PathLike)):
            config = cls._load_config(
                config,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
                subfolder=config_folder,
                trust_remote_code=trust_remote_code,
            )

        from_pretrained_method = cls._from_transformers if export else cls._from_pretrained

        return from_pretrained_method(
            model_id=model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
