import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from transformers import AutoConfig

import requests
from huggingface_hub import HfApi, HfFolder, hf_hub_download

from .utils import CONFIG_NAME


logger = logging.getLogger(__name__)


class OptimizedModel(ABC):
    config_class = AutoConfig
    load_tf_weights = None
    base_model_prefix = "optimized_model"

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__()
        self.model = model
        self.config = config

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model, needs to be overwritten.
        """
        pass

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~onnxruntime.modeling_ort.OptimizedModel.from_pretrained`] class method.
        Arguments:
            save_directory (`str` or `os.PathLike`):
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

        # Save the config
        self.config.save_pretrained(save_directory)

        # saving model weights/files
        self._save_pretrained(save_directory, **kwargs)

        if push_to_hub:
            return self.push_to_hub(save_directory, **kwargs)

    @abstractmethod
    def _save_pretrained(self, save_directory, **kwargs):
        """
        Save a model weights into a directory, so that it can be re-loaded using the
        [`~onnxruntime.modeling_ort.OptimizedModel.from_pretrained`] class method.
        """
        pass

    def push_to_hub(
        self,
        save_directory: str = None,
        repository_id: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        if isinstance(use_auth_token, str):
            huggingface_token = use_auth_token
        elif use_auth_token:
            huggingface_token = HfFolder.get_token()
        else:
            raise ValueError("You need to proivde `use_auth_token` to be able to push to the hub")
        api = HfApi()

        user = api.whoami(huggingface_token)
        self.git_config_username_and_email(git_email=user["email"], git_user=user["fullname"])

        api.create_repo(
            token=huggingface_token,
            name=repository_id,
            organization=user["name"],
            exist_ok=True,
            private=private,
        )
        for path, subdirs, files in os.walk(save_directory):
            for name in files:
                local_file_path = os.path.join(path, name)
                _, hub_file_path = os.path.split(local_file_path)
                # FIXME: when huggingface_hub fixes the return of upload_file
                try:
                    api.upload_file(
                        token=huggingface_token,
                        repo_id=f"{user['name']}/{repository_id}",
                        path_or_fileobj=os.path.join(os.getcwd(), local_file_path),
                        path_in_repo=hub_file_path,
                    )
                except KeyError:
                    pass
                except NameError:
                    pass

    def git_config_username_and_email(self, git_user: str = None, git_email: str = None):
        """
        Set git user name and email (only in the current repo)
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
    def _from_pretrained(
        cls,
        model_id: Union[str, os.PathLike],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """Overwrite this method in subclass to define how to load your model from pretrained"""
        raise NotImplementedError("Overwrite this method in subclass to define how to load your model from pretrained")

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        from_transformers: bool = False,
        force_download: bool = True,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **model_kwargs,
    ):
        """Instantiate a pretrained model from a pre-trained model configuration.

        Arguments:
            model_id (`Union[str, Path]`):
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing a model saved using [`~OptimizedModel.save_pretrained`],
                      e.g., `./my_model_directory/`.
            from_transformers (`bool`, *optional*, defaults to `False`):
                Defines whether the provided `model_id` contains a vanilla Transformers checkpoint.
            force_download (`bool`, *optional*, defaults to `True`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            use_auth_token (`str`, *optional*, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            cache_dir (`str`, *optional*, defaults to `None`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.

        Returns:
            `OptimizedModel`: The loaded optimized model.
        """
        revision = None
        if len(str(model_id).split("@")) == 2:
            model_id, revision = model_id.split("@")

        if os.path.isdir(model_id) and CONFIG_NAME in os.listdir(model_id):
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    use_auth_token=use_auth_token,
                )
            except requests.exceptions.RequestException:
                logger.warning("config.json NOT FOUND in HuggingFace Hub")
                config_file = None

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update({"config": config})

        if from_transformers:
            return cls._from_transformers(
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                use_auth_token=use_auth_token,
                **model_kwargs,
            )
        else:
            return cls._from_pretrained(
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                use_auth_token=use_auth_token,
                **model_kwargs,
            )

    @classmethod
    def _from_transformers(
        cls,
        model_id: Union[str, os.PathLike],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """Overwrite this method in subclass to define how to load your model from vanilla transformers model"""
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to load your model from vanilla transformers model"
        )
