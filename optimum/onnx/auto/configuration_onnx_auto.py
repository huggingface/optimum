# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto ONNX Config class."""
import re
import warnings
import importlib
from typing import List, Union
from tarfile import SUPPORTED_TYPES
from collections import OrderedDict

from transformers.utils import logging
from transformers.file_utils import CONFIG_NAME
from transformers.onnx.config import OnnxConfig
from transformers.onnx.features import FeaturesManager
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.dynamic import get_class_from_dynamic_module
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING,
    MODEL_NAMES_MAPPING,
    AutoConfig,
    _get_class_name,
    _LazyConfigMapping,
)


logger = logging.get_logger(__name__)

ONNX_SUPPORTED_MODELS = FeaturesManager._SUPPORTED_MODEL_TYPE.keys()

ONNX_CONFIG_MAPPING_NAMES = OrderedDict(
    [
        (key, FeaturesManager._SUPPORTED_MODEL_TYPE[key]["default"].func.__self__.__name__)
        for key in ONNX_SUPPORTED_MODELS
        if key in CONFIG_MAPPING.keys()
    ]
)

ONNX_MODEL_NAMES_MAPPING = OrderedDict(
    [(key, MODEL_NAMES_MAPPING[key]) for key in ONNX_SUPPORTED_MODELS if key in CONFIG_MAPPING.keys()]
)


def config_class_to_model_type(config):
    """Converts a onnx config class name to the corresponding model type"""
    for key, cls in ONNX_CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    return None


ONNX_CONFIG_MAPPING = _LazyConfigMapping(ONNX_CONFIG_MAPPING_NAMES)


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {
                model_type: f"[`{config}`]" for model_type, config in ONNX_CONFIG_MAPPING_NAMES.items()
            }
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in ONNX_MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({ONNX_MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            ONNX_CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in ONNX_CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: ONNX_MODEL_NAMES_MAPPING[model_type] for model_type, config in ONNX_CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoOnnxConfig(AutoConfig):
    r"""
    This is an Onnx configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoOnnxConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in ONNX_CONFIG_MAPPING:
            config_class = ONNX_CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(ONNX_CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, task="default", **kwargs):
        r"""
        Instantiate one of the onnx configuration classes of the library from a pretrained model configuration.
        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:
        List options
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:
                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                      namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - A path or url to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.
                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.
        Examples:
        ```python
        >>> from optimum.onnx.auto.configuration_onnx_auto import AutoOnnxConfig
        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoOnnxConfig.from_pretrained("bert-base-uncased", task="default")
        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoOnnxConfig.from_pretrained("dbmdz/bert-base-german-cased", task="default")
        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoOnnxConfig.from_pretrained("./test/bert_saved_model/", task="default")
        >>> # Load a specific configuration file.
        >>> config = AutoOnnxConfig.from_pretrained("./test/bert_saved_model/my_configuration.json", task="default")
        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoOnnxConfig.from_pretrained("bert-base-uncased", task="default", output_attentions=True, foo=False)
        >>> config.output_attentions
        True
        >>> config, unused_kwargs = AutoOnnxConfig.from_pretrained(
        ...     "bert-base-uncased", task="default", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True
        >>> config.unused_kwargs
        {'foo': False}
        ```"""
        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]:
            # When given a custom configuaration
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the configuration file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if kwargs.get("revision", None) is None:
                logger.warn(
                    "Explicitly passing a `revision` is encouraged when loading a configuration with custom code to "
                    "ensure no malicious code has been contributed in a newer revision."
                )
            class_ref = config_dict["auto_map"]["AutoConfig"]
            module_file, class_name = class_ref.split(".")

            if config_dict["auto_map"]["AutoOnnxConfig"]:  # TODO: to confirm the term name
                onnx_class_ref = config_dict["auto_map"]["AutoOnnxConfig"]
                onnx_module_file, onnx_class_name = onnx_class_ref.split(".")
            else:
                onnx_module_file = module_file
                onnx_class_name = re.sub(r"(.*)(Config)", r"\1Onnx\2", class_name)

            config_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, module_file + ".py", class_name, **kwargs
            )
            onnx_config_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, onnx_module_file + ".py", onnx_class_name, **kwargs
            )
            # TODO: action when `task` is not supported by the model
            return onnx_config_class(config_class.from_pretrained(pretrained_model_name_or_path, **kwargs), task=task)
        elif "model_type" in config_dict:
            # TODO: action when `model_type` doesn't have defined OnnxConfig
            config_class = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            onnx_config_class = FeaturesManager._SUPPORTED_MODEL_TYPE[config_dict["model_type"]][task]
            return onnx_config_class(config_class)
        else:
            # Fallback: use pattern matching on the string.
            for pattern in ONNX_CONFIG_MAPPING.keys():
                if pattern in str(pretrained_model_name_or_path):
                    config_class = CONFIG_MAPPING[pattern]
                    onnx_config_class = ONNX_CONFIG_MAPPING[pattern]
                    return onnx_config_class(config_class.from_dict(config_dict, **kwargs), task=task)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {ONNX_CONFIG_MAPPING}, or contain one of the following strings "
            f"in its name: {', '.join(ONNX_CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register(model_type, config):
        """
        Register a new onnx configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`OnnxConfig`]): The onnx config to register.
        """
        if issubclass(config, OnnxConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        ONNX_CONFIG_MAPPING.register(model_type, config)
