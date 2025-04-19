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
"""Base class to peform task-specific preprocessing and evaluation."""

import copy
import functools
import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor

from optimum.utils.import_utils import requires_backends

from .. import logging


if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict
    from transformers import PretrainedConfig


logger = logging.get_logger(__name__)

Preprocessor = Union[PreTrainedTokenizerBase, BaseImageProcessor]


class TaskProcessor(ABC):
    ACCEPTED_PREPROCESSOR_CLASSES: Tuple[Type, ...]
    DEFAULT_DATASET_ARGS: Union[str, Dict[str, Any]]
    DEFAUL_DATASET_DATA_KEYS: Dict[str, str]
    ALLOWED_DATA_KEY_NAMES: Set[str]
    DEFAULT_REF_KEYS: List[str]

    def __init__(
        self,
        config: "PretrainedConfig",
        preprocessor: Preprocessor,
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the class in charge of loading processed datasets and of running evaluation.

        This class should be task-dependent, backend independent.

        Args:
            config (`PretrainedConfig`):
                The config of the model.
            preprocessor: (`Preprocessor`):
                The preprocessor associated to the model. This will be used to prepare the datasets.
            preprocessor_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
                Keyword arguments that will be passed to the preprocessor during dataset processing.
                This allows customizing the behavior of the preprocessor.
        """
        if not isinstance(preprocessor, self.ACCEPTED_PREPROCESSOR_CLASSES):
            raise ValueError(
                f"Preprocessor is incorrect, provided an instance of {type(preprocessor)} but expected one of the "
                f"following type: {', '.join(cls_.__name__ for cls_ in self.ACCEPTED_PREPROCESSOR_CLASSES)}."
            )

        self.config = config
        self.preprocessor = preprocessor
        self.defaults, self.preprocessor_kwargs = self.create_defaults_and_kwargs_from_preprocessor_kwargs(
            preprocessor_kwargs
        )

    def create_defaults_and_kwargs_from_preprocessor_kwargs(
        self, preprocessor_kwargs: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Takes the dictionary of the preprocessor keyword arguments and return two dictionaries:
            - The first dictionary will either contain defaults values if not specified in preprocessor_kwargs or the
            values specified in preprocessor_kwargs.
            - The second dictionary will contain the rest of the keyword arguments.
        """
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        return {}, copy.deepcopy(preprocessor_kwargs)

    @abstractmethod
    def dataset_processing_func(
        self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError("This method must be implemented in subclasses.")

    def create_dataset_processing_func(
        self, data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        return functools.partial(self.dataset_processing_func, data_keys=data_keys, ref_keys=ref_keys)

    def prepare_dataset(
        self,
        dataset: Union["DatasetDict", "Dataset"],
        data_keys: Dict[str, str],
        ref_keys: Optional[List[str]] = None,
        split: Optional[str] = None,
    ) -> Union["DatasetDict", "Dataset"]:
        requires_backends(self, ["datasets"])
        from datasets import Dataset

        if isinstance(dataset, Dataset) and split is not None:
            raise ValueError("A Dataset and a split name were provided, but splits are for DatasetDict.")
        elif split is not None:
            dataset = dataset[split]
        return dataset.map(self.create_dataset_processing_func(data_keys, ref_keys))

    @abstractmethod
    def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        raise NotImplementedError("This method must be implemented in subclasses.")

    @abstractmethod
    def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
        raise NotImplementedError("This method must be implemented in subclasses.")

    def load_dataset(
        self,
        path: str,
        data_keys: Optional[Dict[str, str]] = None,
        ref_keys: Optional[List[str]] = None,
        only_keep_necessary_columns: bool = False,
        load_smallest_split: bool = False,
        num_samples: Optional[int] = None,
        shuffle: bool = False,
        download_config=None,
        **load_dataset_kwargs,
    ) -> Union["DatasetDict", "Dataset"]:
        requires_backends(self, ["datasets"])

        from datasets import Dataset, DatasetDict, DownloadConfig
        from datasets import load_dataset as datasets_load_dataset
        from transformers.utils import http_user_agent

        if download_config is None:
            download_config = DownloadConfig(user_agent=http_user_agent())

        dataset = datasets_load_dataset(path, **load_dataset_kwargs, download_config=download_config)

        if isinstance(dataset, DatasetDict) and load_smallest_split:
            split = load_dataset_kwargs.get("split", None)
            if split is not None:
                raise ValueError(
                    f"A split name was provided ({split}) but load_smallest_split is True, use either one or the other."
                )
            smallest_split = min(dataset.items(), key=lambda item: item[1].num_rows)[0]
            logger.info(
                "Since no split was explicitely provided and load_smallest_split=True, using the smallest split of the "
                f'dataset called "{smallest_split}".'
            )
            dataset = dataset[smallest_split]

        if shuffle:
            dataset = dataset.shuffle()

        if isinstance(dataset, Dataset) and num_samples is not None:
            if num_samples > dataset.num_rows:
                raise ValueError(
                    f"There are only {dataset.num_rows} examples in the dataset, but it was requested to select "
                    f"{num_samples} examples."
                )
            dataset = dataset.select(range(num_samples))

        column_names = dataset.column_names
        if isinstance(column_names, dict):
            column_names = list(set(itertools.chain.from_iterable(column_names.values())))

        if data_keys is None:
            logger.warning("As no data keys were provided, trying to guess them.")
            data_keys = self.try_to_guess_data_keys(column_names)
            if data_keys is None:
                raise ValueError(
                    "Data keys need to be specified manually since they could not be guessed from "
                    f"{', '.join(column_names)}"
                )
            else:
                logger.warning(f"Guessed the following data keys: {data_keys}")
        elif not set(data_keys.keys()) <= self.ALLOWED_DATA_KEY_NAMES:
            raise ValueError(
                f"data_keys contains unallowed keys {set(data_keys.keys())}, allowed_keys: {self.ALLOWED_DATA_KEY_NAMES}."
            )

        if ref_keys is None:
            ref_keys = self.try_to_guess_ref_keys(column_names)
            logger.warning(f"As no ref keys were provided, tried to guess them: {ref_keys}.")

        dataset = self.prepare_dataset(dataset, data_keys=data_keys, ref_keys=ref_keys)

        if only_keep_necessary_columns:
            ref_keys = ref_keys if ref_keys is not None else []
            necessary_columns = self.preprocessor.model_input_names + ref_keys
            if isinstance(dataset, DatasetDict):
                for split_name, split in dataset.items():
                    columns_to_remove = [name for name in split.column_names if name not in necessary_columns]
                    dataset[split_name] = split.remove_columns(columns_to_remove)
            else:
                columns_to_remove = [name for name in dataset.column_names if name not in necessary_columns]
                dataset = dataset.remove_columns(columns_to_remove)

        return dataset

    def load_default_dataset(
        self,
        only_keep_necessary_columns: bool = False,
        load_smallest_split: bool = False,
        num_samples: Optional[int] = None,
        shuffle: bool = False,
        **load_dataset_kwargs,
    ):
        if isinstance(self.DEFAULT_DATASET_ARGS, dict):
            path = self.DEFAULT_DATASET_ARGS.get("path", None)
            if path is None:
                raise ValueError(
                    'When DEFAULT_DATASET_ARGS is a dictionary, it must contain a key called "path" corresponding to '
                    "the path or name of the dataset."
                )
            common_keys = set(self.DEFAULT_DATASET_ARGS.keys()) & set(load_dataset_kwargs.keys())
            if common_keys:
                ", ".join(common_keys)
                logger.warning(
                    "The following provided arguments will be overriden because they are hardcoded when using "
                    "load_default_dataset: {override_config_key}."
                )
            kwargs = copy.deepcopy(load_dataset_kwargs)
            kwargs.update({k: v for k, v in self.DEFAULT_DATASET_ARGS.items() if k != "path"})
        else:
            path = self.DEFAULT_DATASET_ARGS
            kwargs = load_dataset_kwargs

        return self.load_dataset(
            path,
            data_keys=self.DEFAUL_DATASET_DATA_KEYS,
            ref_keys=self.DEFAULT_REF_KEYS,
            only_keep_necessary_columns=only_keep_necessary_columns,
            load_smallest_split=load_smallest_split,
            num_samples=num_samples,
            shuffle=shuffle,
            **kwargs,
        )
