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
"""Text classification processing utilities for preparing datasets and tokenizing text data."""

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase

from .base import TaskProcessor


if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


logger = logging.getLogger(__name__)


class TextClassificationProcessing(TaskProcessor):
    """
    Processor for text classification tasks that handles tokenization and dataset preparation.
    
    This class provides functionality to preprocess text classification datasets by tokenizing
    input text and preparing it for model training or inference. It supports both single text
    and text pair classification tasks.
    
    Attributes:
        ACCEPTED_PREPROCESSOR_CLASSES: Tuple of accepted tokenizer classes.
        DEFAULT_DATASET_ARGS: Default arguments for loading datasets.
        DEFAUL_DATASET_DATA_KEYS: Default keys for accessing dataset text data.
        ALLOWED_DATA_KEY_NAMES: Set of allowed data key names.
        DEFAULT_REF_KEYS: Default reference keys for labels.
    """
    ACCEPTED_PREPROCESSOR_CLASSES = (PreTrainedTokenizerBase,)
    DEFAULT_DATASET_ARGS = {"path": "glue", "name": "sst2"}
    DEFAULT_DATASET_DATA_KEYS = {"primary": "sentence"}
    ALLOWED_DATA_KEY_NAMES = {"primary", "secondary"}
    DEFAULT_REF_KEYS = ["label"]

    def create_defaults_and_kwargs_from_preprocessor_kwargs(
        self, preprocessor_kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract default tokenization parameters and remaining kwargs from preprocessor arguments.
        
        Args:
            preprocessor_kwargs: Dictionary of preprocessor arguments, can be None.
            
        Returns:
            Tuple containing:
                - defaults: Dictionary with default tokenization parameters (padding, truncation, max_length)
                - kwargs: Dictionary with remaining preprocessor arguments
        """
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        kwargs = copy.deepcopy(preprocessor_kwargs)
        defaults = {}
        defaults["padding"] = kwargs.pop("padding", "max_length")
        defaults["truncation"] = kwargs.pop("truncation", True)
        defaults["max_length"] = kwargs.pop("max_length", self.preprocessor.model_max_length)
        return defaults, kwargs

    def dataset_processing_func(
        self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a single dataset example by tokenizing the text input(s).
        
        Args:
            example: Dictionary containing the dataset example data.
            data_keys: Dictionary mapping data key names to column names in the dataset.
            ref_keys: Optional list of reference keys for labels.
            
        Returns:
            Dictionary containing tokenized inputs ready for model consumption.
        """
        tokenized_inputs = self.preprocessor(
            text=example[data_keys["primary"]],
            text_pair=example[data_keys["secondary"]] if "secondary" in data_keys else None,
            **self.defaults,
            **self.preprocessor_kwargs,
        )
        return tokenized_inputs

    def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        """
        Attempt to automatically identify primary and secondary text columns in the dataset.
        
        Args:
            column_names: List of column names in the dataset.
            
        Returns:
            Dictionary mapping 'primary' and optionally 'secondary' to column names,
            or None if primary column cannot be identified.
        """
        primary_key_name = None
        primary_key_name_candidates = ["sentence", "text", "premise"]
        for name in column_names:
            if any(candidate in name for candidate in primary_key_name_candidates):
                primary_key_name = name
                break

        secondary_key_name = None
        secondary_key_name_candidates = ["hypothesis"]
        for name in column_names:
            if any(candidate in name for candidate in secondary_key_name_candidates):
                secondary_key_name = name
                break

        if primary_key_name is None:
            return None
        elif secondary_key_name is None:
            logger.info(
                "Could not infer the secondary key in the dataset, if it does contain one, please provide it manually."
            )
            return {"primary": primary_key_name}
        else:
            return {"primary": primary_key_name, "secondary": secondary_key_name}

    def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
        """
        Attempt to automatically identify label columns in the dataset.
        
        Args:
            column_names: List of column names in the dataset.
            
        Returns:
            List containing the identified label column name, or None if not found.
        """
        for name in column_names:
            if "label" in name:
                return [name]

    def load_dataset(
        self,
        path: str,
        data_keys: Optional[Dict[str, str]] = None,
        ref_keys: Optional[List[str]] = None,
        only_keep_necessary_columns: bool = False,
        load_smallest_split: bool = False,
        num_samples: Optional[int] = None,
        shuffle: bool = False,
        **load_dataset_kwargs,
    ) -> Union["DatasetDict", "Dataset"]:
        """
        Load and prepare a text classification dataset.
        
        Args:
            path: Path or name of the dataset to load.
            data_keys: Optional dictionary mapping data key names to dataset column names.
            ref_keys: Optional list of reference keys for labels.
            only_keep_necessary_columns: Whether to keep only necessary columns.
            load_smallest_split: Whether to load only the smallest dataset split.
            num_samples: Optional number of samples to load.
            shuffle: Whether to shuffle the dataset.
            **load_dataset_kwargs: Additional arguments passed to the dataset loading function.
            
        Returns:
            Loaded dataset as either a DatasetDict or Dataset object.
        """
        dataset = super().load_dataset(
            path,
            data_keys=data_keys,
            ref_keys=ref_keys,
            only_keep_necessary_columns=only_keep_necessary_columns,
            load_smallest_split=load_smallest_split,
            num_samples=num_samples,
            shuffle=shuffle,
            **load_dataset_kwargs,
        )
        # TODO: do we want to do that here?
        # eval_dataset = eval_dataset.align_labels_with_mapping(self.config.label2id, self.ref_keys[0])
        return dataset
