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

import itertools
import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union, Dict, Tuple, Any, Callable, Type
from pathlib import Path

from datasets import DatasetDict, Dataset, load_dataset as datasets_load_dataset


if TYPE_CHECKING:
    from datasets import Dataset, Metric
    from transformers import FeatureExtractionMixin, Pipeline, PretrainedConfig, PreTrainedTokenizerBase


class TaskProcessing(ABC):
    ACCEPTED_PREPROCESSOR_CLASSES: Tuple[Type, ...]
    DEFAULT_DATASET_ARGS: Tuple[Any, ...]
    DEFAUL_DATASET_DATA_KEYS: Dict[str, str] 

    def __init__(
        self,
        config: "PretrainedConfig",
        preprocessor: Union["FeatureExtractionMixin", "PreTrainedTokenizerBase"],
    ):
        """
        Initializes the class in charge of loading processed datasets and of running evaluation.

        This class should be task-dependent, backend independent.

        Args:
            config (`PretrainedConfig`):
                The config of the model.
            preco
        """
        if not isinstance(preprocessor, self.ACCEPTED_PREPROCESSOR_CLASSES):
            raise ValueError(
                f"Preprocessor is incorrect, provided an instance of {type(preprocessor)} but expected one of the "
                f"following type: {', '.join(cls_.__name__ for cls_ in self.ACCEPTED_PREPROCESSOR_CLASSES)}."
            )

        self.config = config
        self.preprocessor = preprocessor


    @abstractmethod
    def dataset_processing_func(self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        raise NotImplementedError("This static method must be implemented in subclasses.")
    
    def create_dataset_processing_func(self, data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        return functools.partial(self.dataset_processing_func, data_keys=data_keys, ref_keys=ref_keys)
        
    def prepare_dataset(self, dataset: Union[DatasetDict, Dataset], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None, split: Optional[str] = None) -> Union[DatasetDict, Dataset]:
        if isinstance(dataset, Dataset) and split is not None:
            raise ValueError("A Dataset and a split name were provided, but splits are for DatasetDict.")
        elif split is not None:
            dataset = dataset[split]
        return dataset.map(self.create_dataset_processing_func(data_keys, ref_keys))

    @abstractmethod
    def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        raise NotImplementedError()

    @abstractmethod
    def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
        raise NotImplementedError()

    def load_dataset(
        self, 
        *args, 
        data_keys: Optional[Dict[str, str]] = None, 
        ref_keys: Optional[List[str]] = None, 
        only_keep_necessary_columns: bool = False,
        **kwargs,
    ) -> Union[DatasetDict, Dataset]:
        dataset = datasets_load_dataset(*args, **kwargs)
        column_names = dataset.column_names
        if isinstance(column_names, dict):
            column_names = list(set(itertools.chain.from_iterable(column_names.values())))

        if data_keys is None:
            data_keys = self.try_to_guess_data_keys(column_names)
            if data_keys is None:
                raise ValueError(
                    "Data keys need to be specified manually since they could not be guessed from "
                    f"{', '.join(column_names)}"
                )

        if ref_keys is None:
            ref_keys = self.try_to_guess_ref_keys(column_names)

        dataset = self.prepare_dataset(dataset, data_keys=data_keys, ref_keys=ref_keys)

        if only_keep_necessary_columns:
            ref_keys = ref_keys if ref_keys is not None else []
            necessary_columns = self.preprocessor.model_input_names + ref_keys
            dataset = dataset.remove_columns([name for name in dataset.column_names if name not in necessary_columns])

        return dataset

    def load_default_dataset(self, only_keep_necessary_columns: bool = False)


    def run_inference(self, eval_dataset: "Dataset", pipeline: "Pipeline") -> Tuple[List, List]:
        """
        Runs inference on the provided dataset using a pipeline, and returns all labels, predictions.

        Args:
            eval_dataset (`Dataset`): 
                Raw dataset to run inference on.
            pipeline (`Pipeline`): 
                Pipeline used for inference. Should be initialized beforehand.

        Returns:
            `Tuple[List, List]` comprising labels and predictions:
            - **labels** are the references for evaluation.
            - **predictions** are the predictions on the dataset using the pipeline.
        """
        raise NotImplementedError()

    def get_metrics(self, predictions: List, references: List, metric: "Metric") -> Dict[str, float]:
        """
        Computes a metric given pre-formatted predictions and references.

        Args:
            predictions (`List`): 
                The predictions.
            references (`List`): 
                The references to compare the predictions against.
            metric (`Metric`): 
                Pre-loaded metric to run evaluation on.

        Returns:
            `Dict[str, float]`: The computed metrics.
        """
        raise NotImplementedError()

    def get_pipeline_kwargs(self) -> Dict[str, Any]:
        """
        Gets task-specific kwargs to initialize the pipeline.

        Returns:
            `Dict[str, Any]`: Task-specific kwargs to initialize the pipeline.
        """
        raise NotImplementedError()
