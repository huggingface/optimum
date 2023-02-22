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


from typing import TYPE_CHECKING, List, Optional, Union, Dict, Tuple, Any


if TYPE_CHECKING:
    from datasets import Dataset, Metric
    from transformers import FeatureExtractionMixin, Pipeline, PretrainedConfig, PreTrainedTokenizerBase


class DatasetProcessing:
    DEFAULT_DATA_KEYS = None
    DEFAULT_REF_KEYS = None

    def __init__(
        self,
        dataset_name_or_path: str,
        eval_split: str,
        preprocessor: Union["FeatureExtractionMixin", "PreTrainedTokenizerBase"],
        static_quantization: bool,
        config: "PretrainedConfig",
        dataset_name: Optional[str] = None,
        data_keys: Optional[Dict[str, str]] = None,
        ref_keys: Optional[List[str]] = None,
        task_args: Optional[Dict] = None,
        num_calibration_samples: Optional[int] = None,
        calibration_split: Optional[str] = None,
        max_eval_samples: Optional[int] = None,
    ):
        """
        Initializes the class in charge of loading datasets, running inference and evaluation.

        This class should be task-dependent, backend independent.

        Args:
            dataset_path (`str`): 
                The path of the dataset, for more information read 
                https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/loading_methods#datasets.load_dataset.path
            dataset_name (`str`): 
                The name of the dataset, fore more information read 
                https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/loading_methods#datasets.load_dataset.name
            preprocessor (`Union[FeatureExtractionMixin, PreTrainedTokenizerBase]`): 
                Preprocessor used for evaluation.
            eval_split (`str`): 
                Dataset split used for evaluation (e.g. "test").
            static_quantization (`bool`): 
                Whether the dataset is used for static quantization or not.
            config ([`PretrainedConfig`]): 
                Model configuration, useful for some tasks.
            data_keys (`Dict[str, str]`): 
                Map "primary" and "secondary" to data column names.
            ref_keys (`List[str]`): 
                References column names.
            task_args(`Optional[Dict[str, Any]]`, defaults to `None`): 
                Task-specific arguments.
            num_calibration_samples (`Optional[int]`, defaults to `None`): 
                Number of calibration samples for static quantization. If left unspecified, the whole dataset will be 
                used.
            calibration_split (`Optional[str]`, defaults to `None`): 
                Calibration split (e.g. "train") for static quantization. Defaults to None.
            max_eval_samples (`Optional[int]`; defaults to `None`): 
                Maximum number of samples to use from the evaluation dataset for evaluation.
        """

        if data_keys is None and self.DEFAULT_DATA_KEYS is None:
            raise ValueError(
                f"Data keys need to be specified since no default data keys exist for {self.__class__.__name__}"
            )
        elif data_keys is None:
            data_keys = self.DEFAULT_DATA_KEYS

        if ref_keys is None and self.DEFAULT_REF_KEYS is None:
            raise ValueError(
                f"Ref keys need to be specified since no default data keys exist for {self.__class__.__name__}"
            )
        elif ref_keys is None:
            ref_keys = self.DEFAULT_REF_KEYS

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.calibration_split = calibration_split
        self.eval_split = eval_split
        self.preprocessor = preprocessor
        self.num_calibration_samples = num_calibration_samples
        self.static_quantization = static_quantization
        self.data_keys = data_keys
        self.ref_keys = ref_keys
        self.task_args = task_args
        self.config = config
        self.max_eval_samples = max_eval_samples

        if len(ref_keys) != 1:
            raise ValueError("Only one label column is supported for now.")

        if self.static_quantization and not self.calibration_split:
            raise ValueError(f"A calibration dataset split must be provided when performing static quantization.")

    def load_datasets(self) -> Dict[str, "Dataset"]:
        """
        Loads the calibration dataset if needed, and the evaluation dataset.

        The evaluation dataset is meant to be used by a pipeline and is therefore not preprocessed. The calibration 
        dataset is preprocessed.

        Returns:
            `Dict[str, Dataset]`: Dictionary holding the datasets.
        """
        raise NotImplementedError()

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
