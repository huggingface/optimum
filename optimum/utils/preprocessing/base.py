from typing import Dict, List, Optional, Union

from datasets import Dataset
from transformers import FeatureExtractionMixin, Pipeline, PretrainedConfig, PreTrainedTokenizerBase


class DatasetProcessing:
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
        preprocessor: Union[FeatureExtractionMixin, PreTrainedTokenizerBase],
        eval_split: str,
        static_quantization: bool,
        data_keys: Dict[str, str],
        ref_keys: List[str],
        config: PretrainedConfig,
        task_args: Optional[Dict] = None,
        num_calibration_samples: Optional[int] = None,
        calibration_split: Optional[str] = None,
        max_eval_samples: Optional[int] = None,
    ):
        """Initialize the class in charge of loading datasets, running inference and evaluation.

        This class should be task-dependent, backend independent.

        Args:
            dataset_path (`str`): Dataset path (https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/loading_methods#datasets.load_dataset.path)
            dataset_name (`str`): Dataset name (https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/loading_methods#datasets.load_dataset.name)
            preprocessor (`Union[FeatureExtractionMixin, PreTrainedTokenizerBase]`): Preprocessor used for evaluation.
            eval_split (`str`): Dataset split used for evaluation (e.g. "test").
            static_quantization (`bool`): Static quantization is used.
            data_keys (`Dict[str, str]`): Map "primary" and "secondary" to data column names.
            ref_keys (`List[str]`): References column names.
            config (`PretrainedConfig`): Model configuration, useful for some tasks.
            task_args(`Dict`, *optional*): Task-specific arguments.
            num_calibration_samples (`int`, *optional*): Number of calibration samples for static quantization. Defaults to None.
            calibration_split (`str`, *optional*): Calibration split (e.g. "train") for static quantization. Defaults to None.
            max_eval_samples (`int`; *optional*): Maximum number of samples to use from the evaluation dataset for evaluation.
        """

        if len(ref_keys) != 1:
            raise ValueError("Only one label column is supported for now.")

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

    def load_datasets(self):
        """Load calibration dataset if needed, and evaluation dataset.

        The evaluation dataset is meant to be used by a pipeline and is therefore not preprocessed. The calibration dataset is preprocessed.

        Returns:
            `Dict`: Dictionary holding the datasets.
        """
        raise NotImplementedError()

    def run_inference(self, eval_dataset: Dataset, pipeline: Pipeline):
        """Run inference on the provided dataset using a pipeline, and return all labels, predictions.

        Args:
            eval_dataset (`Dataset`): Raw dataset to run inference on.
            pipeline (`Pipeline`): Pipeline used for inference. Should be initialized beforehand.

        Returns:
            `tuple(List)` comprising labels and predictions:
            - **labels** are the references for evaluation.
            - **predictions** are the predictions on the dataset using the pipeline.
        """
        raise NotImplementedError()

    def get_metrics(predictions, references, metric):
        """Compute a metric given pre-formatted predictions and references.

        Args:
            predictions (`List`): Predictions.
            references (`List`): References.
            metric (`Metric`): Pre-loaded metric to run evaluation on.

        Returns:
            `Dict`: Computed metrics.
        """
        raise NotImplementedError()

    def get_pipeline_kwargs(self):
        """Get task-specific kwargs to initialize the pipeline.

        Returns:
            `Dict`: Task-specific kwargs to initialize the pipeline.
        """
        raise NotImplementedError()
