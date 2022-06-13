from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union

from . import is_pydantic_available
from .doc import generate_doc_dataclass


if is_pydantic_available():
    from pydantic.dataclasses import dataclass
else:
    from dataclasses import dataclass


class APIFeaturesManager:
    _SUPPORTED_TASKS = ["text-classification", "token-classification", "question-answering"]

    @staticmethod
    def check_supported_model_task_pair(model_type: str, task: str):
        model_type = model_type.lower()
        if model_type not in APIFeaturesManager._SUPPORTED_MODEL_TYPE:
            raise KeyError(
                f"{model_type} is not supported yet. "
                f"Only {list(APIFeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        elif task not in APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]:
            raise KeyError(
                f"{task} is not supported yet for model {model_type}. "
                f"Only {APIFeaturesManager._SUPPORTED_MODEL_TYPE[model_type]} are supported. "
                f"If you want to support {task} please propose a PR or open up an issue."
            )

    @staticmethod
    def check_supported_task(task: str):
        if task not in APIFeaturesManager._SUPPORTED_TASKS:
            raise KeyError(
                f"{task} is not supported yet. "
                f"Only {APIFeaturesManager._SUPPORTED_TASKS} are supported. "
                f"If you want to support {task} please propose a PR or open up an issue."
            )


class Frameworks(str, Enum):
    onnxruntime = "onnxruntime"


class CalibrationMethods(str, Enum):
    minmax = "minmax"
    percentile = "percentile"
    entropy = "entropy"


class QuantizationApproach(str, Enum):
    static = "static"
    dynamic = "dynamic"


@generate_doc_dataclass
@dataclass
class Calibration:
    """Parameters for post-training calibration with static quantization."""

    method: CalibrationMethods = field(
        metadata={"description": 'Calibration method used, either "minmax", "entropy" or "percentile".'}
    )
    num_calibration_samples: int = field(
        metadata={
            "description": "Number of examples to use for the calibration step resulting from static quantization."
        }
    )
    calibration_histogram_percentile: Optional[float] = field(
        default=None, metadata={"description": "The percentile used for the percentile calibration method."}
    )
    calibration_moving_average: Optional[bool] = field(
        default=None,
        metadata={
            "description": "Whether to compute the moving average of the minimum and maximum values for the minmax calibration method."
        },
    )
    calibration_moving_average_constant: Optional[float] = field(
        default=None,
        metadata={
            "description": "Constant smoothing factor to use when computing the moving average of the minimum and maximum values. Effective only when the selected calibration method is minmax and `calibration_moving_average` is set to True."
        },
    )


@generate_doc_dataclass
@dataclass
class FrameworkArgs:
    opset: Optional[int] = field(default=15, metadata={"description": "ONNX opset version to export the model with."})
    optimization_level: Optional[int] = field(default=0, metadata={"description": "ONNX optimization level."})

    def __post_init__(self):
        # validate `opset`
        assert self.opset <= 15, f"Unsupported OnnxRuntime opset: {self.opset}"

        # validate `optimization_level`
        assert self.optimization_level in [
            0,
            1,
            2,
            99,
        ], f"Unsupported OnnxRuntime optimization level: {self.optimization_level}"


@generate_doc_dataclass
@dataclass
class Versions:
    transformers: str = field(metadata={"description": "Transformers version."})
    optimum: str = field(metadata={"description": "Optimum version."})
    optimum_hash: Optional[str] = field(
        default=None, metadata={"description": "Optimum commit hash, in case the dev version is used."}
    )
    onnxruntime: Optional[str] = field(default=None, metadata={"description": "Onnx Runtime version."})
    torch_ort: Optional[str] = field(default=None, metadata={"description": "Torch-ort version."})


@generate_doc_dataclass
@dataclass
class Evaluation:
    time: List[Dict] = field(metadata={"description": "Measures of inference time (latency, throughput)."})
    others: Dict = field(metadata={"description": "Metrics measuring the performance on the given task."})

    def __post_init__(self):
        # validate `others`
        assert "baseline" in self.others
        assert "optimized" in self.others
        for metric_name, metric_dict in self.others["baseline"].items():
            assert metric_dict.keys() == self.others["optimized"][metric_name].keys()


@generate_doc_dataclass
@dataclass
class DatasetArgs:
    """Parameters related to the dataset."""

    path: str = field(metadata={"description": "Path to the dataset, as in `datasets.load_dataset(path)`."})
    eval_split: str = field(metadata={"description": 'Dataset split used for evaluation (e.g. "test").'})
    data_keys: Dict[str, Union[None, str]] = field(
        metadata={
            "description": 'Dataset columns used as input data. At most two, indicated with "primary" and "secondary".'
        }
    )
    ref_keys: List[str] = field(metadata={"description": "Dataset column used for references during evaluation."})
    name: Optional[str] = field(
        default=None, metadata={"description": "Name of the dataset, as in `datasets.load_dataset(path, name)`."}
    )
    calibration_split: Optional[str] = field(
        default=None, metadata={"description": 'Dataset split used for calibration (e.g. "train").'}
    )


@generate_doc_dataclass
@dataclass
class TaskArgs:
    """Task-specific parameters."""

    is_regression: Optional[bool] = field(
        default=None,
        metadata={
            "description": "Text classification specific. Set whether the task is regression (output = one float)."
        },
    )


@dataclass
class _RunBase:
    model_name_or_path: str = field(
        metadata={"description": "Name of the model hosted on the Hub to use for the run."}
    )
    task: str = field(metadata={"description": "Task performed by the model."})
    quantization_approach: QuantizationApproach = field(
        metadata={"description": "Whether to use dynamic or static quantization."}
    )
    dataset: DatasetArgs = field(
        metadata={"description": "Dataset to use. Several keys must be set on top of the dataset name."}
    )
    framework: Frameworks = field(metadata={"description": 'Name of the framework used (e.g. "onnxruntime").'})
    framework_args: FrameworkArgs = field(metadata={"description": "Framework-specific arguments."})


@dataclass
class _RunDefaults:
    operators_to_quantize: Optional[List[str]] = field(
        default_factory=lambda: ["Add", "MatMul"],
        metadata={
            "description": 'Operators to quantize, doing no modifications to others (default: `["Add", "MatMul"]`).'
        },
    )
    node_exclusion: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"description": "Specific nodes to exclude from being quantized (default: `[]`)."},
    )
    per_channel: Optional[bool] = field(
        default=False, metadata={"description": "Whether to quantize per channel (default: `False`)."}
    )
    calibration: Optional[Calibration] = field(
        default=None, metadata={"description": "Calibration parameters, in case static quantization is used."}
    )
    task_args: Optional[TaskArgs] = field(
        default=None, metadata={"description": "Task-specific arguments (default: `None`)."}
    )
    aware_training: Optional[bool] = field(
        default=False,
        metadata={
            "description": "Whether the quantization is to be done with Quantization-Aware Training (not supported)."
        },
    )


@dataclass
class _RunConfigBase:
    """Parameters defining a run. A run is an evaluation of a triplet (model, dataset, metric) coupled with optimization parameters, allowing to compare a transformers baseline and a model optimized with Optimum."""

    metrics: List[str] = field(metadata={"description": "List of metrics to evaluate on."})


@dataclass
class _RunConfigDefaults(_RunDefaults):
    batch_sizes: Optional[List[int]] = field(
        default_factory=lambda: [4, 8],
        metadata={"description": "Batch sizes to include in the run to measure time metrics."},
    )
    input_lengths: Optional[List[int]] = field(
        default_factory=lambda: [128],
        metadata={"description": "Input lengths to include in the run to measure time metrics."},
    )


@dataclass
class Run(_RunDefaults, _RunBase):
    def __post_init__(self):
        # validate `task`
        APIFeaturesManager.check_supported_task(task=self.task)

        # validate `task_args`
        if self.task == "text-classification":
            message = "For text classification, whether the task is regression should be explicity specified in the task_args.is_regression key."
            assert self.task_args != None, message
            assert self.task_args["is_regression"] != None, message

        # validate `dataset`
        if self.quantization_approach == "static":
            assert self.dataset[
                "calibration_split"
            ], "Calibration split should be passed for static quantization in the dataset.calibration_split key."

        # validate `calibration`
        if self.quantization_approach == "static":
            assert (
                self.calibration
            ), "Calibration parameters should be passed for static quantization in the calibration key."

        # validate `aware_training`
        assert self.aware_training == False, "Quantization-Aware Training not supported."


@generate_doc_dataclass
@dataclass
class RunConfig(Run, _RunConfigDefaults, _RunConfigBase):
    """Class holding the parameters to launch a run."""

    pass
