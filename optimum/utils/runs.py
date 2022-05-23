from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Extra, Field, validator

from .doc import add_end_docstrings

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


class BaseModelNoExtra(BaseModel):
    class Config:
        extra = Extra.forbid  # ban additional arguments


class Calibration(BaseModelNoExtra):
    """Parameters for post-training calibration with static quantization.

    Attributes:
        method (`CalibrationMethods`): Calibration method used, either "minmax", "entropy" or "percentile".
        num_calibration_samples (`int`): Number of examples to use for the calibration step resulting from static quantization.
        calibration_histogram_percentile (`float`, *optional*): The percentile used for the percentile calibration method.
        calibration_moving_average (`bool`, *optional*): Whether to compute the moving average of the minimum and maximum values for the minmax calibration method.
        calibration_moving_average_constant (`float`, *optional*): Constant smoothing factor to use when computing the moving average of the minimum and maximum values. Effective only when the selected calibration method is minmax and `calibration_moving_average` is set to True.
    """

    method: CalibrationMethods
    num_calibration_samples: int
    calibration_histogram_percentile: Optional[float] = None
    calibration_moving_average: Optional[bool] = None
    calibration_moving_average_constant: Optional[float] = None


class FrameworkArgs(BaseModelNoExtra):
    opset: Optional[int] = Field(15, description="ONNX opset version to export the model with.")
    optimization_level: Optional[int] = 0

    @validator("opset")
    def opset_check(cls, field_value):
        assert field_value <= 15, f"Unsupported OnnxRuntime opset: {field_value}"
        return field_value

    @validator("optimization_level")
    def model_type_check(cls, field_value, values):
        assert field_value in [0, 1, 2, 99], f"Unsupported OnnxRuntime optimization level: {field_value}"
        return field_value


class Versions(BaseModelNoExtra):
    transformers: str = Field(..., description="Transformers version.")
    optimum: str = Field(..., description="Optimum version.")
    optimum_hash: Optional[str]
    onnxruntime: Optional[str]
    torch_ort: Optional[str]


class Evaluation(BaseModelNoExtra):
    time: List[Dict]
    others: Dict

    @validator("others")
    def others_check(cls, field_value):
        assert "baseline" in field_value
        assert "optimized" in field_value
        for metric_name, metric_dict in field_value["baseline"].items():
            assert metric_dict.keys() == field_value["optimized"][metric_name].keys()
        return field_value


class DatasetArgs(BaseModelNoExtra):
    """Parameters related to the dataset.

    Attributes:
        path (`str`): Path to the dataset, as in `datasets.load_dataset(path)`.
        name (`str`, *optional*): Name of the dataset, as in `datasets.load_dataset(path, name)`.
        calibration_split (`str`, *optional*): Dataset split used for calibration (e.g. "train").
        eval_split (`str`): Dataset split used for evaluation (e.g. "test").
        data_keys (`Dict[str, Union[None, str]]`): Dataset columns used as input data. At most two, indicated with "primary" and "secondary".
        ref_keys (`List[str]`): Dataset column used for references during evaluation.
    """

    path: str
    name: Optional[str] = None
    calibration_split: Optional[str] = None
    eval_split: str
    # TODO auto-infer with train-eval-index if available
    data_keys: Dict[str, Union[None, str]]
    ref_keys: List[str]


class TaskArgs(BaseModelNoExtra):
    """Task-specific parameters.

    Attributes:
        is_regression (`bool`, *optional*): Text classification specific. Set whether the task is regression (output = one float).
    """

    is_regression: Optional[bool] = None


RUN_ATTRIBUTES = """Attributes:
    model_name_or_path (`str`): Name of the model hosted on the Hub to use for the run.
    task (`str`): Task performed by the model.
    task_args (`TaskArgs`, *optional*): Task-specific arguments (default: `None`).
    quantization_approach (`QuantizationApproach`): Whether to use dynamic or static quantization.
    dataset (`DatasetArgs`): Dataset to use. Several keys must be set on top of the dataset name.
    operators_to_quantize (`List[str]`, *optional*): Operators to quantize, doing no modifications to others (default: `["Add", "MatMul"]`).
    node_exclusion (`List[str]`, *optional*): Specific nodes to exclude from being quantized (default: `[]`).
    per_channel (`bool`, *optional*): Whether to quantize per channel (default: `False`).
    calibration (`Calibration`, *optional*): Calibration parameters, in case static quantization is used.
    framework (`Frameworks`): Name of the framework used (e.g. "onnxruntime").
    framework_args (`FrameworkArgs`): Framework-specific arguments.
    aware_training (`bool`, *optional*): Whether the quantization is to be done with Quantization-Aware Training (not supported).
"""


@add_end_docstrings(RUN_ATTRIBUTES)
class Run(BaseModelNoExtra):
    f"""Parameters defining a run. A run is an evaluation of a triplet (model, dataset, metric) coupled with optimization parameters, allowing to compare a transformers baseline and a model optimized with Optimum."""
    model_name_or_path: str
    task: str
    task_args: Optional[TaskArgs] = None
    quantization_approach: QuantizationApproach
    dataset: DatasetArgs
    operators_to_quantize: Optional[List[str]] = ["Add", "MatMul"]
    node_exclusion: Optional[List[str]] = []
    per_channel: Optional[bool] = False
    calibration: Optional[Calibration] = None
    framework: Frameworks
    framework_args: FrameworkArgs
    aware_training: Optional[bool] = False

    @validator("task")
    def model_type_check(cls, field_value):
        APIFeaturesManager.check_supported_task(task=field_value)
        return field_value

    @validator("task_args")
    def task_args_check(cls, field_value, values):
        if values["task"] == "text-classification":
            message = "For text classification, whether the task is regression should be explicity specified in the task_args.is_regression key."
            assert field_value != None, message
            assert field_value.is_regression != None, message
        return field_value

    @validator("dataset")
    def dataset_check(cls, field_value, values):
        if values["quantization_approach"] == "static":
            assert (
                field_value.calibration_split
            ), "Calibration split should be passed for static quantization in the dataset.calibration_split key."
        return field_value

    @validator("calibration")
    def calibration_check(cls, field_value, values):
        if values["quantization_approach"] == "static":
            assert (
                field_value
            ), "Calibration parameters should be passed for static quantization in the calibration key."
        return field_value

    @validator("aware_training")
    def aware_training_check(cls, field_value):
        assert field_value == False, "Quantization-Aware Training not supported."
        return field_value


RUN_CONFIG_ATTRIBUTES = """Attributes:
    metrics (`List[str]`): List of metrics to evaluate on.
    batch_sizes (`List[int]`, *optional*): Batch sizes to include in the run to measure time metrics (default: `[4, 8]`).
    input_lengths (`List[int]`, *optional*): Input lengths to include in the run to measure time metrics (default: `[128]`).
"""


@add_end_docstrings(RUN_CONFIG_ATTRIBUTES)
@add_end_docstrings(RUN_ATTRIBUTES)
class RunConfig(Run):
    """Parameters defining a run. A run is an evaluation of a triplet (model, dataset, metric) coupled with optimization parameters, allowing to compare a transformers baseline and a model optimized with Optimum."""

    metrics: List[str]  # TODO check that the passed metrics are fine for the given task/dataset
    batch_sizes: Optional[List[int]] = [4, 8]
    input_lengths: Optional[List[int]] = [128]
