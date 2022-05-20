from enum import Enum
from typing import Dict, List, Optional, Union

from transformers import AutoConfig

from pydantic import BaseModel, Extra, Field, validator


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
    method: CalibrationMethods
    num_calibration_samples: int
    calibration_histogram_percentile: Optional[float] = None
    calibration_moving_average: Optional[bool] = None
    calibration_moving_average_constant: Optional[float] = None


class FrameworkArgs(BaseModelNoExtra):
    opset: Optional[int] = 15
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
    transformers: str
    optimum: str
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
    path: str
    name: Optional[str] = None
    calibration_split: Optional[str] = None
    eval_split: str
    data_keys: Dict[str, Union[None, str]]  # TODO auto-infer with train-eval-index if available
    ref_keys: List[str]  # TODO auto-infer with train-eval-index if available


class TaskArgs(BaseModelNoExtra):
    is_regression: Optional[bool] = None


class Run(BaseModelNoExtra):
    model_name_or_path: str
    task: str
    task_args: Optional[TaskArgs] = None
    quantization_approach: QuantizationApproach
    dataset: DatasetArgs
    operators_to_quantize: List[str] = ["Add", "MatMul"]
    node_exclusion: List[str] = []
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
        assert field_value == False, "Quantization aware-training not supported."
        return field_value


class RunConfig(Run):
    metrics: List[str]  # TODO check that the passed metrics are fine for the given task/dataset
    batch_sizes: Optional[List[int]] = Field(
        [4, 8], description="Possibly several batch size to include in a single run"
    )
    input_lengths: Optional[List[int]] = Field(
        [128], description="Possibly several input length to include in a single run"
    )
