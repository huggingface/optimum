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
"""Dataset processing factory."""

from typing import TYPE_CHECKING, Any, Type

from optimum.utils.preprocessing.image_classification import ImageClassificationProcessing
from optimum.utils.preprocessing.question_answering import QuestionAnsweringProcessing
from optimum.utils.preprocessing.text_classification import TextClassificationProcessing
from optimum.utils.preprocessing.token_classification import TokenClassificationProcessing


if TYPE_CHECKING:
    from .base import TaskProcessor


class TaskProcessorsManager:
    _TASK_TO_DATASET_PROCESSING_CLASS = {
        "text-classification": TextClassificationProcessing,
        "token-classification": TokenClassificationProcessing,
        "question-answering": QuestionAnsweringProcessing,
        "image-classification": ImageClassificationProcessing,
    }

    @classmethod
    def get_task_processor_class_for_task(cls, task: str) -> Type["TaskProcessor"]:
        if task not in cls._TASK_TO_DATASET_PROCESSING_CLASS:
            supported_tasks = ", ".join(cls._TASK_TO_DATASET_PROCESSING_CLASS.keys())
            raise KeyError(
                f"Could not find a `TaskProcessor` class for the task called {task}, supported tasks: "
                f"{supported_tasks}."
            )
        return cls._TASK_TO_DATASET_PROCESSING_CLASS[task]

    @classmethod
    def for_task(cls, task: str, *dataset_processing_args, **dataset_processing_kwargs: Any) -> "TaskProcessor":
        return cls.get_task_processor_class_for_task(task)(*dataset_processing_args, **dataset_processing_kwargs)
