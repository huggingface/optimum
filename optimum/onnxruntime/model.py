#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset
from transformers import EvalPrediction
from transformers.trainer_pt_utils import nested_concat
from transformers.trainer_utils import EvalLoopOutput

from onnxruntime import InferenceSession


logger = logging.getLogger(__name__)


# TODO : Temporary class, added to perform ONNX models evaluation, will be replaced with ONNXModel class
class ORTModel:
    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        execution_provider: Optional[str] = "CPUExecutionProvider",
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        label_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path (`Union[str, os.PathLike]`):
                The path to the model ONNX Intermediate Representation (IR).
            execution_provider (:obj:`str`, `optional`):
                ONNX Runtime execution provider to use.
            compute_metrics (`Callable[[EvalPrediction], Dict]`, `optional`):
                The function that will be used to compute metrics at evaluation. Must take an `EvalPrediction` and
                return a dictionary string to metric values.
            label_names (`List[str]`, `optional`):
                The list of keys in your dictionary of inputs that correspond to the labels.
        """
        self.compute_metrics = compute_metrics
        self.label_names = ["labels"] if label_names is None else label_names
        self.session = InferenceSession(str(model_path), providers=[execution_provider])
        self.onnx_input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}

    def evaluation_loop(self, dataset: Dataset):
        """
        Run evaluation and returns metrics and predictions.

        Args:
            dataset (`datasets.Dataset`):
                Dataset to use for the evaluation step.
        """
        logger.info("***** Running evaluation *****")
        all_preds = None
        all_labels = None
        for step, inputs in enumerate(dataset):
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            if has_labels:
                labels = tuple(np.array([inputs.get(name)]) for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None
            onnx_inputs = {key: np.array([inputs[key]]) for key in self.onnx_input_names if key in inputs}
            preds = self.session.run(None, onnx_inputs)
            if len(preds) == 1:
                preds = preds[0]
            all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(dataset))
