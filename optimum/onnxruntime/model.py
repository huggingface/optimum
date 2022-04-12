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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset
from transformers import EvalPrediction
from transformers.onnx import OnnxConfig
from transformers.trainer_pt_utils import nested_concat
from transformers.trainer_utils import EvalLoopOutput

import onnx
from onnxruntime import InferenceSession, SessionOptions


logger = logging.getLogger(__name__)


# TODO : Temporary class, added to perform ONNX models evaluation, will be replaced with ONNXModel class
class ORTModel:
    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        onnx_config: OnnxConfig,
        ort_provider: Optional[str] = "CPUExecutionProvider",
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        label_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path (`Union[str, os.PathLike]`):
                The path to the model ONNX Intermediate Representation (IR).
            onnx_config (`OnnxConfig`):
                An ONNX configuration associated to the ONNX model describing metadata on how to export the model
                through the ONNX format.
            ort_provider (:obj:`str`, `optional`):
                ONNX Runtime execution provider to use.
            compute_metrics (`Callable[[EvalPrediction], Dict]`, `optional`):
                The function that will be used to compute metrics at evaluation. Must take an `EvalPrediction` and
                return a dictionary string to metric values.
            label_names (`List[str]`, `optional`):
                The list of keys in your dictionary of inputs that correspond to the labels.
        """
        if not isinstance(onnx_config, OnnxConfig):
            raise TypeError(
                f"The ONNX configuration `onnx_config` associated to the pre-existing ONNX model is of type "
                f"{type(onnx_config)}, which is not an instance of `OnnxConfig`."
            )

        self.onnx_named_inputs = list(onnx_config.inputs.keys())
        self.onnx_named_outputs = list(onnx_config.outputs.keys())
        self.onnx_config = onnx_config
        self.ort_provider = ort_provider
        self.model_path = Path(model_path)
        self.compute_metrics = compute_metrics
        default_label_names = (
            ["start_positions", "end_positions"] if self.onnx_config.task == "question-answering" else ["labels"]
        )
        self.label_names = default_label_names if label_names is None else label_names

    def evaluation_loop(self, dataset: Dataset):
        """
        Run evaluation and returns metrics and predictions.

        Args:
            dataset (`datasets.Dataset`):
                Dataset to use for the evaluation step.
        """
        logger.info(f"***** Running evaluation *****")
        all_preds = None
        all_labels = None
        options = SessionOptions()
        session = InferenceSession(self.model_path.as_posix(), options, providers=[self.ort_provider])
        for step, inputs in enumerate(dataset):
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            if has_labels:
                labels = tuple(np.array([inputs.get(name)]) for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None
            onnx_inputs = {key: np.array([inputs[key]]) for key in self.onnx_config.inputs if key in inputs}
            preds = session.run(self.onnx_named_outputs, onnx_inputs)
            if len(preds) == 1:
                preds = preds[0]
            all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(dataset))
