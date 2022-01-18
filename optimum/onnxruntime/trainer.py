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

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.dependency_versions_check import dep_version_check
from transformers.modeling_utils import PreTrainedModel, unwrap_model

# from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo
from transformers.onnx import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import EvalPrediction, PredictionOutput, Trainer, TrainOutput, set_seed
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging

# For ORTTrainer
import onnx
import onnxruntime
from torch_ort import ORTModule, DebugOptions, LogLevel, set_seed

logger = logging.get_logger(__name__)

class ORTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        feature: str = "default",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        onnxruntime.set_seed(self.args.seed)
        self.ort_model_path = None
        self.session_options = None
        if self.args.local_rank:
            torch.cuda.set_device(self.args.local_rank)
        
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        # Wrap the model with `torch_ort.ORTModule`
        debugOptions = DebugOptions(
            save_onnx=True, 
            onnx_prefix=self.model.config.name_or_path.split("/")[-1]
        )
        debugOptions.save_onnx_models._path = self.args.output_dir
        self.model_wrapped = ORTModule(self.model_wrapped, debugOptions)
        self.model = ORTModule(self.model, debugOptions)
        
        # Training
        train_output = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        
        train_manager = self.model._torch_module._execution_manager._training_manager
        # train_manager._export_model()
        self.session_options, providers, provider_options = train_manager._get_session_config()
        self.ort_model_path = self.session_options.optimized_model_filepath
        self.ort_model_path = "./results/bert-base-cased_torch_exported_training.onnx"

        return train_output

    def evaluate_ort(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **kwargs
    ) -> Dict[str, float]:

        self.infer_sess = None
        onnx_model_path = Path(
            os.path.join(self.args.output_dir, self.model.config.name_or_path.split("/")[-1] + ".onnx")
        )

        if self.ort_model_path:
            # `train()` function has been called, the onnx graph is exported
            pass
        else:
            # Convert the `PreTrainedModel` to an onnx model and export the onnx graph
            
            self._export(onnx_model_path)
            self.ort_model_path = onnx_model_path.as_posix()

        # Can't infer the exported onnx models due to impatible opset
        self.infer_sess = onnxruntime.InferenceSession(
            self.ort_model_path, 
            self.session_options,
            providers=['CPUExecutionProvider'] #['CUDAExecutionProvider']
        )

        output_names = [output.name for output in self.infer_sess._outputs_meta]
        input_names = [ort_input.name for ort_input in self.infer_sess._inputs_meta]

        # load the eval dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        description = "Evaluation"

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(eval_dataloader.dataset))
        logger.info("  Batch size = %d", eval_dataloader.batch_size)
        eval_losses: List[float] = []

        for inputs in tqdm(eval_dataloader, desc=description):

            # for the last batch, pad to the batch size.
            if len(inputs["input_ids"]) < self.args.per_device_eval_batch_size:
                pad_len = self.args.per_device_eval_batch_size - inputs["input_ids"].size()[0]
                # pad input feeds
                for input_name in input_names:
                    print("pad len", pad_len)
                    print("input dim", inputs["input_ids"].dim())
                    inputs[input_name] = torch.nn.functional.pad(inputs[input_name], (0, 0, 0, pad_len))

            input_feed = dict(map(lambda input_name: (input_name, inputs[input_name].numpy()), input_names))
            step_eval_loss = self.infer_sess.run(output_names, input_feed)
            eval_losses += [step_eval_loss[0]]
            break

        # TODO: Other evaluation metrics: accuracy, f1 score, precision, recall...
        metrics = {}
        # loss
        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)

        return metrics
    
    def _save_model(self, model: onnx.ModelProto, file_path: str):
        onnx.save(model, file_path)
    
    def save_model(self, model):
        # save the ortmodule exported model
        path = Path(
            os.path.join(self.args.output_dir, self.model.config.name_or_path.split("/")[-1] + ".onnx")
        )
        self._save_model(
            model,
            path,
        )
    
    def _export(self, model_path: os.PathLike, feature: str = "default", opset: Optional[int] = None) -> None:
        """
        Load and export a model to an ONNX Intermediate Representation (IR).

        Param:
            model_path (:obj:`os.PathLike`):
                The path used to save the model exported to an ONNX Intermediate Representation (IR).
        """
        model_type, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self.model, feature=feature)
        onnx_config = model_onnx_config(self.model.config)
        opset = onnx_config.default_onnx_opset if opset is None else opset
        self.model.to('cpu')
        _ = export(self.tokenizer, self.model, onnx_config, opset, model_path)
