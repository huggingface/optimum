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
import collections
import contextlib
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_reinit, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES

# from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo
from transformers.onnx import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import EvalPrediction, PredictionOutput, Trainer, TrainOutput, set_seed
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging

# For ORTTrainer
import onnx
import onnxruntime
# from onnxruntime.training import checkpoint


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
        self.ort_model = None
        if self.args.local_rank:
            torch.cuda.set_device(self.args.local_rank)

    def update_torch_model(self, checkpoint=None):
        """
        What object is `checkpoint` can't find its definition in ort #TODO Check its origin and `torch_ort.ORTModule`
        """
        if self.ort_model:
            logger.info("Updating weights of torch model from ORT model.")
            ort_state_dict = checkpoint.experimental_state_dict(self.ort_model)
            self.model.load_state_dict(ort_state_dict, strict=False)
        else:
            logger.warning("No ORT model found to update weights from, assuming torch model is up to date.")

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

        if self.ort_model:
            # `train()` function has been called, `self.ort_model` is an instance of `orttrainer.ORTTrainer`.
            self.ort_model.save_as_onnx(onnx_model_path)
            # delete the training model to free up GPU memory
            del self.ort_model
            self.ort_model = None
        else:
            # Convert the `PreTrainedModel` to an onnx model and export the onnx graph
            self._export(onnx_model_path)

        self.infer_sess = onnxruntime.InferenceSession(onnx_model_path.as_posix())
        # if self.args.no_cuda:
        #     self.infer_sess = onnxruntime.InferenceSession(onnx_model_path,
        #                                             # sess_options,
        #                                             providers=['CPUExecutionProvider'],
        #                                             **kwargs)
        # else:
        #     self.infer_sess = onnxruntime.InferenceSession(onnx_model_path.as_posix(), **kwargs)  # sess_options
        #     assert 'CUDAExecutionProvider' in self.infer_sess.get_providers()  # Make sure there is GPU

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
                    inputs[input_name] = torch.nn.functional.pad(inputs[input_name], (0, 0, 0, pad_len))

            input_feed = dict(map(lambda input_name: (input_name, inputs[input_name].numpy()), input_names))
            step_eval_loss = self.infer_sess.run(output_names, input_feed)
            eval_losses += [step_eval_loss[0]]

        # TODO: Other evaluation metrics: accuracy, f1 score, precision, recall...
        metrics = {}
        # loss
        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)

        return metrics

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Overrides the `Trainer.evaluate()` by updating the torch model weights as well as delete the 
        ort training model to free up memory.
        """
        # update the torch model weights and delete the ort training model to free up GPU memory
        self.update_torch_model()
        del self.ort_model
        self.ort_model = None

        eval_dataset = eval_dataset if eval_dataset else self.eval_dataset
        output_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        return output_metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        """
        Overrides the `Trainer.predict()` by updating the torch model weights as well as delete the 
        ort training model to free up memory.
        """
        # update the torch model weights and delete the ort training model to free up GPU memory
        self.update_torch_model()
        del self.ort_model
        self.ort_model = None

        output_metrics = super().predict(test_dataset, ignore_keys, metric_key_prefix)

        return output_metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Update the torch model weights
        self.update_torch_model()

        super()._save(output_dir, state_dict)

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
        _ = export(self.tokenizer, self.model, onnx_config, opset, model_path)
