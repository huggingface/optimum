# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments
from transformers.utils import add_start_docstrings

from .training_args import ORTTrainingArguments


@dataclass
class ORTSeq2SeqTrainingArguments(Seq2SeqTrainingArguments, ORTTrainingArguments):
    """
    Parameters:
        optim (`str` or [`training_args.ORTOptimizerNames`] or [`transformers.training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
            The optimizer to use, including optimizers in Transformers: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor. And optimizers implemented by ONNX Runtime: adamw_ort_fused.
    """

    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
