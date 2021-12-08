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
from typing import Optional, Union

import torch
from transformers import PreTrainedModel

from neural_compressor.experimental import Pruning, Quantization


logger = logging.getLogger(__name__)


class IncOptimizer:
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        quantizer: Optional[Quantization] = None,
        pruner: Optional[Pruning] = None,
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                Model to quantize and/or prune.
            quantizer (:obj:`Quantization`, `optional`):
                Quantization object which handles the quantization process.
            pruner (:obj:`Pruning`, `optional`):
                Pruning object which handles the pruning process.
        """
        from neural_compressor.experimental import common
        from neural_compressor.experimental.scheduler import Scheduler

        if quantizer is None and pruner is None:
            raise RuntimeError("`IncOptimizer` requires either a `quantizer` or `pruner` argument")

        self.scheduler = Scheduler()
        self.scheduler.model = common.Model(model)

        if isinstance(pruner, Pruning):
            self.scheduler.append(pruner)

        if isinstance(quantizer, Quantization):
            self.scheduler.append(quantizer)

    def fit(self):
        opt_model = self.scheduler()
        return opt_model
