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
from typing import Callable, Optional, Union, List

import torch
from transformers import PreTrainedModel

from neural_compressor.experimental import Component, Distillation, Pruning, Quantization, common
from neural_compressor.experimental.scheduler import Scheduler


logger = logging.getLogger(__name__)


class IncOptimizer:
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        components: Optional[List[Component]] = [],
        one_shot_optimization: Optional[bool] = False,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                Model to quantize and/or prune.
            components (List[:obj:`Component`], `optional`):
                List of Component objects which contains Quantization, 
                Pruning, Distillation objects.
            one_shot_optimization (bool, `optional`):
                Whether to do multiple compression processes together.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function which will be combined with pruning.
        """

        if len(components) == 0:
            raise RuntimeError("`IncOptimizer` requires at least one `Quantization`, "
                               "`Pruning` or `Distillation` object")

        self.scheduler = Scheduler()
        self.scheduler.model = common.Model(model)
        
        if one_shot_optimization and len(components) > 1:
            agent = self.scheduler.combine(*components)
            agent.train_func = train_func
            agent.eval_func = eval_func
            for component in components:
                if isinstance(component, Distillation) and hasattr(component, 'criterion'):
                    agent.criterion = component.criterion
            print(agent)
            self.scheduler.append(agent)
        else:
            self.scheduler.append(components)

    def fit(self):
        opt_model = self.scheduler()
        return opt_model
