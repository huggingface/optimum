#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import TYPE_CHECKING, Callable, List

import torch
from transformers import AutoConfig, PreTrainedModel
from transformers.onnx import FeaturesManager


if TYPE_CHECKING:
    from transformers import PretrainedConfig


def copy_signature(signature: inspect.Signature, function: Callable):
    """
    Replace the signature of `function` by `signature`.

    Args:
        signature (`inspect.Signature`): Signature to apply to `function`
        function (`Callable`): Function to override the signature of
    """

    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    wrapper.__signature__ = signature
    return wrapper


def return_tuple(function):
    """
    Decorator allowing to return a tuple instead of a dictionary in case the flag `self.return_dict` is `False`.
    """

    def wrapper(self, *args, **kwargs):
        result = function(self, *args, **kwargs)
        if not self.return_dict and isinstance(result, dict):
            result = tuple([value for _, value in result.items()])
        return result

    return wrapper


class FXModel(PreTrainedModel):
    """
    Base FX class for implementing models using torch.fx. The FXModel is useful to interact with the Hugging Face Hub,
    use transformers toolings as pipelines, as well as exporting fx models to ONNX using `transformers.onnx` toolchain.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    <Tip>
    By default, calling an `FXModel` will return a dictionary. This behavior can be disabled by setting `self.return_dict = False`.
    </Tip>
    """

    config_class = AutoConfig

    def __init__(self, model: "torch.fx.GraphModule", config: "PretrainedConfig", inputs: List, task: str):
        super().__init__(config)
        self.model = model
        self.config = config

        input_parameters = []
        for inp in inputs:
            input_parameters.append(inspect.Parameter(name=inp, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD))

        self.input_signature = inspect.Signature(input_parameters)
        self.return_dict = True

        # registers the FXModel classes into the transformers AutoModel classes
        # to avoid warnings when create a pipeline
        # see https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register("fx_model", AutoConfig)

        auto_class = FeaturesManager.get_model_class_for_feature(task)
        auto_class.register(AutoConfig, self.__class__)

    @return_tuple
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        if attr == "forward":
            # necessary for transformers.onnx.export(), where the model inputs set need to be a subset of the signature of `forward`
            value = copy_signature(signature=self.input_signature, function=value)
        return value


def wrap_fx_model(graph_module: "torch.fx.GraphModule", config: "PretrainedConfig", task: str) -> FXModel:
    """
    Wrap a `torch.fx.GraphModule` into a `transformers.PreTrainedModel`. This allows for an easy use of modified models with torch.fx with
    the transformers ecosystem.

    Arguments:
        graph_module (`torch.fx.GraphModule`):
            Module to wrap in a `PreTrainedModel`.
        config (`PretrainedConfig`):
            Configuration of the original model.
        task (str):
            Task performed by the model. See
            https://github.com/huggingface/transformers/blob/9b1dcba94a1f04f83b187aa6a443e68d8cecbdf5/src/transformers/pipelines/__init__.py#L152 for a comprehensive list.

    Returns:
        `FXModel`: A model inheriting from `PreTrainedModel`, whose `forward` method use the one from `graph_module`.
    """
    inputs = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.target)

    result = FXModel(model=graph_module, config=config, inputs=inputs, task=task)
    return result
