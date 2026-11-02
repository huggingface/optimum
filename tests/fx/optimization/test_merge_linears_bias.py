# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import copy

import pytest
import torch

from optimum.fx.optimization import MergeLinears


class Projections(torch.nn.Module):
    def __init__(self, biases):
        super().__init__()
        self.query = torch.nn.Linear(4, 3, bias=biases[0])
        self.key = torch.nn.Linear(4, 2, bias=biases[1])

    def forward(self, inputs):
        return self.query(inputs), self.key(inputs)


class Model(torch.nn.Module):
    def __init__(self, biases):
        super().__init__()
        self.attention = Projections(biases)

    def forward(self, inputs):
        return self.attention(inputs)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("input_shape", [(4,), (2, 4), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_merge_linears_preserves_biasless_parameters(bias, input_shape, dtype):
    torch.manual_seed(0)
    model = Model((bias, bias)).to(dtype=dtype)
    original_state = copy.deepcopy(model.state_dict())
    inputs = torch.randn(input_shape, dtype=dtype, requires_grad=True)
    expected = model(inputs)
    sum(output.square().mean() for output in expected).backward()
    traced = torch.fx.symbolic_trace(copy.deepcopy(model))
    transformation = MergeLinears()

    for _ in range(2):
        fused = transformation(traced)
        linears = [module for module in fused.modules() if isinstance(module, torch.nn.Linear)]
        assert len(linears) == 1
        assert (linears[0].bias is not None) == bias
        torch.testing.assert_close(fused(inputs), expected)
        traced = transformation(fused, reverse=True)
        assert set(traced.state_dict()) == set(original_state)
        for name, value in traced.state_dict().items():
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        Model((bias, bias)).to(dtype=dtype).load_state_dict(traced.state_dict(), strict=True)
        torch.testing.assert_close(traced(inputs), expected)

    traced.zero_grad()
    actual_inputs = inputs.detach().clone().requires_grad_()
    sum(output.square().mean() for output in traced(actual_inputs)).backward()
    torch.testing.assert_close(actual_inputs.grad, inputs.grad)
    for name, parameter in traced.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(model.named_parameters())[name].grad)


@pytest.mark.parametrize("biases", [(False, True), (True, False)])
def test_merge_linears_warns_when_inserting_a_missing_bias(biases):
    model = Model(biases)
    inputs = torch.randn(2, 4)
    expected = model(inputs)
    with pytest.warns(UserWarning, match="Not all the linear layers"):
        transformed = MergeLinears()(torch.fx.symbolic_trace(model))
    torch.testing.assert_close(transformed(inputs), expected)
