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

from optimum.fx.optimization import FuseBiasInLinear


@pytest.mark.parametrize(
    "calls,sequential,bias",
    [(2, False, True), (3, False, True), (2, True, True), (1, False, True), (2, False, False)],
)
@pytest.mark.parametrize("input_shape", [(2, 4), (2, 3, 4)])
def test_fuse_bias_shared_linear(calls, sequential, bias, input_shape):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 4, bias=bias)

        def forward(self, inputs):
            if sequential:
                for _ in range(calls):
                    inputs = self.projection(inputs)
                return inputs
            return sum(self.projection(inputs + i) for i in range(calls))

    torch.manual_seed(0)
    model = Model().eval()
    inputs = torch.randn(input_shape, requires_grad=True)
    expected = model(inputs)
    expected.sum().backward()
    expected_input_grad = inputs.grad.clone()
    expected_weight_grad = model.projection.weight.grad.clone()
    if bias:
        expected_weight_grad = torch.cat([expected_weight_grad, model.projection.bias.grad[:, None]], dim=1)
    original_state = copy.deepcopy(model.state_dict())
    traced = torch.fx.symbolic_trace(copy.deepcopy(model))
    transformation = FuseBiasInLinear()

    for _ in range(2):
        transformed = transformation(traced)
        torch.testing.assert_close(transformed(inputs), expected)
        assert transformed.projection.bias is None
        assert transformed.projection.weight.shape == (4, 5 if bias else 4)
        assert sum(node.target is torch.cat for node in transformed.graph.nodes) == (calls if bias else 0)

        transformed.zero_grad()
        actual_inputs = inputs.detach().clone().requires_grad_()
        transformed(actual_inputs).sum().backward()
        torch.testing.assert_close(actual_inputs.grad, expected_input_grad)
        torch.testing.assert_close(transformed.projection.weight.grad, expected_weight_grad)

        # A second application must not append another bias column or input.
        transformed = transformation(transformed)
        torch.testing.assert_close(transformed(inputs), expected)

        traced = transformation(transformed, reverse=True)
        torch.testing.assert_close(traced(inputs), expected)
        assert not any(node.target is torch.cat for node in traced.graph.nodes)
        assert traced.state_dict().keys() == original_state.keys()
        for name, value in traced.state_dict().items():
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        model.load_state_dict(traced.state_dict(), strict=True)
