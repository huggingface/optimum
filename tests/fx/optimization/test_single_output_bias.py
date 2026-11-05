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


class Projection(torch.nn.Module):
    def __init__(self, out_features, bias=True):
        super().__init__()
        self.projection = torch.nn.Linear(4, out_features, bias=bias)

    def forward(self, inputs):
        return self.projection(inputs)


@pytest.mark.parametrize("out_features", [1, 3])
@pytest.mark.parametrize("input_shape", [(4,), (2, 4), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fuse_bias_round_trip_preserves_parameter_shapes(out_features, input_shape, dtype):
    torch.manual_seed(0)
    model = Projection(out_features).to(dtype=dtype)
    original_state = copy.deepcopy(model.state_dict())
    inputs = torch.randn(input_shape, dtype=dtype, requires_grad=True)
    expected = model(inputs)
    expected.square().mean().backward()
    traced = torch.fx.symbolic_trace(copy.deepcopy(model))
    transformation = FuseBiasInLinear()

    for _ in range(2):
        fused = transformation(traced)
        torch.testing.assert_close(fused(inputs), expected)
        traced = transformation(fused, reverse=True)
        assert traced.projection.bias.shape == (out_features,)
        for name, value in traced.state_dict().items():
            assert value.shape == original_state[name].shape
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        Projection(out_features).to(dtype=dtype).load_state_dict(traced.state_dict(), strict=True)
        torch.testing.assert_close(traced(inputs), expected)

    traced.zero_grad()
    actual_inputs = inputs.detach().clone().requires_grad_()
    traced(actual_inputs).square().mean().backward()
    torch.testing.assert_close(actual_inputs.grad, inputs.grad)
    for name, parameter in traced.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(model.named_parameters())[name].grad)


@pytest.mark.parametrize("out_features", [1, 3])
def test_fuse_bias_round_trip_keeps_biasless_projection(out_features):
    model = Projection(out_features, bias=False)
    inputs = torch.randn(2, 4)
    transformation = FuseBiasInLinear()
    traced = torch.fx.symbolic_trace(model)
    restored = transformation(transformation(traced), reverse=True)
    assert restored.projection.bias is None
    torch.testing.assert_close(restored(inputs), model(inputs))
