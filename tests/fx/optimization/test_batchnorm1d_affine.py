# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import pytest
import torch

from optimum.fx.optimization import FuseBatchNorm1dInLinear


def _stats(batchnorm):
    with torch.no_grad():
        batchnorm.running_mean.copy_(torch.tensor([0.4, -1.2, 0.7], dtype=batchnorm.running_mean.dtype))
        batchnorm.running_var.copy_(torch.tensor([0.3, 1.5, 2.2], dtype=batchnorm.running_var.dtype))


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("order", ["linear_then_bn", "bn_then_linear"])
def test_non_affine_batchnorm1d_fusion_preserves_outputs(bias, order):
    torch.manual_seed(0)
    linear = torch.nn.Linear(3, 3, bias=bias)
    batchnorm = torch.nn.BatchNorm1d(3, affine=False)
    _stats(batchnorm)
    layers = (linear, batchnorm) if order == "linear_then_bn" else (batchnorm, linear)
    model = torch.nn.Sequential(*layers).eval()
    inputs = torch.tensor([[1.0, -0.5, 2.0], [0.2, 0.3, -1.1], [-2.0, 1.5, 0.4]])
    expected = model(inputs)

    fused = FuseBatchNorm1dInLinear()(torch.fx.symbolic_trace(model))

    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in fused.modules())
    torch.testing.assert_close(fused(inputs), expected)


def test_non_affine_batchnorm1d_before_rectangular_linear():
    torch.manual_seed(1)
    batchnorm = torch.nn.BatchNorm1d(3, affine=False)
    _stats(batchnorm)
    linear = torch.nn.Linear(3, 2, bias=True)
    model = torch.nn.Sequential(batchnorm, linear).eval()
    inputs = torch.tensor([[0.5, -1.0, 1.5], [2.0, 0.0, -0.5]])
    expected = model(inputs)

    fused = FuseBatchNorm1dInLinear()(torch.fx.symbolic_trace(model))

    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in fused.modules())
    torch.testing.assert_close(fused(inputs), expected)
