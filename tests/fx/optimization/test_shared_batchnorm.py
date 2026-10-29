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
from torch import nn
from torch.fx import symbolic_trace

from optimum.fx.optimization import FuseBatchNorm1dInLinear, FuseBatchNorm2dInConv2d


class BatchNormPair(nn.Module):
    def __init__(self, kind, reuse):
        super().__init__()
        self.kind = kind
        self.reuse = reuse
        # Avoid the independent Conv2d bias arithmetic issue.
        self.layer = nn.Conv2d(2, 2, 1, bias=False) if kind == "conv" else nn.Linear(2, 2)
        self.norm = nn.BatchNorm2d(2) if kind == "conv" else nn.BatchNorm1d(2)
        with torch.no_grad():
            self.norm.running_mean.copy_(torch.tensor([0.4, -0.7]))
            self.norm.running_var.copy_(torch.tensor([0.3, 1.7]))
            self.norm.weight.copy_(torch.tensor([1.2, 0.6]))
            self.norm.bias.copy_(torch.tensor([-0.3, 0.5]))

    def forward(self, x):
        intermediate = self.norm(x) if self.kind == "before" else self.layer(x)
        result = self.layer(intermediate) if self.kind == "before" else self.norm(intermediate)
        if self.reuse == "layer":
            return result, self.layer(x + 1)
        if self.reuse == "norm":
            return result, self.norm(x + 1)
        if self.reuse == "both":
            second = self.layer(self.norm(x + 1)) if self.kind == "before" else self.norm(self.layer(x + 1))
            return result, second
        if self.reuse == "multiuser":
            return result, intermediate
        return result


def make_pair(kind, reuse, dtype):
    torch.manual_seed(0)
    model = BatchNormPair(kind, reuse).eval().to(dtype)
    x = torch.randn(3, 2, 4, 4, dtype=dtype) if kind == "conv" else torch.randn(3, 2, dtype=dtype)
    transformation = FuseBatchNorm2dInConv2d() if kind == "conv" else FuseBatchNorm1dInLinear()
    return model, x, transformation


@pytest.mark.parametrize("kind", ["conv", "after", "before"])
@pytest.mark.parametrize("reuse", ["layer", "norm", "both"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alias", [False, True])
def test_batchnorm_fusion_preserves_reused_modules(kind, reuse, dtype, alias):
    model, x, transformation = make_pair(kind, reuse, dtype)
    expected = copy.deepcopy(model)(x)
    traced = symbolic_trace(model)
    if alias:
        # A valid FX graph can refer to the same module through different paths.
        for target in ("layer", "norm"):
            nodes = [node for node in traced.graph.nodes if node.op == "call_module" and node.target == target]
            if len(nodes) > 1:
                traced.add_submodule(target + "_alias", traced.get_submodule(target))
                nodes[-1].target = target + "_alias"
        traced.graph.lint()
        traced.recompile()
    original_code = traced.code
    original_state = copy.deepcopy(traced.state_dict())

    for _ in range(2):
        transformed = transformation(traced)
        torch.testing.assert_close(transformed(x), expected)
        assert transformed.code == original_code
        assert transformed.state_dict().keys() == original_state.keys()
        for key, value in original_state.items():
            torch.testing.assert_close(transformed.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["conv", "after", "before"])
@pytest.mark.parametrize("reuse", ["none", "multiuser"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batchnorm_fusion_single_call_and_multiple_users(kind, reuse, dtype):
    model, x, transformation = make_pair(kind, reuse, dtype)
    expected = copy.deepcopy(model)(x)
    transformed = transformation(symbolic_trace(model))
    torch.testing.assert_close(transformed(x), expected)
    batchnorms = [module for module in transformed.modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))]
    assert len(batchnorms) == (1 if reuse == "multiuser" else 0)
    torch.testing.assert_close(transformation(transformed)(x), expected)


class MixedBatchNormPairs(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.shared = BatchNormPair(kind, "both")
        self.single = BatchNormPair(kind, "none")

    def forward(self, x):
        return self.shared(x), self.single(x)


@pytest.mark.parametrize("kind", ["conv", "after", "before"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batchnorm_fusion_keeps_optimizing_unshared_pairs(kind, dtype):
    _, x, transformation = make_pair(kind, "none", dtype)
    model = MixedBatchNormPairs(kind).eval().to(dtype)
    expected = copy.deepcopy(model)(x)
    transformed = transformation(symbolic_trace(model))
    torch.testing.assert_close(transformed(x), expected)
    assert hasattr(transformed.shared, "norm")
    assert not hasattr(transformed.single, "norm")
