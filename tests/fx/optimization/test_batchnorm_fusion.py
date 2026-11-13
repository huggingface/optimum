import pytest
import torch

from optimum.fx.optimization import FuseBatchNorm2dInConv2d


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_batchnorm_fusion_preserves_outputs(affine, bias, dtype):
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(2, 3, kernel_size=1, bias=bias, dtype=dtype)
    batchnorm = torch.nn.BatchNorm2d(3, affine=affine, dtype=dtype)
    with torch.no_grad():
        batchnorm.running_mean.copy_(torch.tensor([-1.0, 0.5, 1.0]))
        batchnorm.running_var.copy_(torch.tensor([0.5, 1.0, 2.0]))
        if affine:
            batchnorm.weight.copy_(torch.tensor([0.3, 0.8, 1.5]))
            batchnorm.bias.copy_(torch.tensor([-0.7, 0.2, 0.4]))
    model = torch.nn.Sequential(conv, batchnorm).eval()
    inputs = torch.randn(4, 2, 3, 3, dtype=dtype)
    expected = model(inputs)

    fused = FuseBatchNorm2dInConv2d()(torch.fx.symbolic_trace(model))

    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in fused.modules())
    torch.testing.assert_close(fused(inputs), expected)
