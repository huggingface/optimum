import torch
from optimum.exporters.onnx.model_patcher import onnx_compatible_aten__ior_ 

def test_onnx_compatible_aten__ior_():
    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    y = torch.tensor([3, 0, 1], dtype=torch.int32)

    expected = x | y  
    result = onnx_compatible_aten__ior_(x.clone(), y)

    assert torch.equal(result, expected), "onnx_compatible_aten__ior_ did not produce expected result"
