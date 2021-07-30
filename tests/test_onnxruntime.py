import unittest
from pathlib import Path
import tempfile
from transformers.onnx import validate_model_outputs
from optimus.onnxruntime import convert_to_onnx, optimize, quantize


class TestOptimize(unittest.TestCase):

    def test_optimize(self):
        tmp_dir = tempfile.TemporaryDirectory()
        onnx_model = Path(tmp_dir.name).joinpath("model.onnx")
        model_name = "bert-base-uncased"
        opt_level = 1

        tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(
            model_name,
            onnx_model,
            features="default",
            opset=12
        )

        model_type = model.config.model_type
        validate_model_outputs(onnx_config, tokenizer, model, onnx_model, onnx_outputs, atol=1e-4)

        optimized_model = optimize(onnx_model, model_type, opt_level=opt_level, only_onnxruntime=True)
        validate_model_outputs(onnx_config, tokenizer, model, optimized_model, onnx_outputs, atol=1e-4)

        quantized_model = quantize(optimized_model)
        validate_model_outputs(onnx_config, tokenizer, model, quantized_model, onnx_outputs, atol=1.5)

        tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

