import unittest
from pathlib import Path
import tempfile
from transformers.onnx import validate_model_outputs
from optimus.onnxruntime import convert_to_onnx, optimize, quantize


class TestOptimize(unittest.TestCase):

    def test_optimize(self):
        tmp_dir = tempfile.TemporaryDirectory()
        model_names = ["bert-base-cased", "distilbert-base-uncased"]

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                onnx_model = Path(tmp_dir.name).joinpath(model_name + ".onnx")

                tokenizer, model, onnx_config, onnx_outputs = convert_to_onnx(
                    model_name,
                    onnx_model,
                    feature="default",
                    opset=12
                )
                validate_model_outputs(onnx_config, tokenizer, model, onnx_model, onnx_outputs, atol=1e-4)

                optimized_model = optimize(onnx_model, model_type="bert", opt_level=1, only_onnxruntime=True)
                validate_model_outputs(onnx_config, tokenizer, model, optimized_model, onnx_outputs, atol=1e-4)

                quantized_model = quantize(optimized_model)
                validate_model_outputs(onnx_config, tokenizer, model, quantized_model, onnx_outputs, atol=1.5)

        tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

