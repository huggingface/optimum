#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

import gc
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForSequenceClassification,
    BertForSequenceClassification,
    BigBirdForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification,
    GPT2ForSequenceClassification,
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
)

import onnx
from onnx import load as onnx_load
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTConfig, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoQuantizationConfig,
    OptimizationConfig,
    QuantizationConfig,
)
from optimum.onnxruntime.modeling_ort import ORTModelForSequenceClassification
from parameterized import parameterized


class ORTConfigTest(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            optimization_config = OptimizationConfig(optimization_level=2)
            ort_config = ORTConfig(opset=11, quantization=quantization_config, optimization=optimization_config)
            ort_config.save_pretrained(tmp_dir)
            loaded_ort_config = ORTConfig.from_pretrained(tmp_dir)
            self.assertEqual(ort_config.to_dict(), loaded_ort_config.to_dict())


class ORTOptimizerTest(unittest.TestCase):

    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = (
        (BertForSequenceClassification, "hf-internal-testing/tiny-random-bert"),
        (DistilBertForSequenceClassification, "hf-internal-testing/tiny-random-distilbert"),
        (BartForSequenceClassification, "hf-internal-testing/tiny-random-bart"),
        (GPT2ForSequenceClassification, "hf-internal-testing/tiny-random-gpt2"),
        (RobertaForSequenceClassification, "hf-internal-testing/tiny-random-roberta"),
        (ElectraForSequenceClassification, "hf-internal-testing/tiny-random-electra"),
        (XLMRobertaForSequenceClassification, "hf-internal-testing/tiny-xlm-roberta"),
        (BigBirdForSequenceClassification, "hf-internal-testing/tiny-random-big_bird"),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID)
    def test_optimize(self, model_cls, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_cls(AutoConfig.from_pretrained(model_name))
        optimization_config = OptimizationConfig(optimization_level=2, optimize_with_onnxruntime_only=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            optimized_model_path = output_dir.joinpath("model-optimized.onnx")
            optimizer = ORTOptimizer(tokenizer, model, feature="sequence-classification")
            optimizer.export(
                onnx_model_path=model_path,
                onnx_optimized_model_output_path=optimized_model_path,
                optimization_config=optimization_config,
            )
            input = "This is a sample input"
            with torch.no_grad():
                original_outputs = optimizer.model(**optimizer.preprocessor(input, return_tensors="pt"))
            session = InferenceSession(optimized_model_path.as_posix(), providers=["CPUExecutionProvider"])
            ort_input = dict(optimizer.preprocessor(input, return_tensors="np"))
            ort_input = {k: v.astype(np.int64) for k, v in ort_input.items()}
            ort_outputs = session.run(None, ort_input)
            self.assertTrue(np.allclose(original_outputs.logits.cpu().numpy(), ort_outputs[0], atol=1e-4))
            gc.collect()

    def test_optimization_details(self):
        model_name = "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification(AutoConfig.from_pretrained(model_name))
        optimization_config = OptimizationConfig(optimization_level=0, optimize_with_onnxruntime_only=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            optimized_model_path = output_dir.joinpath("model-optimized.onnx")
            optimizer = ORTOptimizer(tokenizer, model, feature="sequence-classification")
            optimizer.export(
                onnx_model_path=model_path,
                onnx_optimized_model_output_path=optimized_model_path,
                optimization_config=optimization_config,
            )
            difference_nodes_number = optimizer.get_nodes_number_difference(model_path, optimized_model_path)
            fused_operator = optimizer.get_fused_operators(model_path)
            sorted_operators_difference = optimizer.get_operators_difference(model_path, optimized_model_path)
            self.assertEqual(difference_nodes_number, 0)
            self.assertEqual(len(fused_operator), 0)
            self.assertEqual(len(sorted_operators_difference), 0)
            gc.collect()

    def test_optimization_fp16(self):
        model_name = "hf-internal-testing/tiny-random-distilbert"
        optimization_config = OptimizationConfig(optimization_level=0, fp16=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            optimized_model_path = output_dir.joinpath("model-optimized.onnx")
            onnx_model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
            onnx_model.save_pretrained(output_dir.as_posix())

            optimizer = ORTOptimizer.from_pretrained(model_name, feature="sequence-classification")
            optimizer.export(
                onnx_model_path=model_path,
                onnx_optimized_model_output_path=optimized_model_path,
                optimization_config=optimization_config,
            )
            model = onnx.load(optimized_model_path.as_posix())
            for w in model.graph.initializer:
                self.assertNotEqual(w.data_type, onnx.onnx_pb.TensorProto.FLOAT)

            onnx_model = ORTModelForSequenceClassification.from_pretrained(
                output_dir.as_posix(), file_name="model-optimized.onnx"
            )
            transformers_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer("This is a sample output", return_tensors="pt")
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            onnx_outputs = onnx_model(**tokens)

            # compare tensor outputs
            self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))


class ORTDynamicQuantizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        (BertForSequenceClassification, "hf-internal-testing/tiny-random-bert", 30),
        (RobertaForSequenceClassification, "hf-internal-testing/tiny-random-roberta", 30),
        (DistilBertForSequenceClassification, "hf-internal-testing/tiny-random-distilbert", 30),
        (BartForSequenceClassification, "hf-internal-testing/tiny-random-bart", 32),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_dynamic_quantization(self, model_cls, model_name, expected_quantized_matmuls):
        qconfig = QuantizationConfig(
            is_static=False,
            format=QuantFormat.QOperator,
            mode=QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QUInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul"],
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_cls(AutoConfig.from_pretrained(model_name))
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            q8_model_path = output_dir.joinpath("model-quantized.onnx")
            quantizer = ORTQuantizer(tokenizer, model, feature="sequence-classification")
            quantizer.export(
                onnx_model_path=model_path,
                onnx_quantized_model_output_path=q8_model_path,
                calibration_tensors_range=None,
                quantization_config=qconfig,
            )
            quantized_model = onnx_load(q8_model_path)
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "MatMul" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)
            gc.collect()


class ORTStaticQuantizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        (BertForSequenceClassification, "hf-internal-testing/tiny-random-bert", 30),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_static_quantization(self, model_cls, model_name, expected_quantized_matmuls):
        qconfig = QuantizationConfig(
            is_static=True,
            format=QuantFormat.QDQ,
            mode=QuantizationMode.QLinearOps,
            activations_dtype=QuantType.QInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul"],
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_cls(AutoConfig.from_pretrained(model_name))

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_path = output_dir.joinpath("model.onnx")
            q8_model_path = output_dir.joinpath("model-quantized.onnx")
            quantizer = ORTQuantizer(tokenizer, model, feature="sequence-classification")
            calibration_dataset = quantizer.get_calibration_dataset(
                "glue",
                dataset_config_name="sst2",
                preprocess_function=partial(preprocess_function, tokenizer=quantizer.preprocessor),
                num_samples=40,
                dataset_split="train",
            )
            calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
            ranges = quantizer.fit(
                dataset=calibration_dataset,
                calibration_config=calibration_config,
                onnx_model_path=model_path,
            )
            quantizer.export(
                onnx_model_path=model_path,
                onnx_quantized_model_output_path=q8_model_path,
                calibration_tensors_range=ranges,
                quantization_config=qconfig,
            )

            quantized_model = onnx_load(q8_model_path)
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "MatMul" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)
            gc.collect()


if __name__ == "__main__":
    unittest.main()
