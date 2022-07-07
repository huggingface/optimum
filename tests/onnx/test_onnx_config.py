#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
import tempfile
import unittest
from pathlib import Path

import tensorflow as tf
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)
from transformers.modeling_tf_utils import TFPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager
from transformers.utils import TensorType

import onnxruntime

# OnnxConfig wrapper
from optimum.onnx import OnnxConfigWithLoss, OnnxConfigWithPastAndLoss, OnnxSeq2SeqConfigWithPastAndLoss


class TestOnnxConfigWithLoss(unittest.TestCase):
    # @unittest.skip("Skip OnnxConfigWithLoss test.")
    def test_onnx_config_with_loss(self):
        # Prepare model and dataset
        model_checkpoint = "bert-base-uncased"
        models = {
            AutoModelForSequenceClassification.from_pretrained(model_checkpoint),
            TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint),
        }
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        for model in models:
            with self.subTest(model=model):
                with tempfile.TemporaryDirectory() as tmp_dir:

                    # Wrap OnnxConfig
                    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
                        model, feature="sequence-classification"
                    )
                    onnx_config = model_onnx_config(model.config)
                    wrapped_onnx_config = OnnxConfigWithLoss(onnx_config)

                    # Export model from PyTorch to ONNX
                    onnx_model_path = Path(os.path.join(tmp_dir, f"{model_checkpoint}.onnx"))
                    opset = max(onnx_config.default_onnx_opset, 12)
                    _ = export(
                        preprocessor=tokenizer,
                        model=model,
                        config=wrapped_onnx_config,
                        opset=opset,
                        output=onnx_model_path,
                    )

                    # ONNX Runtime Inference
                    ort_sess = onnxruntime.InferenceSession(
                        onnx_model_path.as_posix(),
                        providers=[
                            "CUDAExecutionProvider"
                            if torch.cuda.is_available()
                            and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                            else "CPUExecutionProvider"
                        ],
                    )
                    if issubclass(type(model), PreTrainedModel):
                        inputs = {
                            "input_ids": torch.tensor(
                                [
                                    [101, 100, 100, 100, 100, 100, 100, 102],
                                    [101, 100, 100, 100, 100, 100, 100, 102],
                                    [101, 100, 100, 100, 100, 100, 100, 102],
                                ]
                            ),
                            "token_type_ids": torch.tensor(
                                [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
                            ),
                            "attention_mask": torch.tensor(
                                [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]
                            ),
                            "labels": torch.LongTensor([0, 0, 0]),
                        }
                    elif issubclass(type(model), TFPreTrainedModel):
                        inputs = {
                            "input_ids": tf.constant(
                                [[101, 100, 100, 100, 100, 100, 100, 102], [101, 100, 100, 100, 100, 100, 100, 102]]
                            ),
                            "token_type_ids": tf.constant([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
                            "attention_mask": tf.constant([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]),
                            "labels": tf.constant([0, 0], dtype=tf.int64),
                        }
                    input_names = [ort_input.name for ort_input in ort_sess._inputs_meta]
                    output_names = [output.name for output in ort_sess._outputs_meta]
                    input_feed = dict(
                        map(lambda input_name: (input_name, inputs[input_name].cpu().numpy()), input_names)
                    )
                    ort_outputs = ort_sess.run(output_names, input_feed)
                    pt_outputs = model(**inputs)

                    # Checkers
                    assert len(ort_outputs) > 1, "There is only one element in outputs, the loss might be missing!"
                    if issubclass(type(model), PreTrainedModel):
                        self.assertAlmostEqual(
                            float(ort_outputs[0]),
                            float(pt_outputs["loss"]),
                            3,
                            "The losses of ONNX Runtime and PyTorch inference are not close enough!",
                        )
                    elif issubclass(type(model), TFPreTrainedModel):
                        for ort_loss, pt_loss in zip(ort_outputs[-1], pt_outputs["loss"]):
                            self.assertAlmostEqual(
                                float(ort_loss),
                                float(pt_loss),
                                3,
                                "The losses of ONNX Runtime and PyTorch inference are not close enough!",
                            )
                    gc.collect()

    # @unittest.skip("Skip OnnxConfigWithPastAndLoss test.")
    def test_onnx_config_with_past_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Prepare model and dataset
            model_checkpoint = "gpt2"
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            # Wrap OnnxConfig
            _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
                model, feature="sequence-classification"
            )
            onnx_config = model_onnx_config(model.config)
            wrapped_onnx_config = OnnxConfigWithPastAndLoss(onnx_config)

            # Export model from PyTorch to ONNX
            onnx_model_path = Path(os.path.join(tmp_dir, f"{model_checkpoint}.onnx"))
            opset = max(onnx_config.default_onnx_opset, 12)
            _ = export(
                preprocessor=tokenizer, model=model, config=wrapped_onnx_config, opset=opset, output=onnx_model_path
            )

            # ONNX Runtime Inference
            ort_sess = onnxruntime.InferenceSession(
                onnx_model_path.as_posix(),
                providers=[
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available() and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                    else "CPUExecutionProvider"
                ],
            )
            inputs = {
                "input_ids": torch.tensor(
                    [
                        [50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
                        [50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
                        [50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]
                ),
                "labels": torch.LongTensor([0, 0, 0]),
            }
            input_names = [ort_input.name for ort_input in ort_sess._inputs_meta]
            output_names = [output.name for output in ort_sess._outputs_meta]
            input_feed = dict(map(lambda input_name: (input_name, inputs[input_name].cpu().numpy()), input_names))
            ort_outputs = ort_sess.run(output_names, input_feed)
            pt_outputs = model(**inputs)

            # Checkers
            assert len(ort_outputs) > 1, "There is only one element in outputs, the loss might be missing!"
            self.assertAlmostEqual(
                float(ort_outputs[0]),
                float(pt_outputs["loss"]),
                3,
                "The losses of ONNX Runtime and PyTorch inference are not close enough!",
            )
            gc.collect()

    # @unittest.skip("Skip OnnxSeq2SeqConfigWithPastAndLoss test.")
    def test_onnx_seq2seq_config_with_past_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Prepare model and dataset
            model_checkpoint = "t5-small"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            # Wrap OnnxConfig
            _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="seq2seq-lm")
            onnx_config = model_onnx_config(model.config)
            wrapped_onnx_config = OnnxSeq2SeqConfigWithPastAndLoss(onnx_config)

            # Export model from PyTorch to ONNX
            onnx_model_path = Path(os.path.join(tmp_dir, f"{model_checkpoint}.onnx"))
            opset = max(onnx_config.default_onnx_opset, 12)

            _ = export(
                preprocessor=tokenizer, model=model, config=wrapped_onnx_config, opset=opset, output=onnx_model_path
            )

            # ONNX Runtime Inference
            ort_sess = onnxruntime.InferenceSession(
                onnx_model_path.as_posix(),
                providers=[
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available() and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                    else "CPUExecutionProvider"
                ],
            )
            inputs = {
                "input_ids": torch.tensor(
                    [
                        [2, 2, 2, 2, 2, 2, 2, 1],
                        [2, 2, 2, 2, 2, 2, 2, 1],
                        [2, 2, 2, 2, 2, 2, 2, 1],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]
                ),
                "decoder_input_ids": torch.tensor(
                    [
                        [2, 2, 2, 2, 2, 2, 2, 1],
                        [2, 2, 2, 2, 2, 2, 2, 1],
                        [2, 2, 2, 2, 2, 2, 2, 1],
                    ]
                ),
                "decoder_attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                    ]
                ),
                "labels": torch.LongTensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            }
            input_names = [ort_input.name for ort_input in ort_sess._inputs_meta]
            output_names = [output.name for output in ort_sess._outputs_meta]
            input_feed = dict(map(lambda input_name: (input_name, inputs[input_name].cpu().numpy()), input_names))
            ort_outputs = ort_sess.run(output_names, input_feed)
            pt_outputs = model(**inputs)

            # Checkers
            assert len(ort_outputs) > 1, "There is only one element in outputs, the loss might be missing!"
            self.assertAlmostEqual(
                float(ort_outputs[0]),
                float(pt_outputs["loss"]),
                3,
                "The losses of ONNX Runtime and PyTorch inference are not close enough",
            )
            gc.collect()


if __name__ == "__main__":
    unittest.main()
