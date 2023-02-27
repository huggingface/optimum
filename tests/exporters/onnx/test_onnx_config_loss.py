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

import onnxruntime
import pytest
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_tf_available

from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfigWithLoss, export

# OnnxConfig wrapper
from optimum.onnxruntime.utils import ONNX_DECODER_NAME
from optimum.utils import DummyTextInputGenerator
from optimum.utils.normalized_config import NormalizedConfigManager


if is_tf_available():
    from transformers import TFAutoModelForSequenceClassification
    from transformers.modeling_tf_utils import TFPreTrainedModel


class TestOnnxConfigWithLoss(unittest.TestCase):
    @pytest.mark.tensorflow_test
    def test_onnx_config_with_loss(self):
        # Prepare model and dataset
        model_checkpoint = "hf-internal-testing/tiny-random-bert"
        models = {
            AutoModelForSequenceClassification.from_pretrained(model_checkpoint),
            TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint),
        }

        for model in models:
            with self.subTest(model=model):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                        model=model, exporter="onnx", task="sequence-classification"
                    )
                    onnx_config = onnx_config_constructor(model.config)

                    wrapped_onnx_config = OnnxConfigWithLoss(onnx_config)

                    # Export model from PyTorch to ONNX
                    onnx_model_path = Path(os.path.join(tmp_dir, f"{model.config.model_type}.onnx"))
                    opset = max(onnx_config.DEFAULT_ONNX_OPSET, 12)
                    _ = export(
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
                    framework = "pt" if isinstance(model, PreTrainedModel) else "tf"
                    normalized_config = NormalizedConfigManager.get_normalized_config_class("bert")(model.config)
                    input_generator = DummyTextInputGenerator(
                        "sequence-classification", normalized_config, batch_size=2, sequence_length=16
                    )

                    inputs = {
                        name: input_generator.generate(name, framework=framework)
                        for name in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    inputs["labels"] = input_generator.constant_tensor(
                        [2], value=0, dtype=inputs["input_ids"].dtype, framework=framework
                    )

                    input_names = [ort_input.name for ort_input in ort_sess._inputs_meta]
                    output_names = [output.name for output in ort_sess._outputs_meta]

                    input_feed = {input_name: inputs[input_name].cpu().numpy() for input_name in input_names}

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

    def test_onnx_decoder_model_with_config_with_loss(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Prepare model and dataset
            model_checkpoint = "hf-internal-testing/tiny-random-gpt2"
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

            # Wrap OnnxConfig
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                model=model, exporter="onnx", task="sequence-classification"
            )
            onnx_config = onnx_config_constructor(model.config)
            wrapped_onnx_config = OnnxConfigWithLoss(onnx_config)

            # Export model from PyTorch to ONNX
            onnx_model_path = Path(os.path.join(tmp_dir, f"{model.config.model_type}.onnx"))
            opset = max(onnx_config.DEFAULT_ONNX_OPSET, 12)
            export(model=model, config=wrapped_onnx_config, opset=opset, output=onnx_model_path)

            # ONNX Runtime Inference
            ort_sess = onnxruntime.InferenceSession(
                onnx_model_path.as_posix(),
                providers=[
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available() and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                    else "CPUExecutionProvider"
                ],
            )

            batch = 3
            seq_length = 8
            inputs = {
                "input_ids": torch.ones((batch, seq_length), dtype=torch.long),
                "attention_mask": torch.ones((batch, seq_length), dtype=torch.long),
                "labels": torch.zeros(batch, dtype=torch.long),
            }
            input_names = [ort_input.name for ort_input in ort_sess._inputs_meta]
            output_names = [output.name for output in ort_sess._outputs_meta]
            input_feed = {input_name: inputs[input_name].cpu().numpy() for input_name in input_names}
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

    def test_onnx_seq2seq_model_with_config_with_loss(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Prepare model and dataset
            model_checkpoint = "hf-internal-testing/tiny-random-t5"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

            # Wrap OnnxConfig(decoders)
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                model=model, exporter="onnx", task="seq2seq-lm"
            )
            onnx_config = onnx_config_constructor(model.config)

            onnx_config_decoder = onnx_config.with_behavior("decoder", use_past=False)

            wrapped_onnx_config_decoder = OnnxConfigWithLoss(onnx_config_decoder)

            # Export decoder models from PyTorch to ONNX
            opset = max(onnx_config.DEFAULT_ONNX_OPSET, 12)

            onnx_model_path = Path(tmp_dir).joinpath(ONNX_DECODER_NAME)

            export(
                model=model,
                config=wrapped_onnx_config_decoder,
                opset=opset,
                output=onnx_model_path,
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
            batch = 3
            encoder_seq_length = 8
            encoder_hidden_states_shape = (batch, encoder_seq_length, model.config.hidden_size)
            inputs = {
                "input_ids": torch.ones((batch, encoder_seq_length), dtype=torch.long),
                "encoder_hidden_states": torch.zeros(encoder_hidden_states_shape),
                "encoder_attention_mask": torch.ones((batch, encoder_seq_length), dtype=torch.long),
                "labels": torch.zeros((batch, encoder_seq_length), dtype=torch.long),
            }
            input_names = [ort_input.name for ort_input in ort_sess._inputs_meta]
            output_names = [output.name for output in ort_sess._outputs_meta]
            input_feed = {input_name: inputs[input_name].cpu().numpy() for input_name in input_names}

            ort_sess.run(output_names, input_feed)

            gc.collect()


if __name__ == "__main__":
    unittest.main()
