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
import gc
import os
import tempfile
import unittest

import pytest
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PretrainedConfig,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import require_torch_gpu

import onnxruntime
import requests
from huggingface_hub.utils import EntryNotFoundError
from optimum.onnxruntime import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    ONNX_WEIGHTS_NAME,
    ORTModelForCausalLM,
    ORTModelForCustomTasks,
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForQuestionAnswering,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from optimum.onnxruntime.modeling_ort import ORTModel
from optimum.onnxruntime.modeling_seq2seq import ORTDecoder, ORTEncoder
from optimum.pipelines import pipeline
from optimum.utils import CONFIG_NAME
from optimum.utils.testing_utils import require_hf_token
from parameterized import parameterized


MODEL_NAMES = {
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "bert": "hf-internal-testing/tiny-random-bert",
    # FIXME: Error: ONNX export failed: Couldn't export Python operator SymmetricQuantFunction
    # "ibert": "hf-internal-testing/tiny-random-ibert",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
    "electra": "hf-internal-testing/tiny-random-electra",
    "albert": "hf-internal-testing/tiny-random-albert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "t5": "hf-internal-testing/tiny-random-t5",
    "marian": "sshleifer/tiny-marian-en-de",
    "m2m_100": "valhalla/m2m100_tiny_random",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "vit": "hf-internal-testing/tiny-random-vit",
}

SEED = 42


class ORTModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.LOCAL_MODEL_PATH = "assets/onnx"
        self.ONNX_MODEL_ID = "philschmid/distilbert-onnx"
        self.FAIL_ONNX_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.ONNX_SEQ2SEQ_MODEL_ID = "optimum/t5-small"

    def test_load_model_from_local_path(self):
        model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_from_hub(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoder)
        self.assertIsInstance(model.decoder_with_past, ORTDecoder)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_seq2seq_model_without_past_from_hub(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=False)
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoder)
        self.assertTrue(model.decoder_with_past is None)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_without_onnx_model(self):
        with self.assertRaises(EntryNotFoundError):
            ORTModel.from_pretrained(self.FAIL_ONNX_MODEL_ID)

    def test_model_on_cpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)

    # test string device input for to()
    def test_model_on_cpu_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        cpu = torch.device("cpu")
        model.to("cpu")
        self.assertEqual(model.device, cpu)

    @require_torch_gpu
    def test_model_on_gpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, gpu)
    
    # test string device input for to()
    @require_torch_gpu
    def test_model_on_gpu_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to("cuda")
        self.assertEqual(model.device, gpu)

    def test_seq2seq_model_on_cpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.encoder._device, cpu)
        self.assertEqual(model.decoder._device, cpu)
        self.assertEqual(model.decoder_with_past._device, cpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CPUExecutionProvider")
    
    # test string device input for to()
    def test_seq2seq_model_on_cpu_str(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        cpu = torch.device("cpu")
        model.to("cpu")
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.encoder._device, cpu)
        self.assertEqual(model.decoder._device, cpu)
        self.assertEqual(model.decoder_with_past._device, cpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CPUExecutionProvider")

    @require_torch_gpu
    def test_seq2seq_model_on_gpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, gpu)
        self.assertEqual(model.encoder._device, gpu)
        self.assertEqual(model.decoder._device, gpu)
        self.assertEqual(model.decoder_with_past._device, gpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CUDAExecutionProvider")

    # test string device input for to()
    @require_torch_gpu
    def test_seq2seq_model_on_gpu_str(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
        gpu = torch.device("cuda")
        model.to("cuda")
        self.assertEqual(model.device, gpu)
        self.assertEqual(model.encoder._device, gpu)
        self.assertEqual(model.decoder._device, gpu)
        self.assertEqual(model.decoder_with_past._device, gpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CUDAExecutionProvider")

    @require_hf_token
    def test_load_model_from_hub_private(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None))
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and ONNX exported model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_save_seq2seq_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=True)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            # Verify config and ONNX exported encoder, decoder and decoder with past are present in folder
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_save_seq2seq_model_without_past(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID, use_cache=False)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            # Verify config and ONNX exported encoder and decoder present in folder
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_save_model_with_different_name(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_model_name = "model-test.onnx"
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)

            # save two models to simulate a optimization
            model.save_pretrained(tmpdirname)
            model.save_pretrained(tmpdirname, file_name=test_model_name)

            model = ORTModel.from_pretrained(tmpdirname, file_name=test_model_name)

            self.assertEqual(model.latest_model_name, test_model_name)

    @require_hf_token
    def test_save_model_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(
                tmpdirname,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id=self.HUB_REPOSITORY,
                private=True,
            )


class ORTModelForQuestionAnsweringIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "distilbert",
        "bert",
        "camembert",
        "roberta",
        "xlm-roberta",
        "electra",
        "albert",
        "bart",
        "mbart",
    )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForQuestionAnswering.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("start_logits" in onnx_outputs)
        self.assertTrue("end_logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.start_logits, torch.Tensor)
        self.assertIsInstance(onnx_outputs.end_logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.start_logits, transformers_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(onnx_outputs.end_logits, transformers_outputs.end_logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        self.assertEqual(pipe.device, pipe.model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("question-answering")
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer, device=0)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

        gc.collect()


class ORTModelForSequenceClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "distilbert",
        "bert",
        "camembert",
        "roberta",
        "xlm-roberta",
        "electra",
        "albert",
        "bart",
        "mbart",
    )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("text-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    def test_pipeline_zero_shot_classification(self):
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            "typeform/distilbert-base-uncased-mnli", from_transformers=True
        )
        tokenizer = get_preprocessor("typeform/distilbert-base-uncased-mnli")
        pipe = pipeline("zero-shot-classification", model=onnx_model, tokenizer=tokenizer)
        sequence_to_classify = "Who are you voting for in 2020?"
        candidate_labels = ["Europe", "public health", "politics", "elections"]
        hypothesis_template = "This text is about {}."
        outputs = pipe(
            sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template
        )

        # compare model output class
        self.assertTrue(all(score > 0.0 for score in outputs["scores"]))
        self.assertTrue(all(isinstance(label, str) for label in outputs["labels"]))


class ORTModelForTokenClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "distilbert",
        "bert",
        "camembert",
        "roberta",
        "xlm-roberta",
        "electra",
        "albert",
    )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForTokenClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        # TODO: shouldn't it be all instead of any?
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("token-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()


class ORTModelForFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "distilbert",
        "bert",
        "camembert",
        "roberta",
        "xlm-roberta",
        "electra",
        "albert",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("last_hidden_state" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.last_hidden_state, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(
            torch.allclose(onnx_outputs.last_hidden_state, transformers_outputs.last_hidden_state, atol=1e-4)
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("feature-extraction")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()


class ORTModelForCausalLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("gpt2",)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForCausalLM.from_pretrained(MODEL_NAMES["vit"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")

        # General case
        outputs = model.generate(**tokens)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)
        self.assertTrue(len(res[0]) > len(text))

        # With input ids
        tokens = tokenizer(text, return_tensors="pt")
        outputs = model.generate(input_ids=tokens["input_ids"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)
        self.assertTrue(len(res[0]) > len(text))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("text-generation")
        text = "My Name is Philipp and i live"
        outputs = pipe(text)

        # compare model output class
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

        gc.collect()


class ORTModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "vit": "hf-internal-testing/tiny-random-vit",
    }

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForImageClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        set_seed(SEED)
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)

        self.assertIsInstance(onnx_model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        onnx_outputs = onnx_model(**inputs)

        self.assertTrue("logits" in onnx_outputs)
        self.assertTrue(isinstance(onnx_outputs.logits, torch.Tensor))

        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, trtfs_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline_ort_model(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("image-classification")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()


class ORTModelForSeq2SeqLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "t5",
        "bart",
        "mbart",
        "marian",
        "m2m_100",
        "bigbird_pegasus",
    )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAMES["bert"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")

        # General case
        outputs = model.generate(**tokens)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

        # With input ids
        outputs = model.generate(input_ids=tokens["input_ids"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        self.assertIsInstance(onnx_model.decoder, ORTDecoder)
        self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoder)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        onnx_outputs = onnx_model(**tokens, **decoder_inputs)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens, **decoder_inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_text_generation(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)

        # Text2Text generation
        pipe = pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        # Summarization
        pipe = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["translation_text"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        # Text2text generation
        pipe = pipeline("text2text-generation")
        text = "This is a test"
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["generated_text"], str)

        # Summarization
        pipe = pipeline("summarization")
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_de")
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["translation_text"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    def test_pipeline_on_gpu(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    def test_compare_with_and_without_past_key_values_model_outputs(self):
        model_id = MODEL_NAMES["t5"]
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        model_with_pkv = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=True)
        outputs_model_with_pkv = model_with_pkv.generate(**tokens)
        model_without_pkv = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=False)
        outputs_model_without_pkv = model_without_pkv.generate(**tokens)
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))


class ORTModelForCustomTasksIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "sbert": "optimum/sbert-all-MiniLM-L6-with-pooler",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        outputs = model(**tokens)
        self.assertIsInstance(outputs.pooler_output, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline_ort_model(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pipe.device, onnx_model.device)
