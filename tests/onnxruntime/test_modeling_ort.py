import os
import shutil
import tempfile
import unittest
from pathlib import Path

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
    pipeline,
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
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForQuestionAnswering,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from optimum.onnxruntime.modeling_ort import ORTModel
from optimum.onnxruntime.modeling_seq2seq import ORTDecoder, ORTEncoder
from optimum.utils import CONFIG_NAME
from optimum.utils.testing_utils import require_hf_token
from parameterized import parameterized


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
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID)
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoder)
        self.assertIsInstance(model.decoder_with_past, ORTDecoder)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_without_onnx_model(self):
        with self.assertRaises(EntryNotFoundError):
            ORTModel.from_pretrained(self.FAIL_ONNX_MODEL_ID)

    def test_model_on_cpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)

    @require_torch_gpu
    def test_model_on_gpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, gpu)

    def test_seq2seq_model_on_cpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertEqual(model.encoder._device, cpu)
        self.assertEqual(model.decoder._device, cpu)
        self.assertEqual(model.decoder_with_past._device, cpu)
        self.assertEqual(model.encoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder.session.get_providers()[0], "CPUExecutionProvider")
        self.assertEqual(model.decoder_with_past.session.get_providers()[0], "CPUExecutionProvider")

    @require_torch_gpu
    def test_seq2seq_model_on_gpu(self):
        model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
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
            model = ORTModelForSeq2SeqLM.from_pretrained(self.ONNX_SEQ2SEQ_MODEL_ID)
            model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            # Verify config and ONNX exported encoder, decoder and decoder present in folder
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME in folder_contents)
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
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "distilbert": "hf-internal-testing/tiny-random-distilbert",
        "bert": "hf-internal-testing/tiny-random-bert",
        # FIXME: Error: ONNX export failed: Couldn't export Python operator SymmetricQuantFunction
        # "ibert": "hf-internal-testing/tiny-random-ibert",
        "camembert": "etalab-ia/camembert-base-squadFR-fquad-piaf",
        "roberta": "hf-internal-testing/tiny-random-roberta",
        # TODO: used real model do to big difference in output
        # "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
        "xlm-roberta": "deepset/xlm-roberta-base-squad2",
        "electra": "hf-internal-testing/tiny-random-electra",
        "albert": "hf-internal-testing/tiny-random-albert",
        "bart": "hf-internal-testing/tiny-random-bart",
        "mbart": "hf-internal-testing/tiny-random-mbart",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = ORTModelForQuestionAnswering.from_pretrained("t5-small", from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("start_logits" in outputs)
        self.assertTrue("end_logits" in outputs)

        self.assertIsInstance(outputs.start_logits, torch.Tensor)
        self.assertIsInstance(outputs.end_logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        onnx_outputs = onnx_model(**tokens)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.start_logits, transformers_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(onnx_outputs.end_logits, transformers_outputs.end_logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer, device=0)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pp(question, context)
        # check model device
        self.assertEqual(pp.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pp.device, pp.model.device)


class ORTModelForSequenceClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "distilbert": "hf-internal-testing/tiny-random-distilbert",
        "bert": "hf-internal-testing/tiny-random-bert",
        # FIXME: Error: ONNX export failed: Couldn't export Python operator SymmetricQuantFunction
        # "ibert": "hf-internal-testing/tiny-random-ibert",
        "camembert": "cmarkea/distilcamembert-base-sentiment",
        "roberta": "hf-internal-testing/tiny-random-roberta",
        # TODO: used real model do to big difference in output
        # "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
        "xlm-roberta": "unitary/multilingual-toxic-xlm-roberta",
        "electra": "hf-internal-testing/tiny-random-electra",
        "albert": "hf-internal-testing/tiny-random-albert",
        "bart": "hf-internal-testing/tiny-random-bart",
        "mbart": "hf-internal-testing/tiny-random-mbart",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = ORTModelForSequenceClassification.from_pretrained("t5-small", from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_forward_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        outputs = model(**tokens)
        self.assertTrue("logits" in outputs)
        self.assertIsInstance(outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        onnx_outputs = onnx_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pp.device, onnx_model.device)

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
        self.assertTrue(any(score > 0.0 for score in outputs["scores"]))
        self.assertTrue(any(isinstance(label, str) for label in outputs["labels"]))


class ORTModelForTokenClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "distilbert": "hf-internal-testing/tiny-random-distilbert",
        "bert": "hf-internal-testing/tiny-random-bert",
        # FIXME: Error: ONNX export failed: Couldn't export Python operator SymmetricQuantFunction
        # "ibert": "hf-internal-testing/tiny-random-ibert",
        "camembert": "cmarkea/distilcamembert-base-ner",
        "roberta": "hf-internal-testing/tiny-random-roberta",
        # TODO: used real model do to big difference in output
        # "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
        "xlm-roberta": "Davlan/xlm-roberta-base-wikiann-ner",
        "electra": "hf-internal-testing/tiny-random-electra",
        "albert": "hf-internal-testing/tiny-random-albert",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = ORTModelForTokenClassification.from_pretrained("t5-small", from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("logits" in outputs)
        self.assertIsInstance(outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        onnx_outputs = onnx_model(**tokens)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(any(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(any(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pp.device, onnx_model.device)


class ORTModelForFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "distilbert": "hf-internal-testing/tiny-random-distilbert",
        "bert": "hf-internal-testing/tiny-random-bert",
        # FIXME: Error: ONNX export failed: Couldn't export Python operator SymmetricQuantFunction
        # "ibert": "hf-internal-testing/tiny-random-ibert",
        "camembert": "cmarkea/distilcamembert-base",
        "roberta": "hf-internal-testing/tiny-random-roberta",
        # TODO: used real model do to big difference in output
        # "xlm-roberta": "hf-internal-testing/tiny -xlm-roberta",
        "xlm-roberta": "xlm-roberta-base",
        "electra": "hf-internal-testing/tiny-random-electra",
        "albert": "hf-internal-testing/tiny-random-albert",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("last_hidden_state" in outputs)
        self.assertIsInstance(outputs.last_hidden_state, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        onnx_outputs = onnx_model(**tokens)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(
            torch.allclose(onnx_outputs.last_hidden_state, transformers_outputs.last_hidden_state, atol=1e-4)
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pp.device, onnx_model.device)


class ORTModelForCausalLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "gpt2": "hf-internal-testing/tiny-random-gpt2",
        "distilgpt2": "distilgpt2",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = ORTModelForCausalLM.from_pretrained("google/vit-base-patch16-224", from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("logits" in outputs)
        self.assertIsInstance(outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_generate_utils(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(
            text,
            return_tensors="pt",
        )
        outputs = model.generate(**tokens)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)
        self.assertTrue(len(res[0]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_generate_utils_with_input_ids(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(
            text,
            return_tensors="pt",
        )
        outputs = model.generate(input_ids=tokens["input_ids"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)
        self.assertTrue(len(res[0]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        onnx_outputs = onnx_model(**tokens)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pipe(text)

        # compare model output class
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pp = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pp.device, onnx_model.device)


class ORTModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "vit": "hf-internal-testing/tiny-random-vit",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = ORTModelForImageClassification.from_pretrained("t5-small", from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_forward_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        self.assertTrue("logits" in outputs)
        self.assertTrue(isinstance(outputs.logits, torch.Tensor))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        trfs_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)
        onnx_outputs = onnx_model(**inputs)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, trtfs_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = get_preprocessor(model_id)
        pp = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pp(url)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = get_preprocessor(model_id)
        pp = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pp(url)
        # check model device
        self.assertEqual(pp.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = get_preprocessor(model_id)
        pp = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor)
        self.assertEqual(pp.device, onnx_model.device)


class ORTModelForSeq2SeqLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "t5": "hf-internal-testing/tiny-random-t5",
        "bart": "hf-internal-testing/tiny-random-bart",
        "mbart": "hf-internal-testing/tiny-random-mbart",
        "marian": "sshleifer/tiny-marian-en-de",
        "m2m_100": "valhalla/m2m100_tiny_random",
        "longt5": "google/long-t5-local-base",
        "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        self.assertIsInstance(model.encoder, ORTEncoder)
        self.assertIsInstance(model.decoder, ORTDecoder)
        self.assertIsInstance(model.decoder_with_past, ORTDecoder)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = ORTModelForSeq2SeqLM.from_pretrained("bert-base-uncased", from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        decoder_start_token_id = model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        outputs = model(**tokens, **decoder_inputs)
        self.assertTrue("logits" in outputs)
        self.assertIsInstance(outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_generate_utils(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**tokens)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_generate_utils_with_input_ids(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        outputs = model.generate(input_ids=tokens["input_ids"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        decoder_start_token_id = model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        onnx_outputs = onnx_model(**tokens, **decoder_inputs)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens, **decoder_inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline_text_generation(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)

        # compare model output class
        self.assertIsInstance(outputs[0]["generated_text"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline_text_generation(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)

        # compare model output class
        self.assertIsInstance(outputs[0]["summary_text"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline_text_generation(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)

        # compare model output class
        self.assertIsInstance(outputs[0]["translation_text"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pipe.device, onnx_model.device)
