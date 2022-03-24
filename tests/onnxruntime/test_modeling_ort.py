import os
import tempfile
import unittest

import torch
from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    pipeline,
)

import onnxruntime
from optimum.onnxruntime import (
    CONFIG_NAME,
    ONNX_WEIGHTS_NAME,
    OnnxForFeatureExtraction,
    OnnxForQuestionAnswering,
    OnnxForSequenceClassification,
    OnnxForTokenClassification,
    OnnxModel,
)
from optimum.utils import CONFIG_NAME
from optimum.utils.testing_utils import require_hf_token
from parameterized import parameterized


class OnnxModelIntergrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.LOCAL_MODEL_PATH = "tests/assets/onnx"
        self.ONNX_MODEL_ID = "philschmid/distilbert-onnx"
        self.FAIL_ONNX_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"

    def test_load_model_from_local_path(self):
        model = OnnxModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = OnnxModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_without_onnx_model(self):
        with self.assertRaises(Exception) as context:
            OnnxModel.from_pretrained(self.FAIL_ONNX_MODEL_ID)
        self.assertEqual("Not Found", context.exception.response.reason)

    @require_hf_token
    def test_load_model_from_hub_private(self):
        model = OnnxModel.from_pretrained(self.ONNX_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None))
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = OnnxModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and pytorch_model.bin
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    @require_hf_token
    def test_save_model_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = OnnxModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(
                tmpdirname,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id=self.HUB_REPOSITORY,
                private=True,
            )


class OnnxForQuestionAnsweringIntergrationTest(unittest.TestCase):
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
        model = OnnxForQuestionAnswering.from_transformers(model_id)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = OnnxForQuestionAnswering.from_transformers("t5-small")

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = OnnxForQuestionAnswering.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("start_logits" in outputs)
        self.assertTrue("end_logits" in outputs)

        self.assertTrue(isinstance(outputs.start_logits, torch.Tensor))
        self.assertTrue(isinstance(outputs.end_logits, torch.Tensor))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForQuestionAnswering.from_transformers(model_id)
        trfs_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        onnx_outputs = onnx_model(**tokens)
        with torch.no_grad():
            trtfs_outputs = trfs_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.start_logits, trtfs_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(onnx_outputs.end_logits, trtfs_outputs.end_logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForQuestionAnswering.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pp(question, context)

        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))


class OnnxForSequenceClassificationIntergrationTest(unittest.TestCase):
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
        model = OnnxForSequenceClassification.from_transformers(model_id)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = OnnxForSequenceClassification.from_transformers("t5-small")

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_forward_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = OnnxForSequenceClassification.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("logits" in outputs)
        self.assertTrue(isinstance(outputs.logits, torch.Tensor))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForSequenceClassification.from_transformers(model_id)
        trfs_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        with torch.no_grad():
            trtfs_outputs = trfs_model(**tokens)
        onnx_outputs = onnx_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, trtfs_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForSequenceClassification.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    def test_pipeline_zero_shot_classification(self):
        onnx_model = OnnxForSequenceClassification.from_transformers("typeform/distilbert-base-uncased-mnli")
        tokenizer = AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")
        pp = pipeline("zero-shot-classification", model=onnx_model, tokenizer=tokenizer)
        sequence_to_classify = "Who are you voting for in 2020?"
        candidate_labels = ["Europe", "public health", "politics", "elections"]
        hypothesis_template = "This text is about {}."
        outputs = pp(sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template)

        # compare model output class
        self.assertTrue(any(score > 0.0 for score in outputs["scores"]))
        self.assertTrue(any(isinstance(label, str) for label in outputs["labels"]))


class OnnxForTokenClassificationIntergrationTest(unittest.TestCase):
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
        model = OnnxForTokenClassification.from_transformers(model_id)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = OnnxForTokenClassification.from_transformers("t5-small")

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = OnnxForTokenClassification.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("logits" in outputs)
        self.assertTrue(isinstance(outputs.logits, torch.Tensor))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForTokenClassification.from_transformers(model_id)
        trfs_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        onnx_outputs = onnx_model(**tokens)
        with torch.no_grad():
            trtfs_outputs = trfs_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.logits, trtfs_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForTokenClassification.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)

        # compare model output class
        self.assertTrue(any(item["score"] > 0.0 for item in outputs))


class OnnxForFeatureExtractionIntergrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {
        "distilbert": "hf-internal-testing/tiny-random-distilbert",
        "bert": "hf-internal-testing/tiny-random-bert",
        # FIXME: Error: ONNX export failed: Couldn't export Python operator SymmetricQuantFunction
        # "ibert": "hf-internal-testing/tiny-random-ibert",
        "camembert": "cmarkea/distilcamembert-base",
        "roberta": "hf-internal-testing/tiny-random-roberta",
        # TODO: used real model do to big difference in output
        # "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
        "xlm-roberta": "xlm-roberta-base",
        "electra": "hf-internal-testing/tiny-random-electra",
        "albert": "hf-internal-testing/tiny-random-albert",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_supported_transformers_architectures(self, *args, **kwargs):
        model_arch, model_id = args
        model = OnnxForFeatureExtraction.from_transformers(model_id)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = OnnxForFeatureExtraction.from_transformers("google/vit-base-patch16-224")

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = OnnxForFeatureExtraction.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("last_hidden_state" in outputs)
        self.assertTrue(isinstance(outputs.last_hidden_state, torch.Tensor))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForFeatureExtraction.from_transformers(model_id)
        trfs_model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        onnx_outputs = onnx_model(**tokens)
        with torch.no_grad():
            trtfs_outputs = trfs_model(**tokens)

        # compare tensor outputs
        self.assertTrue(torch.allclose(onnx_outputs.last_hidden_state, trtfs_outputs.last_hidden_state, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = OnnxForFeatureExtraction.from_transformers(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)

        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))
