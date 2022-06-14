import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    pipeline,
)
from transformers.testing_utils import require_torch_gpu

import onnxruntime
from optimum.onnxruntime import (
    ONNX_WEIGHTS_NAME,
    ORTModelForCausalLM,
    ORTModelForFeatureExtraction,
    ORTModelForQuestionAnswering,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from optimum.onnxruntime.modeling_ort import ORTModel
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

    def test_load_model_from_local_path(self):
        model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_without_onnx_model(self):
        with self.assertRaises(Exception) as context:
            ORTModel.from_pretrained(self.FAIL_ONNX_MODEL_ID)
        self.assertEqual("Not Found", context.exception.response.reason)

    def test_model_on_cpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertEqual(model.device, torch.device("cpu"))

    @require_hf_token
    def test_load_model_from_hub_private(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None))
        self.assertIsInstance(model.model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and pytorch_model.bin
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_save_model_with_different_name(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_model_name = "model-test.onnx"
            local_model_path = str(Path(self.LOCAL_MODEL_PATH).joinpath("model.onnx").absolute())
            # copy two models to simulate a optimization
            shutil.copy(local_model_path, os.path.join(tmpdirname, test_model_name))
            shutil.copy(local_model_path, os.path.join(tmpdirname, "model.onnx"))

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
            model = ORTModelForQuestionAnswering.from_pretrained("t5-small")

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pp(question, context)

        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer, device=0)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pp(question, context)
        # check model device
        self.assertEqual(pp.model.device.type, "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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
            model = ORTModelForSequenceClassification.from_pretrained("t5-small", from_transformers=Tru)

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_forward_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type, "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pp.device, onnx_model.device)

    def test_pipeline_zero_shot_classification(self):
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            "typeform/distilbert-base-uncased-mnli", from_transformers=True
        )
        tokenizer = AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")
        pp = pipeline("zero-shot-classification", model=onnx_model, tokenizer=tokenizer)
        sequence_to_classify = "Who are you voting for in 2020?"
        candidate_labels = ["Europe", "public health", "politics", "elections"]
        hypothesis_template = "This text is about {}."
        outputs = pp(sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template)

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
            model = ORTModelForTokenClassification.from_pretrained("t5-small", from_transformers=Tru)

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)

        # compare model output class
        self.assertTrue(any(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type, "cuda")
        # compare model output class
        self.assertTrue(any(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            model = ORTModelForFeatureExtraction.from_pretrained("google/vit-base-patch16-224", from_transformers=Tru)

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
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
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)

        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type, "cuda")
        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

        self.assertTrue("Unrecognized configuration class", context.exception)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample output",
            return_tensors="pt",
        )
        outputs = model(**tokens)
        self.assertTrue("logits" in outputs)
        self.assertTrue(isinstance(outputs.logits, torch.Tensor))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_generate_utils(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "This is a sample output"
        tokens = tokenizer(
            text,
            return_tensors="pt",
        )
        outputs = model.generate(**tokens)
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertTrue(isinstance(res[0], str))
        self.assertTrue(len(res[0]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_generate_utils_with_input_ids(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "This is a sample output"
        tokens = tokenizer(
            text,
            return_tensors="pt",
        )
        outputs = model.generate(input_ids=tokens["input_ids"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertTrue(isinstance(res[0], str))
        self.assertTrue(len(res[0]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_compare_to_transformers(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        trfs_model = AutoModelForCausalLM.from_pretrained(model_id)
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
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pp(text)

        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live"
        outputs = pp(text)
        # check model device
        self.assertEqual(pp.model.device.type, "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))
        self.assertTrue(len(outputs[0]["generated_text"]) > len(text))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCausalLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pp = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pp.device, onnx_model.device)
