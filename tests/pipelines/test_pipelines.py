# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import unittest
from typing import Any, Dict

import numpy as np
from huggingface_hub.constants import HF_HUB_CACHE
from PIL import Image
from transformers import AutoTokenizer
from transformers.pipelines import Pipeline

from optimum.pipelines import pipeline as optimum_pipeline
from optimum.utils.testing_utils import remove_directory


GENERATE_KWARGS = {"max_new_tokens": 10, "min_new_tokens": 5, "do_sample": True}


class ORTPipelineTest(unittest.TestCase):
    """Test ORT pipelines for all supported tasks"""

    def _create_dummy_text(self) -> str:
        """Create dummy text input for text-based tasks"""
        return "This is a test sentence for the pipeline."

    def _create_dummy_image(self) -> Image.Image:
        """Create dummy image input for image-based tasks"""
        np_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(np_image)

    def _create_dummy_audio(self) -> Dict[str, Any]:
        """Create dummy audio input for audio-based tasks"""
        sample_rate = 16000
        audio_array = np.random.randn(sample_rate).astype(np.float32)
        return {"array": audio_array, "sampling_rate": sample_rate}

    def test_text_classification_pipeline(self):
        """Test text classification ORT pipeline"""
        pipe = optimum_pipeline(task="text-classification", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = self._create_dummy_text()
        result = pipe(text)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("label", result[0])
        self.assertIn("score", result[0])

    def test_token_classification_pipeline(self):
        """Test token classification ORT pipeline"""
        pipe = optimum_pipeline(task="token-classification", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = self._create_dummy_text()
        result = pipe(text)

        self.assertIsInstance(result, list)
        if len(result) > 0:
            self.assertIn("entity", result[0])
            self.assertIn("score", result[0])
            self.assertIn("word", result[0])

    def test_question_answering_pipeline(self):
        """Test question answering ORT pipeline"""
        pipe = optimum_pipeline(task="question-answering", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        question = "What animal jumps?"
        context = "The quick brown fox jumps over the lazy dog."
        result = pipe(question=question, context=context)

        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("score", result)
        self.assertIn("start", result)
        self.assertIn("end", result)

    def test_fill_mask_pipeline(self):
        """Test fill mask ORT pipeline"""
        pipe = optimum_pipeline(task="fill-mask", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = "The weather is <mask> today."
        result = pipe(text)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("token_str", result[0])
        self.assertIn("score", result[0])

    def test_feature_extraction_pipeline(self):
        """Test feature extraction ORT pipeline"""
        pipe = optimum_pipeline(task="feature-extraction", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = self._create_dummy_text()
        result = pipe(text)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[0][0], list)

    def test_text_generation_pipeline(self):
        """Test text generation ORT pipeline"""
        pipe = optimum_pipeline(task="text-generation", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = "The future of AI is"
        result = pipe(text, **GENERATE_KWARGS)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("generated_text", result[0])
        self.assertTrue(result[0]["generated_text"].startswith(text))

    def test_summarization_pipeline(self):
        """Test summarization ORT pipeline"""
        pipe = optimum_pipeline(task="summarization", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = "The quick brown fox jumps over the lazy dog."
        result = pipe(text, **GENERATE_KWARGS)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("summary_text", result[0])

    def test_translation_pipeline(self):
        """Test translation ORT pipeline"""
        pipe = optimum_pipeline(task="translation_en_to_de", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = "Hello, how are you?"
        result = pipe(text, **GENERATE_KWARGS)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("translation_text", result[0])

    def test_text2text_generation_pipeline(self):
        """Test text2text generation ORT pipeline"""
        pipe = optimum_pipeline(task="text2text-generation", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = "translate English to German: Hello, how are you?"
        result = pipe(text, **GENERATE_KWARGS)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("generated_text", result[0])

    def test_zero_shot_classification_pipeline(self):
        """Test zero shot classification ORT pipeline"""
        pipe = optimum_pipeline(task="zero-shot-classification", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = "This is a great movie with excellent acting."
        candidate_labels = ["positive", "negative", "neutral"]
        result = pipe(text, candidate_labels)

        self.assertIsInstance(result, dict)
        self.assertIn("labels", result)
        self.assertIn("scores", result)
        self.assertEqual(len(result["labels"]), len(candidate_labels))

    def test_image_classification_pipeline(self):
        """Test image classification ORT pipeline"""
        pipe = optimum_pipeline(task="image-classification", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        image = self._create_dummy_image()
        result = pipe(image)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("label", result[0])
        self.assertIn("score", result[0])

    def test_image_segmentation_pipeline(self):
        """Test image segmentation ORT pipeline"""
        pipe = optimum_pipeline(task="image-segmentation", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        image = self._create_dummy_image()
        result = pipe(image)

        self.assertIsInstance(result, list)
        if len(result) > 0:
            self.assertIn("label", result[0])
            self.assertIn("score", result[0])
            self.assertIn("mask", result[0])

    def test_image_to_text_pipeline(self):
        """Test image to text ORT pipeline"""
        pipe = optimum_pipeline(task="image-to-text", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        image = self._create_dummy_image()
        result = pipe(image, generate_kwargs=GENERATE_KWARGS)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("generated_text", result[0])

    def test_image_to_image_pipeline(self):
        """Test image to image ORT pipeline"""
        pipe = optimum_pipeline(task="image-to-image", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        image = self._create_dummy_image()
        result = pipe(image)

        self.assertIsInstance(result, Image.Image)

    # TODO: Enable when fixed in optimum-onnx
    # def test_automatic_speech_recognition_pipeline(self):
    #     """Test automatic speech recognition ORT pipeline"""
    #     pipe = optimum_pipeline(task="automatic-speech-recognition", accelerator="ort")
    #     audio = self._create_dummy_audio()
    #     result = pipe(audio, generate_kwargs=GENERATE_KWARGS)

    #     self.assertIsInstance(result, dict)
    #     self.assertIn("text", result)

    def test_audio_classification_pipeline(self):
        """Test audio classification ORT pipeline"""
        pipe = optimum_pipeline(task="audio-classification", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        audio = self._create_dummy_audio()
        result = pipe(audio)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("label", result[0])
        self.assertIn("score", result[0])

    def test_pipeline_with_ort_model(self):
        """Test ORT pipeline with a model already in ONNX format"""
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        model = ORTModelForFeatureExtraction.from_pretrained("distilbert-base-cased", export=True)
        pipe = optimum_pipeline(task="feature-extraction", model=model, tokenizer=tokenizer, accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = self._create_dummy_text()
        result = pipe(text)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[0][0], list)

    def test_pipeline_with_model_id(self):
        """Test ORT pipeline with a custom model id"""
        pipe = optimum_pipeline(task="feature-extraction", model="distilbert-base-cased", accelerator="ort")
        self.assertIsInstance(pipe, Pipeline)
        text = self._create_dummy_text()
        result = pipe(text)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    def test_pipeline_with_invalid_task(self):
        """Test ORT pipeline with an unsupported task"""
        with self.assertRaises(KeyError) as context:
            _ = optimum_pipeline(task="invalid-task", accelerator="ort")
        self.assertIn("Unknown task invalid-task", str(context.exception))

    def test_pipeline_with_invalid_accelerator(self):
        """Test ORT pipeline with an unsupported accelerator"""
        with self.assertRaises(ValueError) as context:
            _ = optimum_pipeline(task="feature-extraction", accelerator="invalid-accelerator")
        self.assertIn("Accelerator invalid-accelerator not recognized", str(context.exception))

    def tearDown(self):
        remove_directory(HF_HUB_CACHE)


if __name__ == "__main__":
    unittest.main()
