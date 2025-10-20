# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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


CONFIG_NAME = "config.json"

DIFFUSION_MODEL_UNET_SUBFOLDER = "unet"
DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER = "transformer"
DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER = "vae_decoder"
DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER = "vae_encoder"
DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER = "text_encoder"
DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER = "text_encoder_2"
DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER = "text_encoder_3"
DIFFUSION_PIPELINE_CONFIG_FILE_NAME = "model_index.json"
DIFFUSION_MODEL_CONFIG_FILE_NAME = "config.json"

ONNX_WEIGHTS_NAME = "model.onnx"  # TODO: remove and use the optimum-onnx one

ALL_TASKS = [
    "audio-classification",
    "audio-frame-classification",
    "audio-xvector",
    "automatic-speech-recognition",
    "depth-estimation",
    "document-question-answering",
    "feature-extraction",
    "fill-mask",
    "image-classification",
    "image-segmentation",
    "image-text-to-text",
    "image-to-image",
    "image-to-text",
    "inpainting",
    "keypoint-detection",
    "mask-generation",
    "masked-im",
    "multiple-choice",
    "object-detection",
    "question-answering",
    "reinforcement-learning",
    "semantic-segmentation",
    "sentence-similarity",
    "text-classification",
    "text-generation",
    "text-to-audio",
    "text-to-image",
    "text2text-generation",
    "time-series-forecasting",
    "token-classification",
    "visual-question-answering",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
]
