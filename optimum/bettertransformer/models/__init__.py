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
import warnings

from .encoder_models import (
    AlbertLayerBetterTransformer,
    BartEncoderLayerBetterTransformer,
    BertLayerBetterTransformer,
    DistilBertLayerBetterTransformer,
    FSMTEncoderLayerBetterTransformer,
    ViltLayerBetterTransformer,
    ViTLayerBetterTransformer,
    Wav2Vec2EncoderLayerBetterTransformer,
    WhisperEncoderLayerBetterTransformer,
    TapasLayerBetterTransformer,

)


BETTER_TRANFORMER_LAYERS_MAPPING_DICT = {
    # Bert Family
    "BertLayer": BertLayerBetterTransformer,
    "ElectraLayer": BertLayerBetterTransformer,
    "Data2VecTextLayer": BertLayerBetterTransformer,
    "CamembertLayer": BertLayerBetterTransformer,
    "MarkupLMLayer": BertLayerBetterTransformer,
    "RobertaLayer": BertLayerBetterTransformer,
    "SplinterLayer": BertLayerBetterTransformer,
    "ErnieLayer": BertLayerBetterTransformer,
    "LayoutLMLayer": BertLayerBetterTransformer,
    "BertGenerationLayer": BertLayerBetterTransformer,
    "XLMRobertaLayer": BertLayerBetterTransformer,
    # Albert Family
    "AlbertLayer": AlbertLayerBetterTransformer,
    # Bart family
    "BartEncoderLayer": BartEncoderLayerBetterTransformer,
    # "PLBartEncoderLayer": bart.BartEncoderLayerBetterTransformer,
    # "MarianEncoderLayer": bart.BartEncoderLayerBetterTransformer,
    # "TimeSeriesTransformerEncoderLayer": bart.BartEncoderLayerBetterTransformer,
    # "BlenderbotSmallEncoderLayer": bart.BartEncoderLayerBetterTransformer,
    # T5 family - needs to check compatibility first
    # "T5Block": t5.T5LayerBetterTransformer,
    # Some models cannot be tested such as:
    # "QDQBertLayer": BertLayerBetterTransformer, --> needs torch quantization
    # "RealmLayer": BertLayerBetterTransformer, --> not mapped in AutoModel
    # DistilBert:
    "TransformerBlock": DistilBertLayerBetterTransformer,
    # WhisperModel
    "WhisperEncoderLayer": WhisperEncoderLayerBetterTransformer,
    # Wav2vec2 family:
    "Wav2Vec2EncoderLayer": Wav2Vec2EncoderLayerBetterTransformer,
    "HubertEncoderLayer": Wav2Vec2EncoderLayerBetterTransformer,
    # "UniSpeechEncoderLayer": Wav2Vec2EncoderLayerBetterTransformer,
    # "Data2VecAudioEncoderLayer": Wav2Vec2EncoderLayerBetterTransformer,
    # ViT Family:
    "ViTLayer": ViTLayerBetterTransformer,
    "DeiTLayer": ViTLayerBetterTransformer,
    "ViTMAELayer": ViTLayerBetterTransformer,
    "ViTMSNLayer": ViTLayerBetterTransformer,
    "YolosLayer": ViTLayerBetterTransformer,
    # FSMTModel:
    "EncoderLayer": FSMTEncoderLayerBetterTransformer,
    "ViltLayer": ViltLayerBetterTransformer,

    # Tapas Model
    "TapasLayer": TapasLayerBetterTransformer,
}


class warn_uncompatible_save(object):
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "You are calling `save_pretrained` to a `BetterTransformer` converted model you may likely encounter unexepected behaviors. ",
            UserWarning,
        )
        return self.callback(*args, **kwargs)
