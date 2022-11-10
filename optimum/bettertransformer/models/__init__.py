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
    WhisperEncoderLayerBetterTransformer,
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
    # TODO: add Wav2vec2
}


def is_supporting_bettertransformer(module_name):
    r"""
    An utility function that checks if the input module is compatible with its `BetterTransformer`
    implementation.

    Args:
        module_name, (`str`, **required**):
            Input module_name
    Returns:
        The corresponding `torch.nn.Module` of the `BetterTransformer` layer, or `None`
        if the `module_name` is not in the list of supported modules.
    """
    if module_name in BETTER_TRANFORMER_LAYERS_MAPPING_DICT.keys():
        return BETTER_TRANFORMER_LAYERS_MAPPING_DICT[module_name]
    else:
        return None


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
