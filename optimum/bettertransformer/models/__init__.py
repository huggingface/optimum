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
from . import bart, bert, distilbert


FAST_LAYERS_MAPPING_DICT = {
    # Bert Family
    "BertLayer": bert.BertLayerFast,
    "ElectraLayer": bert.BertLayerFast,
    "Data2VecTextLayer": bert.BertLayerFast,
    "CamembertLayer": bert.BertLayerFast,
    "MarkupLMLayer": bert.BertLayerFast,
    "RobertaLayer": bert.BertLayerFast,
    "SplinterLayer": bert.BertLayerFast,
    "ErnieLayer": bert.BertLayerFast,
    "LayoutLMLayer": bert.BertLayerFast,
    "BertGenerationLayer": bert.BertLayerFast,
    "RobertaLayer": bert.BertLayerFast,
    "Data2VecTextLayer": bert.BertLayerFast,
    "XLMRobertaLayer": bert.BertLayerFast,
    # Bart family - need to tweak the tests a bit
    "BartEncoderLayer": bart.BartLayerFast,
    # "PLBartEncoderLayer": bart.BartLayerFast,
    # "MarianEncoderLayer": bart.BartLayerFast,
    # "TimeSeriesTransformerEncoderLayer": bart.BartLayerFast,
    # "BlenderbotSmallEncoderLayer": bart.BartLayerFast,
    # T5 family - needs to check compatibility first
    # "T5Block": t5.T5LayerFast,
    # Some models cannot be tested such as:
    # "QDQBertLayer": bert.BertLayerFast, --> needs torch quantization
    # "RealmLayer": bert.BertLayerFast, --> not mapped in AutoModel
    "TransformerBlock": distilbert.DistilBertLayerFast,
}


def is_module_fast(module_name):
    if module_name not in FAST_LAYERS_MAPPING_DICT.keys():
        return False
    else:
        return FAST_LAYERS_MAPPING_DICT[module_name]
