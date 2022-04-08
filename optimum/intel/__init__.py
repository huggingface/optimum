#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

from .neural_compressor.configuration import IncConfig, IncPruningConfig, IncQuantizationConfig
from .neural_compressor.optimization import IncOptimizer
from .neural_compressor.pruning import IncPruner, IncPruningMode
from .neural_compressor.quantization import IncQuantizationMode, IncQuantizer
from .neural_compressor.trainer import IncTrainer
