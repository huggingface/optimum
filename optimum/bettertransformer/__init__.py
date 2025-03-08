# Copyright 2022 The HuggingFace and Meta Team.  All rights reserved.
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


from ..utils.import_utils import _transformers_version, is_transformers_version


if is_transformers_version(">=", "4.49"):
    raise RuntimeError(
        f"BetterTransformer requires transformers<4.49 but found {_transformers_version}. "
        "`optimum.bettertransformer` is deprecated and will be removed in optimum v2.0."
    )

from .models import BetterTransformerManager
from .transformation import BetterTransformer
