#!/usr/bin/env python
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import importlib
import logging
import sys


if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata
from importlib.util import find_spec, module_from_spec

from .utils import is_onnxruntime_available


logger = logging.getLogger(__name__)


def load_namespace_modules(namespace: str, module: str):
    """Load modules with a specific name inside a namespace

    This method operates on namespace packages:
    https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

    For each package inside the specified `namespace`, it looks for the specified `module` and loads it.

    Args:
        namespace (`str`):
            The namespace containing modules to be loaded.
        module (`str`):
            The name of the module to load in each namespace package.
    """
    for dist in importlib_metadata.distributions():
        dist_name = dist.metadata["Name"]
        if dist_name is None:
            continue
        if dist_name == f"{namespace}-benchmark":
            continue
        if not dist_name.startswith(f"{namespace}-"):
            continue
        package_import_name = dist_name.replace("-", ".")
        module_import_name = f"{package_import_name}.{module}"
        if module_import_name in sys.modules:
            # Module already loaded
            continue
        backend_spec = find_spec(module_import_name)
        if backend_spec is None:
            continue
        try:
            imported_module = module_from_spec(backend_spec)
            sys.modules[module_import_name] = imported_module
            backend_spec.loader.exec_module(imported_module)
            logger.debug(f"Successfully loaded {module_import_name}")
        except Exception as e:
            logger.error(f"An exception occured while loading {module_import_name}: {e}.")


def load_subpackages():
    """Load optimum subpackages

    This method goes through packages inside the `optimum` namespace and loads the `subpackage` module if it exists.

    This module is then in charge of registering the subpackage commands.
    """
    SUBPACKAGE_LOADER = "subpackage"
    load_namespace_modules("optimum", SUBPACKAGE_LOADER)

    # Load subpackages from internal modules not explicitly defined as namespace packages
    loader_name = "." + SUBPACKAGE_LOADER
    if is_onnxruntime_available():
        importlib.import_module(loader_name, package="optimum.onnxruntime")
