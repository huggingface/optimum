# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import importlib
import logging
import pkgutil


logger = logging.getLogger(__name__)

recipe_registry = {}

package_name = "optimum.exporters.executorch.recipes"


def register_recipe(recipe_name):
    """
    Decorator to register a recipe for exporting and lowering an ExecuTorch model under a specific name.

    Args:
        recipe_name (`str`):
            The name of the recipe to associate with a callable recipe.

    Returns:
        `Callable`:
            The original function wrapped as a registered recipe.

    Example:
        ```python
        @register_recipe("my_new_recipe")
        def my_new_recipe(...):
            ...
        ```
    """

    def decorator(func):
        recipe_registry[recipe_name] = func
        return func

    return decorator


def discover_recipes():
    """
    Dynamically discovers and imports all recipe modules within the `optimum.exporters.executorch.recipes` package.

    Ensures recipes under `./recipes` directory are dynamically loaded without requiring manual imports.

    Notes:
        New recipes **must** be added to the `./recipes` directory to be discovered and used by `main_export`.
        Failure to do so will prevent dynamic discovery and registration. Recipes must also use the
        `@register_recipe` decorator to be properly registered in the `recipe_registry`.
    """
    package = importlib.import_module(package_name)
    package_path = package.__path__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        logger.info(f"Importing {package_name}.{module_name}")
        importlib.import_module(f"{package_name}.{module_name}")
