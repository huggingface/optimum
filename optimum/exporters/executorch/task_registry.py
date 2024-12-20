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

task_registry = {}

package_name = "optimum.exporters.executorch.tasks"


def register_task(task_name):
    """
    Decorator to register a task under a specific name.

    Args:
        task_name (`str`):
            The name of the task to associate with a callable task.

    Returns:
        `Callable`:
            The original function wrapped as a registered task.

    Example:
        ```python
        @register_task("my_new_task")
        def my_new_task(...):
            ...
        ```
    """

    def decorator(func):
        task_registry[task_name] = func
        return func

    return decorator


def discover_tasks():
    """
    Dynamically discovers and imports all task modules within the `optimum.exporters.executorch.tasks` package.

    Ensures tasks under `./tasks` directory are dynamically loaded without requiring manual imports.

    Notes:
        New tasks **must** be added to the `./tasks` directory to be discovered and used by `main_export`.
        Failure to do so will prevent dynamic discovery and registration. Tasks must also use the
        `@register_task` decorator to be properly registered in the `task_registry`.
    """
    package = importlib.import_module(package_name)
    package_path = package.__path__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        logger.info(f"Importing {package_name}.{module_name}")
        importlib.import_module(f"{package_name}.{module_name}")
