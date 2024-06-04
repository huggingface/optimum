import logging
import sys


if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata
from importlib.util import find_spec, module_from_spec


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


def load_backends():
    """Load optimum backends

    This method goes through packages inside the `optimum` namespace and loads the `backend` module if it exists.

    This module is then in charge of registering the backend commands.
    """
    OPTIMUM_NAMESPACE = "optimum"
    OPTIMUM_BACKEND_MODULE = "backend"
    return load_namespace_modules(OPTIMUM_NAMESPACE, OPTIMUM_BACKEND_MODULE)
