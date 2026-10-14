"""Helpers for importing experiment scripts as runtime modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module_from_path(path: str | Path, module_name: str) -> object:
    """Load a Python file as an importable module.

    Args:
        path: Python file path.
        module_name: Runtime module name to register in ``sys.modules``.

    Returns:
        Loaded Python module object.

    Raises:
        ImportError: If the module spec or loader cannot be created.
    """

    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
