"""Compatibility helper for legacy financial simulation entrypoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def run_current_financial_suite() -> None:
    """Load and run the current LR-W20-Binomial financial suite.

    The financial scripts in this directory historically had several standalone
    entrypoints. They now share the current implementation in
    ``08_financial_validation_suite.py`` so old commands remain executable
    without duplicating logic.
    """

    suite_path = Path(__file__).with_name("08_financial_validation_suite.py")
    spec = importlib.util.spec_from_file_location("current_financial_suite", suite_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load financial suite from {suite_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()
