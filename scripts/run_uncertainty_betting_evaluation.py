#!/usr/bin/env python3
"""Compatibility entry point for the shared frozen approach-comparison runner."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ev_approach_runner import approach_comparison_main


def main() -> None:
    """Run the same frozen approach-comparison CLI as the unified evaluator."""
    approach_comparison_main()


if __name__ == "__main__":
    main()
