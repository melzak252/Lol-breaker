#!/usr/bin/env python3
"""Refresh older OddsPortal CSV rows into a new provenance-bearing run."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.oddsportal_collection import refresh_main

if __name__ == "__main__":
    raise SystemExit(refresh_main())
