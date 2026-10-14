#!/usr/bin/env python3
"""Write fresh outcome-symmetric C0 EMEA diagnostics from a frozen protocol.

Usage: .venv/bin/python scripts/research/run_c0_emea_diagnostics.py --protocol PATH
No inference, fitting, acquisition, or candidate selection is performed.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.c0_emea_diagnostics import main

if __name__ == "__main__":
    raise SystemExit(main())
