"""Freeze or run the Plan07 family map-bridge reliability subprotocol."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.family_reliability import main


if __name__ == "__main__":
    raise SystemExit(main())
