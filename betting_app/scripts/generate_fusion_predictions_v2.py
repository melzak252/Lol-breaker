"""Run the repository-level Fusion v2 prediction generator.

The deployment historically invoked this script from ``betting_app/scripts``.
Execute the root script by path to avoid importing this wrapper recursively.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_fusion_predictions_v2.py"
    runpy.run_path(str(script_path), run_name="__main__")
