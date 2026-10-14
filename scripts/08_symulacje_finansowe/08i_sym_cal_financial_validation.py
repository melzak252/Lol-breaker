"""Financial validation runner for the final Sym-Cal model.

This entrypoint reuses the current financial validation suite, but switches the
input probability stream from the raw W20-Binomial logistic model to the final
order-symmetrized and Platt-calibrated model used in the thesis.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINANCIAL_SUITE = Path(__file__).with_name("08_financial_validation_suite.py")
INPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "final_symmetric_calibrated_market_comparison"
    / "final_symmetric_calibrated_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "financial_sym_cal_point8"
MODEL_PROBABILITY = "prob_sym_cal_lr_elasticnet_w20_binomial"


def load_financial_suite() -> object:
    """Load the reusable financial validation suite module.

    Returns:
        Imported financial validation module.

    Raises:
        RuntimeError: If the suite module cannot be loaded.
    """

    spec = importlib.util.spec_from_file_location("financial_suite", FINANCIAL_SUITE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load financial suite from {FINANCIAL_SUITE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_sym_cal_candidates(suite: object) -> list[object]:
    """Build model, market, and hybrid candidates for the final model.

    Args:
        suite: Loaded financial suite module exposing the ``Candidate`` class.

    Returns:
        Candidate configurations evaluated by the financial suite.
    """

    candidates = [
        suite.Candidate("Market Open", "market"),
        suite.Candidate("Sym-Cal LR-ElasticNet-W20-Binomial", "model", temperature=1.0),
        suite.Candidate("Sym-Cal LR-ElasticNet T=0.80", "model", temperature=0.8),
        suite.Candidate("Sym-Cal LR-ElasticNet T=0.90", "model", temperature=0.9),
        suite.Candidate("Sym-Cal LR-ElasticNet T=1.10", "model", temperature=1.1),
        suite.Candidate("Sym-Cal LR-ElasticNet T=1.20", "model", temperature=1.2),
    ]

    for temperature in (0.8, 0.9, 1.0):
        for alpha in np.round(np.arange(0.1, 1.0, 0.1), 2):
            if temperature == 1.0:
                name = f"Sym-Cal Hybrid a={alpha:.2f}"
            else:
                name = f"Sym-Cal Hybrid a={alpha:.2f} T={temperature:.2f}"
            candidates.append(
                suite.Candidate(
                    name=name,
                    source="hybrid",
                    alpha=float(alpha),
                    temperature=float(temperature),
                )
            )

    return candidates


def main() -> None:
    """Run financial validation on the final Sym-Cal probability stream."""

    suite = load_financial_suite()
    suite.INPUT_PATH = INPUT_PATH
    suite.OUTPUT_DIR = OUTPUT_DIR
    suite.MODEL_PROBABILITY = MODEL_PROBABILITY
    suite.build_candidates = lambda: build_sym_cal_candidates(suite)
    suite.main()


if __name__ == "__main__":
    main()
