"""Generate iso-parameter ROI and MaxDD curves from cached grid results.

This script does not rerun the financial simulation. It reads the dense grid
created by ``08e_dense_alpha_kelly_temperature_grid.py`` and creates three
dual-axis plots:

1. isotherm: ROI and MaxDD as a function of alpha at fixed T and Kelly,
2. iso-alpha: ROI and MaxDD as a function of Kelly at fixed alpha and T,
3. iso-Kelly: ROI and MaxDD as a function of T at fixed alpha and Kelly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.thesis_style import (
    DARK_TEXT,
    PASTEL_BLUE,
    PASTEL_ORANGE,
    apply_thesis_style,
    clean_axis,
)


OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "financial_point8"
DENSE_GRID_PATH = OUTPUT_DIR / "financial_dense_alpha_kelly_temperature_grid.csv"

# Risk-controlled reference point selected from the dense grid.
REFERENCE_ALPHA = 0.30
REFERENCE_TEMPERATURE = 0.60
REFERENCE_KELLY = 0.10


def load_dense_grid() -> pd.DataFrame:
    """Load the cached dense financial grid."""

    if not DENSE_GRID_PATH.exists():
        raise FileNotFoundError(
            f"Missing dense grid at {DENSE_GRID_PATH}. Run "
            "scripts/08_symulacje_finansowe/08e_dense_alpha_kelly_temperature_grid.py first."
        )
    return pd.read_csv(DENSE_GRID_PATH)


def save_dual_axis_plot(
    data: pd.DataFrame,
    x_column: str,
    x_label: str,
    title: str,
    output_name: str,
) -> None:
    """Save a dual-axis ROI and MaxDD curve.

    Args:
        data: Filtered grid data for a single iso-parameter slice.
        x_column: Column used as the x-axis.
        x_label: Human-readable x-axis label.
        title: Plot title.
        output_name: Output PNG filename.
    """

    apply_thesis_style(context="paper")
    ordered = data.sort_values(x_column)

    fig, ax_roi = plt.subplots(figsize=(8.6, 5.0))
    ax_dd = ax_roi.twinx()

    roi_line = ax_roi.plot(
        ordered[x_column],
        ordered["roi_pct"],
        color=PASTEL_BLUE,
        marker="o",
        linewidth=2.2,
        label="ROI",
    )
    dd_line = ax_dd.plot(
        ordered[x_column],
        ordered["max_drawdown_pct"],
        color=PASTEL_ORANGE,
        marker="s",
        linewidth=2.2,
        label="MaxDD",
    )

    ax_dd.axhline(25, color=DARK_TEXT, linestyle="--", linewidth=1.1, alpha=0.7)
    ax_dd.text(
        ordered[x_column].min(),
        25.8,
        "próg 25% MaxDD",
        color=DARK_TEXT,
        fontsize=9,
    )

    ax_roi.set_title(title)
    ax_roi.set_xlabel(x_label)
    ax_roi.set_ylabel("ROI [%]", color=PASTEL_BLUE)
    ax_dd.set_ylabel("MaxDD [%]", color=PASTEL_ORANGE)
    ax_roi.tick_params(axis="y", labelcolor=PASTEL_BLUE)
    ax_dd.tick_params(axis="y", labelcolor=PASTEL_ORANGE)

    clean_axis(ax_roi)
    ax_dd.spines["top"].set_visible(False)
    ax_dd.grid(False)

    lines = roi_line + dd_line
    labels = [line.get_label() for line in lines]
    ax_roi.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name)
    plt.close(fig)


def build_iso_slices(grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build isotherm, iso-alpha and iso-Kelly slices from the dense grid."""

    isotherm = grid[
        (grid["temperature"] == REFERENCE_TEMPERATURE)
        & (grid["kelly"] == REFERENCE_KELLY)
    ].copy()
    isoalpha = grid[
        (grid["alpha"] == REFERENCE_ALPHA)
        & (grid["temperature"] == REFERENCE_TEMPERATURE)
    ].copy()
    isokelly = grid[
        (grid["alpha"] == REFERENCE_ALPHA)
        & (grid["kelly"] == REFERENCE_KELLY)
    ].copy()
    return {
        "isotherm": isotherm,
        "isoalpha": isoalpha,
        "isokelly": isokelly,
    }


def main() -> None:
    """Generate all iso-parameter curves and save their data."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = load_dense_grid()
    slices = build_iso_slices(grid)

    slices["isotherm"].to_csv(OUTPUT_DIR / "financial_isotherm_T0.60_K0.10.csv", index=False)
    slices["isoalpha"].to_csv(OUTPUT_DIR / "financial_isoalpha_a0.30_T0.60.csv", index=False)
    slices["isokelly"].to_csv(OUTPUT_DIR / "financial_isokelly_a0.30_K0.10.csv", index=False)

    save_dual_axis_plot(
        slices["isotherm"],
        "alpha",
        "Alfa modelu w hybrydzie",
        f"Izoterma T={REFERENCE_TEMPERATURE:.2f}, Kelly={REFERENCE_KELLY:.2f}",
        "financial_isotherm_roi_maxdd.png",
    )
    save_dual_axis_plot(
        slices["isoalpha"],
        "kelly",
        "Współczynnik Kelly'ego",
        f"Izoalfa α={REFERENCE_ALPHA:.2f}, T={REFERENCE_TEMPERATURE:.2f}",
        "financial_isoalpha_roi_maxdd.png",
    )
    save_dual_axis_plot(
        slices["isokelly"],
        "temperature",
        "Temperatura T",
        f"Izokelly Kelly={REFERENCE_KELLY:.2f}, α={REFERENCE_ALPHA:.2f}",
        "financial_isokelly_roi_maxdd.png",
    )

    print("Saved iso-parameter ROI/MaxDD figures:")
    print(f"- {OUTPUT_DIR / 'financial_isotherm_roi_maxdd.png'}")
    print(f"- {OUTPUT_DIR / 'financial_isoalpha_roi_maxdd.png'}")
    print(f"- {OUTPUT_DIR / 'financial_isokelly_roi_maxdd.png'}")


if __name__ == "__main__":
    main()
