"""Generate polished financial visualizations for whitepaper chapter 8.

The script reads already generated outputs from ``08_financial_validation_suite.py``
and creates thesis-friendly plots without rerunning simulations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT_DIR / "docs" / "assets" / "financial_point8"


def configure_style() -> None:
    """Configure a consistent visual style for report figures."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def save_figure(path: Path) -> None:
    """Save current matplotlib figure and close it.

    Args:
        path: Output path for the figure.
    """
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", transparent=False)
    plt.close()


def add_bar_labels(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    """Annotate horizontal bars with their values.

    Args:
        ax: Axes containing bar containers.
        fmt: Format string for labels.
    """
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=9)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load financial and probability summary data.

    Returns:
        Tuple with financial summary and probability metrics.
    """
    summary = pd.read_csv(ASSET_DIR / "financial_validation_summary.csv")
    probability = pd.read_csv(ASSET_DIR / "financial_validation_probability_metrics.csv")
    return summary, probability


def plot_probability_quality(probability: pd.DataFrame) -> None:
    """Plot 2024+ LogLoss ranking for model candidates.

    Args:
        probability: Probability metrics table.
    """
    data = probability[probability["scope"] == "2024+"].copy()
    data = data.sort_values("logloss", ascending=True)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=data,
        x="logloss",
        y="candidate",
        hue="candidate",
        palette="viridis",
        legend=False,
    )
    ax.set_title("Jakość probabilistyczna 2024+ — niższy LogLoss jest lepszy", weight="bold")
    ax.set_xlabel("LogLoss")
    ax.set_ylabel("")
    ax.set_xlim(data["logloss"].min() - 0.005, data["logloss"].max() + 0.005)
    add_bar_labels(ax, "{:.4f}")
    save_figure(ASSET_DIR / "financial_probability_logloss_2024.png")


def plot_fixed_percent(summary: pd.DataFrame) -> None:
    """Plot fixed-percent ROI and MaxDD for 2024+.

    Args:
        summary: Financial summary table.
    """
    data = summary[
        (summary["scope"] == "2024+")
        & (summary["staking_policy"] == "Fixed percent 2% min2 max100")
    ].copy()
    data = data.sort_values("roi_pct", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    sns.barplot(
        data=data,
        x="roi_pct",
        y="candidate",
        hue="candidate",
        palette="mako",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Fixed percent — ROI 2024+", weight="bold")
    axes[0].set_xlabel("ROI [%]")
    axes[0].set_ylabel("")
    add_bar_labels(axes[0], "{:.0f}")

    sns.barplot(
        data=data,
        x="max_drawdown_pct",
        y="candidate",
        hue="candidate",
        palette="rocket",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Fixed percent — Max Drawdown 2024+", weight="bold")
    axes[1].set_xlabel("MaxDD [%]")
    axes[1].set_ylabel("")
    add_bar_labels(axes[1], "{:.1f}")
    save_figure(ASSET_DIR / "financial_fixed_percent_roi_drawdown_2024.png")


def plot_kelly_tradeoff(summary: pd.DataFrame) -> None:
    """Plot Kelly risk-return scatter for 2024+.

    Args:
        summary: Financial summary table.
    """
    data = summary[
        (summary["scope"] == "2024+")
        & (summary["staking_policy"].str.contains("Kelly", regex=False))
    ].copy()
    data["bets_size"] = data["bets"].clip(lower=1) / data["bets"].max() * 650

    plt.figure(figsize=(11, 7))
    ax = sns.scatterplot(
        data=data,
        x="max_drawdown_pct",
        y="yield_pct",
        hue="candidate",
        size="bets_size",
        sizes=(80, 650),
        alpha=0.8,
        palette="tab10",
        legend="brief",
    )
    ax.set_title("Kelly 2024+ — kompromis Yield vs Max Drawdown", weight="bold")
    ax.set_xlabel("Max Drawdown [%]")
    ax.set_ylabel("Yield [%]")
    ax.axvline(50, color="crimson", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(50.5, data["yield_pct"].min(), "próg 50% MaxDD", color="crimson", fontsize=9)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), title="Kandydat")
    save_figure(ASSET_DIR / "financial_kelly_yield_drawdown_scatter_2024.png")


def plot_stability(summary: pd.DataFrame) -> None:
    """Plot 2021+ vs 2024+ stability for selected hybrid Kelly variants.

    Args:
        summary: Financial summary table.
    """
    selected_candidates = ["Hybrid a=0.48 T=0.60", "Hybrid a=0.48 T=0.70"]
    selected_staking = ["Kelly 0.25 min2 max100", "Kelly 0.50 min2 max100"]
    data = summary[
        summary["candidate"].isin(selected_candidates)
        & summary["staking_policy"].isin(selected_staking)
    ].copy()
    data["variant"] = data["candidate"] + " | " + data["staking_policy"].str.replace(" min2 max100", "", regex=False)

    plt.figure(figsize=(13, 6))
    ax = sns.barplot(
        data=data,
        x="variant",
        y="yield_pct",
        hue="scope",
        palette="Set2",
    )
    ax.set_title("Stabilność najlepszych hybryd: 2021+ vs 2024+", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Yield [%]")
    ax.tick_params(axis="x", rotation=25)
    add_bar_labels(ax, "{:.1f}")
    save_figure(ASSET_DIR / "financial_stability_2021_vs_2024.png")


def plot_clv(summary: pd.DataFrame) -> None:
    """Plot average CLV for 2024+ candidates under fixed stake.

    Args:
        summary: Financial summary table.
    """
    data = summary[
        (summary["scope"] == "2024+")
        & (summary["staking_policy"] == "Fixed stake 10")
    ].copy()
    data = data.sort_values("avg_clv_pct", ascending=True)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=data,
        x="avg_clv_pct",
        y="candidate",
        hue="candidate",
        palette="crest",
        legend=False,
    )
    ax.set_title("CLV 2024+ — czy strategia bije kurs zamknięcia?", weight="bold")
    ax.set_xlabel("Średni CLV [%]")
    ax.set_ylabel("")
    ax.axvline(0, color="black", linewidth=1)
    add_bar_labels(ax, "{:.1f}")
    save_figure(ASSET_DIR / "financial_clv_2024_fixedstake.png")


def main() -> None:
    """Generate all polished financial figures."""
    configure_style()
    summary, probability = load_data()
    plot_probability_quality(probability)
    plot_fixed_percent(summary)
    plot_kelly_tradeoff(summary)
    plot_stability(summary)
    plot_clv(summary)
    print("Generated polished financial figures in", ASSET_DIR)


if __name__ == "__main__":
    main()
