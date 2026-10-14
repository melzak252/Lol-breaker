"""Create readable thesis figures for metamodel chapter 6.

The modelling experiments are intentionally not rerun here. This script reads
already generated CSV artefacts and produces compact, horizontal charts with
short labels, value annotations, and zoomed axes suitable for Obsidian/PDF.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"


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
            "font.family": "DejaVu Sans",
        }
    )


def annotate_horizontal_bars(ax: plt.Axes, decimals: int = 4) -> None:
    """Annotate horizontal bars with their numeric values.

    Args:
        ax: Matplotlib axes containing horizontal bars.
        decimals: Number of decimal places to show.
    """
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * 0.01
    for patch in ax.patches:
        width = patch.get_width()
        y_pos = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + offset,
            y_pos,
            f"{width:.{decimals}f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )


def save_horizontal_metric_chart(
    data: pd.DataFrame,
    label_column: str,
    metric_column: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ascending: bool,
    palette: str = "viridis",
    x_padding: float = 0.02,
) -> None:
    """Save a compact horizontal metric ranking chart.

    Args:
        data: Source dataframe.
        label_column: Column used as y-axis label.
        metric_column: Metric column used as bar value.
        output_path: Destination PNG path.
        title: Chart title.
        xlabel: X-axis label.
        ascending: Sort order; True means lower metric is better.
        palette: Seaborn color palette name.
        x_padding: Axis padding around min/max metric values.
    """
    plot_df = data.sort_values(metric_column, ascending=not ascending).copy()
    height = max(4.8, 0.52 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(11.5, height))
    sns.barplot(
        data=plot_df,
        x=metric_column,
        y=label_column,
        hue=label_column,
        palette=palette,
        ax=ax,
        orient="h",
        legend=False,
    )

    min_value = plot_df[metric_column].min()
    max_value = plot_df[metric_column].max()
    spread = max(max_value - min_value, 0.001)
    ax.set_xlim(min_value - spread * 0.35, max_value + spread * (1.0 + x_padding))
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    ax.grid(axis="y", visible=False)
    annotate_horizontal_bars(ax)
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_variant_figures() -> None:
    """Create readable figures for historical metamodel variants."""
    path = ASSETS / "metamodel_point6" / "metamodel_variant_metrics.csv"
    data = pd.read_csv(path)
    label_map = {
        "Stage 1 LGBM ratings": "Stage 1\nLGBM ratings",
        "Stage 2 Isotonic": "Stage 2\nIsotonic",
        "Stage 2 Platt": "Stage 2\nPlatt",
        "Player Glicko-2": "Player\nGlicko-2",
        "Simple Avg Player Ratings": "Simple avg\nplayer",
        "Simple Avg All Ratings": "Simple avg\nall",
    }
    data["label_short"] = data["model"].map(label_map).fillna(data["model"])

    save_horizontal_metric_chart(
        data=data,
        label_column="label_short",
        metric_column="logloss",
        output_path=ASSETS / "metamodel_point6" / "metamodel_variant_logloss_readable.png",
        title="Metamodel vs baseline — LogLoss (niżej lepiej)",
        xlabel="LogLoss",
        ascending=True,
        palette="mako_r",
    )
    save_horizontal_metric_chart(
        data=data,
        label_column="label_short",
        metric_column="auc",
        output_path=ASSETS / "metamodel_point6" / "metamodel_variant_auc_readable.png",
        title="Metamodel vs baseline — AUC (wyżej lepiej)",
        xlabel="AUC",
        ascending=False,
        palette="viridis",
    )


def create_best_config_figures() -> None:
    """Create readable figures for combined best configuration search."""
    path = ASSETS / "metamodel_best_config_point6" / "best_metamodel_config_results.csv"
    data = pd.read_csv(path)
    data = data.head(8).copy()
    feature_map = {
        "optuna_player_full_context_w50": "Full W50",
        "optuna_player_core_context_w50": "Core W50",
        "optuna_player_uncertainty": "Player only",
    }
    data["label_short"] = data.apply(
        lambda row: (
            f"{feature_map.get(row['feature_set'], row['feature_set'])} | "
            f"upd {int(row['update_interval'])} | mask {row['mask_rate']:.1f}"
        ),
        axis=1,
    )

    save_horizontal_metric_chart(
        data=data,
        label_column="label_short",
        metric_column="logloss",
        output_path=ASSETS
        / "metamodel_best_config_point6"
        / "best_metamodel_config_logloss_readable.png",
        title="Łączny test najlepszych konfiguracji — LogLoss",
        xlabel="LogLoss",
        ascending=True,
        palette="crest_r",
    )
    save_horizontal_metric_chart(
        data=data,
        label_column="label_short",
        metric_column="auc",
        output_path=ASSETS
        / "metamodel_best_config_point6"
        / "best_metamodel_config_auc_readable.png",
        title="Łączny test najlepszych konfiguracji — AUC",
        xlabel="AUC",
        ascending=False,
        palette="crest",
    )


def create_ablation_overview() -> None:
    """Create a single readable overview of the best ablation per group."""
    path = ASSETS / "metamodel_experiments_point6" / "metamodel_ablation_results.csv"
    data = pd.read_csv(path)
    idx = data.groupby("experiment_group")["logloss"].idxmin()
    best = data.loc[idx].copy()
    group_labels = {
        "player_vs_team_features": "Player vs team",
        "context_feature_set": "Zakres kontekstu",
        "update_interval": "Update window",
        "masking_rate": "Masking",
        "context_window": "Context window",
    }
    best["label_short"] = best.apply(
        lambda row: f"{group_labels.get(row['experiment_group'], row['experiment_group'])}: {row['variant']}",
        axis=1,
    )

    save_horizontal_metric_chart(
        data=best,
        label_column="label_short",
        metric_column="logloss",
        output_path=ASSETS
        / "metamodel_experiments_point6"
        / "metamodel_ablation_logloss_overview.png",
        title="Najlepsze warianty w eksperymentach ablation — LogLoss",
        xlabel="LogLoss",
        ascending=True,
        palette="flare_r",
    )


def main() -> None:
    """Generate all readable chapter 6 figures."""
    configure_style()
    create_variant_figures()
    create_best_config_figures()
    create_ablation_overview()
    print("Readable metamodel figures generated.")


if __name__ == "__main__":
    main()
