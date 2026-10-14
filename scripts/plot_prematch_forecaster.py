#!/usr/bin/env python3
"""Render data coverage and stored forecast results; never fit or select models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#1f618d", "#148f77", "#b9770e", "#884ea0"]


def dataset_overview(protocol: dict) -> None:
    feature_dir, report_dir = Path(protocol["feature_dir"]), Path(protocol["report_dir"])
    metadata = pd.read_parquet(feature_dir / "metadata.parquet")
    audit = json.loads((feature_dir / "audit.json").read_text())
    names = audit["feature_names"]
    with np.load(feature_dir / "arrays.npz") as archive:
        appearances = np.rint(np.expm1(archive["players"][..., names["player"].index("log_appearance_count")]))
        team = archive["teams"]
    minimum_history = appearances.min(axis=(1, 2))
    known = (team[..., names["team"].index("prior_lineup_known")] == 1).all(axis=1)
    unchanged = known & (team[..., names["team"].index("prior_lineup_overlap_fraction")] == 1).all(axis=1)
    year = metadata.date.astype(str).str[:4]
    cohorts = pd.crosstab(year, metadata.best_of)
    support = pd.cut(minimum_history, [-.5, .5, 10.5, 50.5, np.inf], labels=["No recorded history", "Minimum 1-10 maps", "Minimum 11-50 maps", "All ten >50 maps"])
    support_share = pd.crosstab(year, support, normalize="index", dropna=False) * 100
    roster_category = np.where(~known, "Prior lineup unavailable", np.where(unchanged, "Both lineups unchanged", "Prior known; lineup changed"))
    roster_share = pd.crosstab(year, roster_category, normalize="index") * 100
    missing = pd.Series({field: 100 * values["missing"] / values["seen"] for field, values in audit["stat_missingness"].items()}).sort_values()
    overview = pd.DataFrame({"year": year, "minimum_history_maps": minimum_history, "has_cold_player": minimum_history == 0, "both_prior_lineups_known": known, "both_lineups_unchanged": unchanged}).groupby("year").agg(targets=("minimum_history_maps", "size"), median_minimum_history=("minimum_history_maps", "median"), fraction_with_cold_player=("has_cold_player", "mean"), fraction_with_both_prior_lineups=("both_prior_lineups_known", "mean"), fraction_with_both_lineups_unchanged=("both_lineups_unchanged", "mean"))
    overview.to_csv(report_dir / "data_profiles.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    cohorts.plot.bar(stacked=True, ax=axes[0, 0], color=COLORS[:3], rot=45)
    axes[0, 0].set(title="Binary series targets by recorded year", xlabel="", ylabel="Series")
    axes[0, 0].legend(title="Best of", frameon=False)
    axes[0, 1].barh(missing.index, missing.values, color=COLORS[0])
    for position, value in enumerate(missing):
        axes[0, 1].text(value + .3, position, f"{value:.2f}%", va="center", fontsize=9)
    axes[0, 1].set(title="Missing past-map performance fields", xlabel="Missing among all player-map records (%)", xlim=(0, missing.max() + 5))
    support_share.plot.bar(stacked=True, ax=axes[1, 0], color=COLORS, rot=45)
    axes[1, 0].set(title="Pre-target player-history support", xlabel="", ylabel="Targets (%)", ylim=(0, 100))
    axes[1, 0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -.22), ncol=2)
    roster_share.plot.bar(stacked=True, ax=axes[1, 1], color=COLORS[:3], rot=45)
    axes[1, 1].set(title="Realized first-map versus previous released lineup\nPlayer membership only; role order can differ", xlabel="", ylabel="Targets (%)", ylim=(0, 100))
    axes[1, 1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -.22))
    fig.suptitle(f"LoL source corpus: {len(metadata):,} binary targets; all {audit['exact_counts']['maps']:,} maps supply history\nCoverage through {metadata.date.max()}; roster availability is a scenario, not announcement evidence", fontsize=13)
    fig.savefig(report_dir / "dataset_overview.png", dpi=160)
    plt.close(fig)


def result_figures(protocol: dict) -> None:
    report_dir = Path(protocol["report_dir"])
    development = pd.read_csv(report_dir / "development_metrics.csv").sort_values("log_loss", ascending=False)
    selection = json.loads((report_dir / "selection.json").read_text())
    summary = json.loads((report_dir / "final_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    positions = np.arange(len(development))
    colors = [COLORS[1] if name == selection["selected"] else COLORS[0] for name in development.model]
    ax.scatter(development.log_loss, positions, c=colors, s=45)
    ax.set_yticks(positions, development.model)
    for position, value in zip(positions, development.log_loss):
        ax.annotate(f"{value:.5f}", (value, position), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
    width = development.log_loss.max() - development.log_loss.min()
    ax.set_xlim(development.log_loss.min() - .05 * max(width, .01), development.log_loss.max() + .2 * max(width, .01))
    ax.set(title=f"Development selection: {selection['selection_n']:,} common series\n{selection['selection_period'][0]}–{selection['selection_period'][1]}; lower log loss is better", xlabel="Pooled development log loss")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(report_dir / "development_comparison.png", dpi=160)
    plt.close(fig)

    calibration = pd.read_csv(report_dir / "final_calibration.csv")
    calibration = calibration.loc[calibration.matches.gt(0)]
    monthly = pd.read_csv(report_dir / "final_monthly_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    axes[0].plot(calibration.forecast, calibration.observed, color=COLORS[0], linewidth=1)
    axes[0].scatter(calibration.forecast, calibration.observed, s=20 + 2 * np.sqrt(calibration.matches), color=COLORS[0])
    for row in calibration.itertuples():
        axes[0].annotate(str(row.matches), (row.forecast, row.observed), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[0].set(title="Selected recipe: descriptive calibration\nLabels are bin counts, not independent evidence", xlabel="Mean forecast probability of A winning", ylabel="Observed fraction of A wins", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[0].set_aspect("equal")
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].bar(monthly.month, monthly.delta_ll, color=[COLORS[1] if value < 0 else "#b03a2e" for value in monthly.delta_ll])
    axes[1].margins(y=.15)
    for position, row in enumerate(monthly.itertuples()):
        axes[1].annotate(f"n={row.matches}", (position, row.delta_ll), xytext=(0, -12 if row.delta_ll < 0 else 4), textcoords="offset points", ha="center", fontsize=8)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set(title=f"Selected minus reference monthly loss\nNegative favors selected; n={summary['rows']:,}", ylabel="Delta log loss")
    fig.suptitle(f"Locked historical assessment: {summary['date_min']}–{summary['date_max']} — not prospective confirmation", fontsize=13)
    fig.savefig(report_dir / "final_diagnostics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    dataset_overview(protocol)
    if not args.data_only:
        result_figures(protocol)
    print(json.dumps({"report_dir": protocol["report_dir"], "data_only": args.data_only}))


if __name__ == "__main__":
    main()
