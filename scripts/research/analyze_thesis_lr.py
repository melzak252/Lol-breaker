"""Export fixed, retrospective LR diagnostic evidence from an already joined panel.

No training, calibration, model selection, or source-data mutation occurs here.
Run: python scripts/research/analyze_thesis_lr.py --panel panel.parquet --output NEW_DIR
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.metrics import binary_log_loss_vector
from src.analysis.thesis_evaluation import block_difference
from src.analysis.thesis_metrics import _metrics
from src.models.competition_tiers import classify_competition
from src.visualization.thesis_style import apply_thesis_style, PASTEL_BLUE, PASTEL_PURPLE

STATUS = "HISTORICAL_DEVELOPMENT_ONLY"
DRAWS = 5000
BOOTSTRAP_SEED = 20260921  # Same draws/seed contract as thesis_evaluation.
SEEDS = (7, 42, 2026)
NEW_STREAMS = tuple(
    f"p_{family}_{variant}_seed{seed}"
    for family in ("lr", "mlp")
    for variant in ("ts_standard", "original")
    for seed in SEEDS
) + ("p_mlp_original_no_early_stopping", "p_mlp_ts_standard_no_early_stopping", "p_ts_standard")
BASELINES = ("p_logistic", "p_mlp", "p_player_gl", "p_player_ts")
WINDOWS = {"W3": "p_lr_w3", "W5": "p_lr_w5", "W10": "p_lr_w10", "W20": "p_logistic", "W50": "p_lr_w50"}
ABLATIONS = {
    "no_context": "p_lr_player_only",
    "glicko_context": "p_lr_glicko_context",
    "no_player_gl": "p_lr_no_player_gl",
    "no_glicko": "p_lr_no_glicko",
    "player_team_58": "p_lr_player_team",
    "no_binomial": "p_supp_no_binomial",
    "player_team_80": "p_supp_full_player_team",
    "mask_base_10pct": "p_supp_mask_base10",
    "mask_player_gl_10pct": "p_supp_mask_playergl10",
    "mask_player_gl_30pct": "p_supp_mask_playergl30",
}
METRIC_FIELDS = ("logloss", "brier", "auc", "accuracy", "ece")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Validate the supplied join, never silently intersect model coverage."""
    required = {"golgg_match_id", "date", "y_true", "market_open", "market_close", *BASELINES, *NEW_STREAMS}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing panel columns: {missing}")
    if not panel.columns.is_unique:
        raise ValueError("Duplicate panel column names")
    if len(panel) != 11550:
        raise ValueError(f"Expected exactly 11550 full-cohort rows, got {len(panel)}")
    frame = panel.copy()
    ids = frame.golgg_match_id
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.astype(str).duplicated().any():
        raise ValueError("Missing or duplicate match IDs")
    frame["golgg_match_id"] = ids.astype(str)
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.isna().any() or (frame.date < pd.Timestamp("2024-01-01")).any():
        raise ValueError("Panel dates must be nonmissing and from 2024-01 onward")
    if not frame.y_true.isin([0, 1]).all():
        raise ValueError("Panel targets must be binary and nonmissing")
    models = [column for column in frame if column.startswith("p_")]
    coverage = []
    for column in models + ["market_open", "market_close"]:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        available = ~np.isnan(values)
        if column in models and not available.all():
            raise ValueError(f"Incomplete full-cohort coverage for {column}")
        if not (np.isfinite(values[available]) & (values[available] >= 0) & (values[available] <= 1)).all():
            raise ValueError(f"Nonfinite or out-of-range probabilities in {column}")
        frame[column] = values
        coverage.append({"model": column, "n_panel": len(frame), "n_available": int(available.sum()),
                         "n_missing": int((~available).sum()), "new_stream": column.removesuffix("_raw") in NEW_STREAMS})
    common = frame[["market_open", "market_close"]].notna().all(axis=1)
    if int(common.sum()) != 3229:
        raise ValueError(f"Expected exactly 3229 common OPEN+CLOSE rows, got {int(common.sum())}")
    return frame.sort_values(["date", "golgg_match_id"]).reset_index(drop=True), models, pd.DataFrame(coverage)


def fixed_contrasts(columns: list[str]) -> list[dict]:
    rows = []

    def add(kind: str, candidate: str, reference: str, label: str, seed=None):
        rows.append(dict(contrast_id=f"{candidate}__minus__{reference}", kind=kind,
                         candidate=candidate, reference=reference, label=label, model_seed=seed))

    for reference in ("p_mlp", "p_extra_trees", "p_hist_gradient_boosting", "p_lightgbm",
                      "p_player_gl", "market_open", "market_close"):
        add("original_baseline", "p_logistic", reference, f"Original LR minus {reference}")
    for seed in SEEDS:
        lr = f"p_lr_ts_standard_seed{seed}"
        mlp = f"p_mlp_ts_standard_seed{seed}"
        old_lr = f"p_lr_original_seed{seed}"
        old_mlp = f"p_mlp_original_seed{seed}"
        add("corrected_lr_mlp", lr, mlp, f"Corrected LR minus MLP, seed {seed}", seed)
        add("original_seed_lr_mlp", old_lr, old_mlp, f"Original LR minus MLP, seed {seed}", seed)
        add("lr_correction", lr, old_lr, f"LR corrected minus original, seed {seed}", seed)
        add("mlp_correction", mlp, old_mlp, f"MLP corrected minus original, seed {seed}", seed)
    for reference in ("p_player_gl", "market_open", "market_close"):
        add("corrected_baseline", "p_lr_ts_standard_seed42", reference,
            f"Corrected LR seed 42 minus {reference}", 42)
    for variant in ("original", "ts_standard"):
        control = f"p_mlp_{variant}_no_early_stopping"
        for family in ("mlp", "lr"):
            add("early_stopping_control", control, f"p_{family}_{variant}_seed42",
                f"{variant} MLP no ES minus {family.upper()}, seed 42", 42)
    add("rating_correction", "p_ts_standard", "p_player_ts", "Standard TS minus original player TS")
    for kind, mapping in (("window", WINDOWS), ("ablation", ABLATIONS)):
        for label, column in mapping.items():
            if column in columns and column != "p_logistic":
                add(kind, column, "p_logistic", f"{label} minus original LR W20")
    return rows


def grouping_labels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    competition = frame["competition"] if "competition" in frame else pd.Series("unknown", index=frame.index)
    competition = competition.fillna("unknown").astype(str).replace(r"^\s*$", "unknown", regex=True)
    return {
        "month": frame.date.dt.strftime("%Y-%m").to_numpy(),
        "tournament_year_proxy": (frame.date.dt.year.astype(str) + ":" + competition).to_numpy(),
    }


def taxonomy(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Use the existing date-aware taxonomy; never map unknown to a low tier."""
    names = frame["competition"] if "competition" in frame else pd.Series(None, index=frame.index, dtype=object)
    identities = [classify_competition(name, when.date()) for name, when in zip(names, frame.date)]
    tiers = pd.Series([identity.tier.value for identity in identities], index=frame.index)
    audit = pd.DataFrame({
        "competition": names.fillna("unknown").astype(str), "year": frame.date.dt.year,
        "taxonomy_tier": tiers, "taxonomy_family": [identity.family for identity in identities],
        "taxonomy_scope": [identity.scope.value for identity in identities],
        "matched_rule": [identity.matched_rule for identity in identities],
    })
    if "competition_tier" in frame:
        audit["supplied_tier"] = frame.competition_tier.fillna("unknown").astype(str)
    keys = list(audit.columns)
    return tiers, audit.groupby(keys, dropna=False, sort=True).size().rename("n").reset_index()


def interval_rows(delta: np.ndarray, labels: np.ndarray, blocking: str, metadata: dict) -> list[dict]:
    result = block_difference(delta, labels, draws=DRAWS, seed=BOOTSTRAP_SEED)
    common = {**metadata, "blocking": blocking, "n": result["n"], "blocks": result["blocks"],
              "draws": result["draws"], "bootstrap_seed": result["seed"],
              "interval_level": 0.95, "interval_status": "NOMINAL_NOT_SELECTION_ADJUSTED",
              "selection_status": STATUS}
    return [
        {**common, "estimand": "match_weighted", "delta": result["delta"],
         "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"]},
        {**common, "estimand": "equal_month" if blocking == "month" else "equal_tournament_year_proxy",
         "delta": result["equal_block_delta"], "ci_lower": result["equal_block_ci_lower"],
         "ci_upper": result["equal_block_ci_upper"]},
    ]


def tail_rows(loss: np.ndarray, metadata: dict) -> list[dict]:
    """Descriptive tails of the existing per-match LogLoss, not another score."""
    ordered = np.sort(loss)
    rows = [{**metadata, "tail_statistic": f"quantile_{q:g}", "tail_fraction": 1 - q,
             "n": len(loss), "tail_n": None, "value": float(np.quantile(loss, q))}
            for q in (0.90, 0.95, 0.99, 1.0)]
    for fraction in (0.05, 0.01):
        count = max(1, int(np.ceil(len(loss) * fraction)))
        rows.append({**metadata, "tail_statistic": f"worst_{fraction:g}_mean",
                     "tail_fraction": fraction, "n": len(loss), "tail_n": count,
                     "value": float(ordered[-count:].mean())})
    return rows


def save_chart(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def charts(comparisons: pd.DataFrame, slices: pd.DataFrame, output: Path) -> None:
    apply_thesis_style()
    for cohort in ("sports", "market_common"):
        data = comparisons[(comparisons.cohort == cohort) & (comparisons.metric == "logloss")
                           & (comparisons.blocking == "month") & (comparisons.estimand == "match_weighted")
                           & ~comparisons.kind.isin(["window", "ablation", "original_seed_lr_mlp"])]
        fig, ax = plt.subplots(figsize=(10, 7))
        positions = np.arange(len(data))
        # Drawing intervals directly also handles the rare percentile CI not enclosing its point.
        ax.hlines(positions, data.ci_lower, data.ci_upper, color=PASTEL_BLUE, linewidth=1.6)
        ax.scatter(data.delta, positions, color=PASTEL_BLUE, s=22, zorder=3)
        ax.set_yticks(positions, [f"{r.label} [N={r.n}, M={r.blocks}]" for r in data.itertuples()], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Paired Δ LogLoss: candidate − reference (negative favors candidate)")
        ax.set_title(f"{cohort}: fixed contrasts; monthly paired bootstrap, 5000 draws\n"
                     "Match-weighted; nominal 95% intervals, not selection-adjusted")
        ax.grid(axis="x", alpha=0.25)
        fig.text(0.5, 0.005, STATUS, ha="center", fontsize=8)
        save_chart(fig, output, f"overall_contrasts_{cohort}")

    monthly = slices[(slices.dimension == "month") & (slices.metric == "logloss")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, cohort in zip(axes, ("sports", "market_common")):
        subset = monthly[monthly.cohort == cohort]
        months = sorted(subset["slice"].unique())
        counts = subset.drop_duplicates("slice").set_index("slice")["n"]
        for version, color in (("original", PASTEL_BLUE), ("corrected_seed42", PASTEL_PURPLE)):
            data = subset[subset.version == version].set_index("slice").reindex(months)
            ax.plot(np.arange(len(months)), data.delta, marker="o", markersize=3,
                    label=version.replace("_", " "), color=color)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_ylabel("Δ LogLoss: LR − MLP")
        ax.set_title(f"{cohort}; paired within-month differences, N={int(counts.sum())}; descriptive only")
        ax.set_xticks(np.arange(len(months)), [f"{month}\nn={int(counts.loc[month])}" for month in months],
                      rotation=70, ha="right", fontsize=7)
        ax.tick_params(labelbottom=True)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Month and identical paired match count; negative favors LR")
    fig.suptitle(f"Monthly stability — {STATUS}", fontsize=11)
    save_chart(fig, output, "monthly_lr_mlp_differences")


def analyze(panel_path: Path, output: Path) -> dict:
    """Validate and export evidence; output must not already exist."""
    panel_path, output = Path(panel_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    if panel_path.suffix.lower() == ".parquet":
        panel = pd.read_parquet(panel_path)
    elif panel_path.suffix.lower() == ".csv":
        panel = pd.read_csv(panel_path, dtype={"golgg_match_id": str})
    else:
        raise ValueError("Panel must be .parquet or .csv")
    frame, models, coverage = validate_panel(panel)
    tiers, taxonomy_audit = taxonomy(frame)
    frame["analysis_competition_tier"] = tiers
    cohorts = {"sports": frame, "market_common": frame[frame[["market_open", "market_close"]].notna().all(axis=1)]}
    contrasts = fixed_contrasts(models)
    metric_rows, reliability, tails, comparisons, slice_metrics, paired_slices, leave_out = [], [], [], [], [], [], []
    selected_pairs = (("original", "p_logistic", "p_mlp"),
                      ("corrected_seed42", "p_lr_ts_standard_seed42", "p_mlp_ts_standard_seed42"))
    for cohort, part in cohorts.items():
        columns = models + (["market_open", "market_close"] if cohort == "market_common" else [])
        y = part.y_true.to_numpy(dtype=float)
        groups = grouping_labels(part)
        losses = {}
        for model in columns:
            p = part[model].to_numpy(dtype=float)
            scores, bins = _metrics(y, p)
            key = {"cohort": cohort, "model": model, "selection_status": STATUS}
            metric_rows.append({**key, "months": len(np.unique(groups["month"])), **scores})
            reliability.extend({**key, **row} for row in bins)
            losses[model] = {"logloss": binary_log_loss_vector(y, p), "brier": (p - y) ** 2}
            tails.extend(tail_rows(losses[model]["logloss"], key))
        for contrast in contrasts:
            candidate, reference = contrast["candidate"], contrast["reference"]
            if candidate not in losses or reference not in losses:
                continue  # OPEN/CLOSE have no full-cohort comparator.
            for metric in ("logloss", "brier"):
                delta = losses[candidate][metric] - losses[reference][metric]
                key = {"cohort": cohort, **contrast, "metric": metric}
                for blocking, labels in groups.items():
                    comparisons.extend(interval_rows(delta, labels, blocking, key))
        bon = part["BoN"] if "BoN" in part else pd.Series(None, index=part.index, dtype=object)
        dimensions = {"year": part.date.dt.year.astype(str).to_numpy(), "month": groups["month"],
                      "BoN": bon.fillna("unknown").astype(str).to_numpy(),
                      "competition_tier": part.analysis_competition_tier.to_numpy()}
        for dimension, labels in dimensions.items():
            for label in sorted(np.unique(labels)):
                mask = labels == label
                n = int(mask.sum())
                month_count = int(len(np.unique(groups["month"][mask])))
                key = {"cohort": cohort, "dimension": dimension, "slice": label, "n": n,
                       "months": month_count, "positive_n": int(y[mask].sum()),
                       "negative_n": int(n - y[mask].sum()), "inference": "DESCRIPTIVE_ONLY",
                       "selection_status": STATUS}
                for version, candidate, reference in selected_pairs:
                    scored = {}
                    for model in (candidate, reference):
                        scored[model], _ = _metrics(y[mask], part[model].to_numpy(dtype=float)[mask])
                        slice_metrics.append({**key, "version": version, "model": model, **scored[model]})
                    for metric in METRIC_FIELDS:
                        paired_slices.append({**key, "version": version, "candidate": candidate,
                                              "reference": reference, "metric": metric,
                                              "candidate_value": scored[candidate][metric],
                                              "reference_value": scored[reference][metric],
                                              "delta": scored[candidate][metric] - scored[reference][metric]})
        for version, candidate, mlp in selected_pairs:
            references = [mlp] + (["market_open"] if cohort == "market_common" else [])
            for reference in references:
                for metric in ("logloss", "brier"):
                    delta = losses[candidate][metric] - losses[reference][metric]
                    monthly = pd.DataFrame({"month": groups["month"], "delta": delta}).groupby("month").delta.agg(["sum", "count", "mean"])
                    for omitted, row in monthly.iterrows():
                        kept = monthly.drop(index=omitted)
                        leave_out.append({"cohort": cohort, "version": version, "candidate": candidate,
                                          "reference": reference, "metric": metric, "omitted_month": omitted,
                                          "omitted_n": int(row["count"]), "n": int(kept["count"].sum()),
                                          "months": len(kept), "full_delta": float(delta.mean()),
                                          "delta": float(kept["sum"].sum() / kept["count"].sum()),
                                          "full_equal_month_delta": float(monthly["mean"].mean()),
                                          "equal_month_delta": float(kept["mean"].mean()),
                                          "inference": "DESCRIPTIVE_ONLY_NO_REFIT", "selection_status": STATUS})
    metrics = pd.DataFrame(metric_rows)
    comparison_table = pd.DataFrame(comparisons)
    paired_table = pd.DataFrame(paired_slices)
    availability = pd.DataFrame([
        {"kind": kind, "variant": label, "model": model, "present": model in models,
         "reference": "p_logistic", "selection_status": STATUS}
        for kind, mapping in (("window", WINDOWS), ("ablation", ABLATIONS))
        for label, model in mapping.items()
    ])
    sensitivity_metrics = availability[availability.present].merge(metrics, on=["model", "selection_status"], how="inner")
    output.mkdir(parents=True, exist_ok=False)
    tables = {
        "metrics": metrics, "reliability": pd.DataFrame(reliability), "tails": pd.DataFrame(tails),
        "comparisons": comparison_table, "slices": pd.DataFrame(slice_metrics), "paired_slices": paired_table,
        "leave_one_month_out": pd.DataFrame(leave_out), "stream_coverage": coverage,
        "taxonomy": taxonomy_audit, "sensitivity_availability": availability,
        "window_ablation_metrics": sensitivity_metrics,
        "window_ablation_comparisons": comparison_table[comparison_table.kind.isin(["window", "ablation"])],
        "contrast_plan": pd.DataFrame(contrasts),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    charts(comparison_table, paired_table, output)
    dependencies = [Path(__file__), ROOT / "src/analysis/thesis_evaluation.py", ROOT / "src/analysis/thesis_metrics.py",
                    ROOT / "src/analysis/metrics.py", ROOT / "src/models/competition_tiers.py",
                    ROOT / "src/visualization/thesis_style.py"]
    receipt = {
        "execution_status": "COMPLETE", "selection_status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(), "panel": str(panel_path.resolve()),
        "panel_sha256": sha256(panel_path), "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in dependencies},
        "python": platform.python_version(),
        "versions": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
        "cohorts": {name: {"n": len(part), "months": int(part.date.dt.to_period("M").nunique()),
                              "first_date": str(part.date.min()), "last_date": str(part.date.max()),
                              "id_set_sha256": hashlib.sha256("\n".join(sorted(part.golgg_match_id)).encode()).hexdigest()}
                    for name, part in cohorts.items()},
        "required_new_streams": list(NEW_STREAMS), "scored_streams": models,
        "bootstrap": {"draws": DRAWS, "seed": BOOTSTRAP_SEED, "interval_level": 0.95,
                      "implementation": "src.analysis.thesis_evaluation.block_difference",
                      "blocking": ["month", "tournament_year_proxy"],
                      "estimands": ["match_weighted", "equal_month", "equal_tournament_year_proxy"],
                      "interval_status": "NOMINAL_NOT_SELECTION_ADJUSTED"},
        "metric_policy": {"implementation": "src.analysis.thesis_metrics._metrics",
                          "per_match_logloss": "src.analysis.metrics.binary_log_loss_vector; epsilon=1e-15",
                          "brier": "Original probability squared error", "accuracy": "p >= 0.5",
                          "ece": "10 equal-width bins; endpoints included; original probabilities",
                          "tails": "LogLoss quantiles (numpy linear interpolation) and mean of worst ceil(N*fraction) rows; no trimming of primary metrics",
                          "undefined": "CSV empty cells denote undefined AUC/calibration fits or single-block intervals"},
        "taxonomy": {"implementation": "src.models.competition_tiers.classify_competition(name, match_date)",
                     "slice_policy": "Recomputed auditable taxonomy; supplied tier retained only in taxonomy audit, never silently relabelled",
                     "unknown_policy": "Missing/unmatched names remain unknown; missing BoN remains unknown"},
        "limitations": [
            "Historical diagnostic, not prospective confirmation or new model selection; no winning seed/window chosen.",
            "Nominal bootstrap intervals are not adjusted for historical selection or multiple comparisons.",
            "Tournament-year blocks concatenate year and tournament name; sensitivity proxy, not certified edition IDs.",
            "Per-year/month/BoN/tier comparisons are descriptive only; counts explicit, no subgroup significance claims.",
            "Leave-one-month-out removes scoring rows only, not model training or prior calibration data.",
            "Market metrics use exactly the common OPEN+CLOSE cohort; no separate OPEN-only/CLOSE-only score populations.",
            "First-map roster-known scenario and retrospective market quotes; precise release/quote-time availability uncertified.",
            "Joined-panel validation proves row/finite coverage, not upstream joins, calibration provenance, or model fitting correctness.",
        ],
        "table_rows": {name: len(table) for name, table in tables.items()},
        "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "execution_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = analyze(args.panel, args.output)
    print(json.dumps({"selection_status": receipt["selection_status"], "output": str(args.output),
                      "cohorts": receipt["cohorts"], "execution_status": receipt["execution_status"]}, indent=2))


if __name__ == "__main__":
    main()
