"""Monthly block-bootstrap LogLoss differences by prediction horizon.

Porównuje model teoretyczny i hybrydowy z bookmakerem w podziale na horyzont
czasowy (godziny przed meczem). Metodologia: miesięczny block bootstrap
(10 000 resampli, seed 42), ΔLogLoss = bookmaker - model, 95% CI, p-value.

Jednostką obserwacji jest model × mecz × horyzont. Wiele snapshotów kursów
w tym samym meczu i horyzoncie jest najpierw agregowane do jednej
prawdopodobieństwa rynku, dzięki czemu bootstrap nie zawyża liczebności próby.

Usage:
  docker exec ensemblelegends-betting-api python3 \\
    /app/betting_app/scripts/horizon_block_bootstrap.py

Output (/app/docs/assets/horizon_block_bootstrap/):
  - horizon_block_bootstrap_results.csv
  - horizon_monthly_observed_differences.csv
  - horizon_block_bootstrap_samples.csv
  - horizon_block_bootstrap_ci.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_CWD = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _CWD.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from betting_app.core.db import get_session
from sqlalchemy import text

# ── Configuration ──────────────────────────────────────────────────────────

MODELS: list[dict] = [
    {
        "name": "Sym-Cal LR-ElasticNet-W20-Binomial",
        "version": "exp-039",
        "label": "Thesis",
        "kind": "thesis",
    },
    {
        "name": "Hybrid-Thesis-Market",
        "version": "a0.35-t0.80",
        "label": "Hybrid",
        "kind": "hybrid",
    },
]

THESIS_MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
THESIS_MODEL_VERSION = "exp-039"
ANALYSIS_FEATURES_VERSION = "exp060-db-backfill-v1"
HYBRID_ALPHA = 0.35
HYBRID_TEMPERATURE = 0.80

N_BOOTSTRAPS = 10_000
RANDOM_SEED = 42
EPSILON = 1e-15

BIN_DEFS: list[tuple[str, float, float]] = [
    ("0-2h",   0,   2),
    ("2-6h",   2,   6),
    ("6-12h",  6,  12),
    ("12-24h", 12,  24),
    ("24-48h", 24,  48),
    ("48h+",   48,  9999),
]

OUTPUT_DIR = Path(os.getenv("BOOTSTRAP_DIR", str(_PROJECT_ROOT / "data" / "horizon_block_bootstrap")))


# ── Data structures ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BootstrapResult:
    model_label: str
    model_name: str
    comparison: str
    label: str
    hours_start: float
    hours_end: float
    sample_size: int
    n_blocks: int
    model_logloss: float
    benchmark_logloss: float
    observed_difference: float
    ci_low: float
    ci_high: float
    p_one_sided: float
    significant_05: bool


# ── Helpers ────────────────────────────────────────────────────────────────

def _implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.5


def _remove_margin(p_a: float, p_b: float) -> tuple[float, float]:
    total = p_a + p_b
    if total <= 0:
        return 0.5, 0.5
    return p_a / total, p_b / total


def log_loss_vector(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    y_pred_c = np.clip(y_pred, EPSILON, 1 - EPSILON)
    return -(y_true * np.log(y_pred_c) + (1 - y_true) * np.log(1 - y_pred_c))


def _assign_bin(hours_before: float) -> str | None:
    for label, hmin, hmax in BIN_DEFS:
        if hmin <= hours_before < hmax:
            return label
    return None


def _temperature_probability(prob: float, temperature: float) -> float:
    """Temperature-scale a probability, matching production hybrid logic."""
    eps = 1e-15
    clipped = max(eps, min(1.0 - eps, float(prob)))
    logit = np.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + np.exp(-(logit / temperature))))


# ── Data loading ───────────────────────────────────────────────────────────

def load_model_data(model_cfg: dict) -> pd.DataFrame:
    """Load the one reproducible EXP-060 source and derive the selected model.

    Historical stored hybrid rows are deliberately not used: older deployments
    wrote a0.05/a0.50 variants, so mixing them makes the bootstrap claim a
    comparison for no single model.  The current a0.35 hybrid is reconstructed
    from each snapshot's market probability and the EXP-060 thesis prediction.
    """
    records: list[dict] = []
    model_kind = str(model_cfg["kind"])

    with get_session() as sess:
        matches = sess.execute(text("""
            SELECT DISTINCT ON (cp.canonical_match_id)
                cp.canonical_match_id AS match_id,
                cm.start_time_normalized,
                cm.normalized_team_a,
                cm.normalized_team_b,
                cm.league,
                cm.winner_side,
                cp.prob_a,
                cp.prob_b
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cp.canonical_match_id = cm.id
            WHERE cp.model_name = :model_name
              AND cp.model_version = :model_version
              AND cp.features_version = :features_version
              AND cp.predicted_at <= cm.start_time_normalized::timestamptz
              AND cm.status IN ('finished', 'completed')
              AND cm.winner_side IS NOT NULL
              AND cm.start_time_normalized IS NOT NULL
            ORDER BY cp.canonical_match_id, cp.predicted_at DESC
        """), {
            "model_name": THESIS_MODEL_NAME,
            "model_version": THESIS_MODEL_VERSION,
            "features_version": ANALYSIS_FEATURES_VERSION,
        }).fetchall()

        match_ids = [m.match_id for m in matches]
        if not match_ids:
            print(f"  ⚠ No finished matches for {model_cfg['label']}")
            return pd.DataFrame()

        snap_params = {}
        snap_placeholders = []
        for i, mid in enumerate(match_ids):
            key = f"mid{i}"
            snap_params[key] = mid
            snap_placeholders.append(f":{key}")

        snapshots = sess.execute(text(f"""
            SELECT os.canonical_match_id, os.scraped_at,
                   os.odds_a, os.odds_b,
                   os.raw_team_a, os.raw_team_b
            FROM odds_snapshots os
            WHERE os.canonical_match_id IN ({','.join(snap_placeholders)})
              AND os.market_type = 'match_winner'
              AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a IS NOT NULL
              AND os.odds_b IS NOT NULL
            ORDER BY os.canonical_match_id, os.scraped_at
        """), snap_params).fetchall()

    match_map = {m.match_id: m for m in matches}

    for snap in snapshots:
        m = match_map.get(snap.canonical_match_id)
        if not m:
            continue

        match_start = m.start_time_normalized
        if isinstance(match_start, str):
            match_start = datetime.fromisoformat(match_start.replace("Z", "+00:00"))
        scraped_at = snap.scraped_at
        if isinstance(scraped_at, str):
            scraped_at = datetime.fromisoformat(scraped_at.replace("Z", "+00:00"))
        if not match_start or not scraped_at:
            continue

        hours_before = (match_start - scraped_at).total_seconds() / 3600.0
        if hours_before < 0:
            continue

        bin_label = _assign_bin(hours_before)
        if not bin_label:
            continue

        n_a = str(m.normalized_team_a or "")
        n_b = str(m.normalized_team_b or "")
        raw_a = str(snap.raw_team_a or "")
        raw_b = str(snap.raw_team_b or "")
        odds_a = snap.odds_a
        odds_b = snap.odds_b

        from betting_app.services.canonical_match_service import align_snapshot_odds
        aligned = align_snapshot_odds(
            n_a, n_b, raw_a, raw_b, odds_a, odds_b,
            league=m.league, source_system="bookmaker", match_date=match_start,
        )
        if aligned is None:
            continue
        aligned_a, aligned_b = aligned

        p_a_raw = _implied_prob(aligned_a)
        p_b_raw = _implied_prob(aligned_b)
        p_a, p_b = _remove_margin(p_a_raw, p_b_raw)

        winner = str(m.winner_side or "")
        if winner == "team_a":
            y_true = 1
        elif winner == "team_b":
            y_true = 0
        else:
            continue

        thesis_prob = float(m.prob_a) if m.prob_a is not None else None
        model_prob = thesis_prob
        if model_kind == "hybrid" and thesis_prob is not None:
            model_prob = (
                HYBRID_ALPHA * _temperature_probability(thesis_prob, HYBRID_TEMPERATURE)
                + (1.0 - HYBRID_ALPHA) * p_a
            )
        if model_prob is None:
            continue

        month_str = match_start.strftime("%Y-%m")

        records.append({
            "match_id": m.match_id,
            "y_true": y_true,
            "model_prob": model_prob,
            "bookmaker_prob": p_a,
            "hours_before": hours_before,
            "horizon_bin": bin_label,
            "month": month_str,
        })

    snapshot_df = pd.DataFrame(records)
    if snapshot_df.empty:
        print("  No data after filtering.")
        return snapshot_df

    # Match-oriented statistics: collapse all bookmaker snapshots from the same
    # model × match × horizon into one observation before computing LogLoss or
    # bootstrapping. Snapshot count is kept only as coverage/diagnostic context.
    df = (
        snapshot_df
        .groupby(["match_id", "horizon_bin"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            model_prob=("model_prob", "mean"),
            bookmaker_prob=("bookmaker_prob", "mean"),
            hours_before=("hours_before", "mean"),
            month=("month", "first"),
            n_snapshots=("bookmaker_prob", "size"),
        )
    )
    print(
        f"  Loaded {len(snapshot_df):,} snapshots -> {len(df):,} match-horizon observations, "
        f"{df['match_id'].nunique():,} matches, {df['month'].nunique():,} months, "
        f"{df['horizon_bin'].nunique():,} bins."
    )
    return df


# ── Bootstrap ──────────────────────────────────────────────────────────────

def per_bin_bootstrap(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Monthly block bootstrap for one horizon bin.

    Input `df` is already match-oriented: one row per model × match × horizon.
    Monthly blocks are resampled and weighted by number of match observations.
    """
    working = df.dropna(subset=["model_prob", "bookmaker_prob", "y_true", "month"]).copy()
    if working.empty:
        return np.array([]), pd.DataFrame()

    working["model_loss"] = log_loss_vector(working["y_true"], working["model_prob"])
    working["bookmaker_loss"] = log_loss_vector(working["y_true"], working["bookmaker_prob"])
    working["loss_diff"] = working["bookmaker_loss"] - working["model_loss"]

    month_summary = (
        working.groupby("month", as_index=False)
        .agg(
            n_snapshots=("n_snapshots", "sum"),
            n_matches=("match_id", "nunique"),
            n_observations=("loss_diff", "size"),
            model_loss_sum=("model_loss", "sum"),
            bookmaker_loss_sum=("bookmaker_loss", "sum"),
            diff_sum=("loss_diff", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    month_summary["model_logloss"] = month_summary["model_loss_sum"] / month_summary["n_matches"]
    month_summary["bookmaker_logloss"] = month_summary["bookmaker_loss_sum"] / month_summary["n_matches"]
    month_summary["mean_difference"] = month_summary["diff_sum"] / month_summary["n_matches"]

    n_blocks = len(month_summary)
    diff_values = np.empty(N_BOOTSTRAPS, dtype=float)
    indices = np.arange(n_blocks)
    n_m_arr = month_summary["n_matches"].to_numpy(dtype=float)
    d_sums = month_summary["diff_sum"].to_numpy(dtype=float)

    for i in range(N_BOOTSTRAPS):
        sampled = rng.choice(indices, size=n_blocks, replace=True)
        diff_values[i] = d_sums[sampled].sum() / n_m_arr[sampled].sum()

    return diff_values, month_summary


def summarize(
    df: pd.DataFrame,
    label: str,
    hmin: float,
    hmax: float,
    model_label: str,
    model_name: str,
    bootstrap_diffs: np.ndarray,
    month_summary: pd.DataFrame,
) -> BootstrapResult:
    working = df.dropna(subset=["model_prob", "bookmaker_prob", "y_true", "month"])
    if working.empty or len(bootstrap_diffs) == 0:
        return BootstrapResult(
            model_label=model_label, model_name=model_name,
            comparison=f"{model_label} {label} vs Bookmaker", label=label,
            hours_start=hmin, hours_end=hmax if hmax < 9999 else float("inf"),
            sample_size=0, n_blocks=0,
            model_logloss=0.0, benchmark_logloss=0.0,
            observed_difference=0.0, ci_low=0.0, ci_high=0.0,
            p_one_sided=1.0, significant_05=False,
        )

    ml = log_loss_vector(working["y_true"], working["model_prob"])
    bl = log_loss_vector(working["y_true"], working["bookmaker_prob"])
    obs_diff = float(bl.mean() - ml.mean())
    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])
    p = float((np.sum(bootstrap_diffs <= 0.0) + 1) / (len(bootstrap_diffs) + 1))

    return BootstrapResult(
        model_label=model_label, model_name=model_name,
        comparison=f"{model_label} {label} vs Bookmaker", label=label,
        hours_start=hmin, hours_end=hmax if hmax < 9999 else float("inf"),
        sample_size=int(working["match_id"].nunique()), n_blocks=len(month_summary),
        model_logloss=float(ml.mean()), benchmark_logloss=float(bl.mean()),
        observed_difference=obs_diff,
        ci_low=ci_low, ci_high=ci_high,
        p_one_sided=p, significant_05=bool(ci_low > 0.0),
    )


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_combined_ci(results: pd.DataFrame, output_path: Path) -> None:
    """Panelled plot: Thesis (left), Hybrid (right), ΔLogLoss per horizon bin."""

    models_ordered = ["Thesis", "Hybrid"]
    colors_palette = {"Thesis": "#4C72B0", "Hybrid": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    for ax, mdl in zip(axes, models_ordered):
        sub = results[results["model_label"] == mdl].sort_values("hours_start").reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"{mdl} — brak danych", fontsize=13, weight="bold")
            continue

        y_pos = np.arange(len(sub))
        sig_color = ["#6F8F72" if s else "#A68A64" for s in sub["significant_05"]]

        ax.errorbar(
            sub["observed_difference"], y_pos,
            xerr=[sub["observed_difference"] - sub["ci_low"],
                  sub["ci_high"] - sub["observed_difference"]],
            fmt="none", ecolor="#4A5568", elinewidth=2.0, capsize=4,
        )
        ax.scatter(sub["observed_difference"], y_pos, s=80,
                   color=colors_palette[mdl], zorder=3, edgecolors="white", linewidth=0.5)
        ax.axvline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["label"], fontsize=11)
        ax.set_title(mdl, fontsize=14, weight="bold")
        ax.grid(axis="x", alpha=0.25)

        # Annotate sample sizes
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.annotate(f"n={row['sample_size']}",
                        xy=(row["observed_difference"], i),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=8, color="#666666",
                        ha="left" if row["observed_difference"] >= 0 else "right")

    axes[0].set_ylabel("Horyzont", fontsize=12)
    fig.supxlabel("Δ LogLoss: bookmaker - model", fontsize=12, y=0.02)
    fig.suptitle("Miesięczny block bootstrap — różnica LogLoss względem bookmakera",
                 fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    all_results: list[BootstrapResult] = []
    all_monthly: list[pd.DataFrame] = []
    bootstrap_samples: dict[str, np.ndarray] = {}

    for model_cfg in MODELS:
        mdl_name = model_cfg["name"]
        mdl_label = model_cfg["label"]
        print(f"\n{'='*60}")
        print(f"Model: {mdl_label} ({mdl_name})")
        print(f"{'='*60}")

        df = load_model_data(model_cfg)
        if df.empty:
            print("  ⚠ Skipping — no data.")
            continue

        for label, hmin, hmax in BIN_DEFS:
            print(f"  Bootstrap {label} …", end=" ")
            bin_df = df[df["horizon_bin"] == label].copy()
            if bin_df.empty:
                print("⚠ no data, skip")
                continue

            diffs, month_summary = per_bin_bootstrap(bin_df, rng)
            if len(diffs) == 0:
                print("⚠ bootstrap failed")
                continue

            result = summarize(bin_df, label, hmin, hmax, mdl_label, mdl_name,
                               diffs, month_summary)
            all_results.append(result)

            samp_key = f"{mdl_label}_{label}"
            bootstrap_samples[samp_key] = diffs

            ms = month_summary.copy()
            ms["model_label"] = mdl_label
            ms["horizon_bin"] = label
            all_monthly.append(ms)

            sig_str = "✅" if result.significant_05 else "❌"
            print(f"Δ={result.observed_difference:.4f}  "
                  f"CI=[{result.ci_low:.4f},{result.ci_high:.4f}]  "
                  f"p={result.p_one_sided:.4f}  "
                  f"matches={result.sample_size}  blocks={result.n_blocks}  {sig_str}")

    if not all_results:
        print("ERROR: No results generated.")
        sys.exit(1)

    # ── Output CSVs ──
    results_df = pd.DataFrame([r.__dict__ for r in all_results])
    results_df = results_df.sort_values(["model_label", "hours_start"]).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "horizon_block_bootstrap_results.csv", index=False)

    if all_monthly:
        monthly_df = pd.concat(all_monthly, ignore_index=True)
        monthly_df.to_csv(OUTPUT_DIR / "horizon_monthly_observed_differences.csv", index=False)

    bootstrap_df = pd.DataFrame(bootstrap_samples)
    bootstrap_df.to_csv(OUTPUT_DIR / "horizon_block_bootstrap_samples.csv", index=False)

    # ── Plot ──
    plot_combined_ci(results_df, OUTPUT_DIR / "horizon_block_bootstrap_ci.png")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary_cols = ["model_label", "label", "sample_size", "n_blocks",
                    "observed_difference", "ci_low", "ci_high",
                    "p_one_sided", "significant_05"]
    print(results_df[summary_cols].to_string(index=False))
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("Files: horizon_block_bootstrap_results.csv, "
          "horizon_monthly_observed_differences.csv, "
          "horizon_block_bootstrap_samples.csv, "
          "horizon_block_bootstrap_ci.png")


if __name__ == "__main__":
    main()
