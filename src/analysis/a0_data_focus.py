"""Report the fixed A0 data-focus study using the existing benchmark primitives."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_benchmark import compute_benchmark_summary
from src.analysis.production_market_benchmark import paired_cluster_interval
from src.analysis.research_benchmark import attach_candidate, losses, safe, validate_frame
from src.models.a0_data_focus import MODES, PROJECT_ROOT, experiment_variants, resolve_research_root
from src.models.a0_native_refit import _sha256, _write_json_exclusive


def _summary(frame, column):
    labels = frame.y.to_numpy(dtype=int)
    probabilities = frame[column].to_numpy(dtype=float)
    result = asdict(compute_benchmark_summary(labels, probabilities, probability_epsilon=None))
    result.update(tail_ge_2_5=int((losses(labels, probabilities) >= 2.5).sum()),
                  min_probability=float(probabilities.min()), max_probability=float(probabilities.max()))
    return result


def _contrast(frame, candidate, reference, draws, seed):
    labels = frame.y.to_numpy(dtype=int)
    a, b = frame[candidate].to_numpy(dtype=float), frame[reference].to_numpy(dtype=float)
    delta = losses(labels, a) - losses(labels, b)
    brier_delta = np.square(a - labels) - np.square(b - labels)
    months = pd.to_datetime(frame.date).dt.to_period("M").astype(str)
    return {
        "candidate": candidate, "reference": reference,
        "log_loss_monthly": paired_cluster_interval(delta, months, draws=draws, seed=seed),
        "brier_monthly": paired_cluster_interval(brier_delta, months, draws=draws, seed=seed),
        "log_loss_tournament_name_proxy": paired_cluster_interval(
            delta, frame.tournament_name.astype(str), draws=draws, seed=seed),
    }


def _market_view(joined, panel, first=False):
    panel = panel.copy()
    panel.golgg_match_id = panel.golgg_match_id.astype(str)
    if panel.golgg_match_id.duplicated().any():
        raise ValueError("duplicate market fixture mapping")
    if not set(panel.golgg_match_id).issubset(set(joined.golgg_match_id)):
        raise ValueError("market cohort is not covered by every candidate")
    candidate = joined.set_index("golgg_match_id").loc[panel.golgg_match_id].reset_index()
    panel = panel.reset_index(drop=True)
    identity = {"y": "y", "best_of": "best_of", "team1_id": "team1_id_a0" if first else "team1_id",
                "team2_id": "team2_id_a0" if first else "team2_id"}
    for key, source in identity.items():
        dtype = float if key in ("y", "best_of") else str
        if not np.array_equal(candidate[key].to_numpy(dtype=dtype), panel[source].to_numpy(dtype=dtype)):
            raise ValueError(f"market identity mismatch: {key}")
    np.testing.assert_allclose(candidate.p_a0_full, panel.p_a0, atol=1e-12, rtol=0)
    candidate["q_consensus"] = panel.q_consensus.to_numpy(dtype=float)
    candidate["fixture_id"] = panel.fixture_id.to_numpy()
    candidate["book_count"] = panel.book_count.to_numpy(dtype=int)
    if not np.isfinite(candidate.q_consensus).all() or not candidate.q_consensus.between(0, 1, inclusive="neither").all():
        raise ValueError("invalid consensus probability")
    return candidate


def evaluate_experiment(protocol_path: Path, research_root: Path | None = None):
    """Score only the prespecified 2026 population; never pad earlier years."""
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    run_id = protocol["run_id"]
    output_root = PROJECT_ROOT / "data/07_model_output/a0_data_focus" / run_id
    reconstruction = protocol.get("schema") == "a0_history_reconstruction_v1"
    report_root = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / run_id
    if (report_root / "evaluation.json").exists():
        raise FileExistsError("refusing to overwrite a completed immutable evaluation")
    variants = experiment_variants(protocol)
    root = resolve_research_root(research_root)
    training = json.loads((report_root / "completed.json").read_text())
    if (training["status"] != "TRAINING_COMPLETE_NOT_SCORED"
            or training["contract"]["protocol_sha256"] != _sha256(protocol_path)):
        raise ValueError("a completed full run of this exact protocol is required")
    for relative, expected_hash in training["artifacts"].items():
        if _sha256(PROJECT_ROOT / relative) != expected_hash:
            raise ValueError(f"training artifact checksum differs: {relative}")
    canonical_path = root / protocol["inputs"]["canonical_relative_to_research_root"]
    if _sha256(canonical_path) != protocol["inputs"]["canonical_sha256"]:
        raise ValueError("canonical checksum differs from protocol")
    canonical = pd.read_parquet(canonical_path)
    joined = canonical.loc[pd.to_datetime(canonical.date).dt.year.eq(2026)].reset_index(drop=True)
    if len(joined) != protocol["split"]["validation_rows"]:
        raise ValueError("wrong canonical 2026 population")
    joined.golgg_match_id = joined.golgg_match_id.astype(str)
    joined = joined.rename(columns={"p_full": "p_a0_full", "p_no_w20": "p_a0_no_w20",
                                     "p_no_organization": "p_a0_no_organization"})
    probability_columns = [f"p_a0_{mode}" for mode in MODES]
    seed_rows = []
    for variant in variants:
        candidate = pd.read_parquet(output_root / f"{variant}.parquet")
        attach_candidate(joined, candidate, "p_full_full")
        if reconstruction:
            with np.load(output_root / f"{variant}_logits.npz") as raw_output:
                seed_probabilities = raw_output["raw_seed_probability"][-len(candidate):]
            for mode_index, mode in enumerate(MODES):
                np.testing.assert_allclose(seed_probabilities[:, mode_index].mean(1),
                                           candidate[f"p_raw_neural_{mode}"], atol=1e-12, rtol=0)
                for seed_index, seed in enumerate(protocol["training"]["seeds"]):
                    member = pd.DataFrame({"y": candidate.y.to_numpy(),
                                           "p": seed_probabilities[:, mode_index, seed_index]})
                    seed_rows.append({"cohort": "all_2026", "variant": variant, "seed": seed,
                                      "mode": mode, "kind": "raw_neural", **_summary(member, "p")})
        candidate.golgg_match_id = candidate.golgg_match_id.astype(str)
        candidate = candidate.set_index("golgg_match_id").loc[joined.golgg_match_id]
        for side in ("roster_a", "roster_b"):
            if [tuple(map(str, value)) for value in joined[side]] != [tuple(map(str, value)) for value in candidate[side]]:
                raise ValueError(f"candidate roster orientation differs: {variant}/{side}")
        if reconstruction:
            for column in ("history_stat_coverage", "csd_xpd_coverage"):
                values = candidate[column].to_numpy(dtype=float)
                if variant == "control":
                    joined[column] = values
                else:
                    np.testing.assert_allclose(joined[column], values, rtol=0, atol=0, equal_nan=True)
            for kind in ("neural", "raw_neural"):
                for mode in MODES:
                    column = f"nll_{kind}_{mode}"
                    values = candidate[column].to_numpy(dtype=float)
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError("invalid scoreline NLL")
                    joined[f"{variant}__{column}"] = values
        for kind in ("full", "neural", "raw_neural"):
            for mode in MODES:
                column = f"{variant}__{kind}__{mode}"
                joined[column] = candidate[f"p_{kind}_{mode}"].to_numpy(dtype=float)
                probability_columns.append(column)
    validate_frame(joined, probability_columns)
    for key in ("market_panel", "first_panel"):
        if _sha256(PROJECT_ROOT / protocol["inputs"][key]) != protocol["inputs"][key + "_sha256"]:
            raise ValueError(f"frozen market input changed: {key}")
    panel = pd.read_parquet(PROJECT_ROOT / protocol["inputs"]["market_panel"])
    panel = panel.loc[panel.horizon.eq("T-1h") & panel.strict_prior_day & panel.book_count.ge(2)]
    first = pd.read_parquet(PROJECT_ROOT / protocol["inputs"]["first_panel"])
    first = first.loc[first.strict_prior_day & first.book_count.ge(2)]
    cohorts = {"all_2026": joined, "market_T1": _market_view(joined, panel),
               "market_FIRST": _market_view(joined, first, first=True)}
    if len(cohorts["market_T1"]) != 1668 or len(cohorts["market_FIRST"]) != 407:
        raise ValueError("prespecified market coverage changed")
    cohorts["outside_T1"] = joined.loc[~joined.golgg_match_id.isin(cohorts["market_T1"].golgg_match_id)].copy()
    draws, seed = protocol["evaluation"]["bootstrap"]["resamples"], protocol["evaluation"]["bootstrap"]["seed"]
    summary_rows, contrasts = [], {}
    for name, frame in cohorts.items():
        frame.to_parquet(report_root / f"{name}_joined.parquet", index=False)
        contrasts[name] = {}
        columns = list(probability_columns)
        if "q_consensus" in frame:
            columns.append("q_consensus")
        for column in columns:
            summary_rows.append({"cohort": name, "model": column, **_summary(frame, column)})
        for variant in variants:
            for kind in ("full", "neural", "raw_neural"):
                for mode in MODES:
                    candidate = f"{variant}__{kind}__{mode}"
                    control = f"control__{kind}__{mode}"
                    if variant != "control":
                        contrasts[name][candidate + "_minus_control"] = _contrast(frame, candidate, control, draws, seed)
                    if reconstruction and variant == "reconstruction":
                        matched = f"supervised_18__{kind}__{mode}"
                        contrasts[name][candidate + "_minus_supervised_18"] = _contrast(frame, candidate, matched, draws, seed)
                    if kind == "full":
                        contrasts[name][candidate + "_minus_A0"] = _contrast(frame, candidate, f"p_a0_{mode}", draws, seed)
                        if "q_consensus" in frame:
                            contrasts[name][candidate + "_minus_consensus"] = _contrast(frame, candidate, "q_consensus", draws, seed)
        if not reconstruction:
            # Exploratory interaction for the original five-arm factorial study.
            labels = frame.y.to_numpy(dtype=int)
            cell_losses = {variant: losses(labels, frame[f"{variant}__full__full"]) for variant in
                           ("control", "market_weighted", "token_dropout", "market_weighted_dropout")}
            interaction = cell_losses["market_weighted_dropout"] - cell_losses["market_weighted"] - cell_losses["token_dropout"] + cell_losses["control"]
            contrasts[name]["exploratory_factorial_LL_interaction"] = paired_cluster_interval(
                interaction, pd.to_datetime(frame.date).dt.to_period("M").astype(str), draws=draws, seed=seed)
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(report_root / "metrics.csv", index=False)
    profiles = []
    for name in ("all_2026", "market_T1"):
        frame = cohorts[name].copy()
        frame["month"] = pd.to_datetime(frame.date).dt.to_period("M").astype(str)
        frame["has_unknown_players"] = frame.unknown_players.gt(0)
        frame["low_experience"] = frame.min_player_series.lt(5)
        frame["new_organization"] = frame.new_organization.astype(bool)
        dimensions = ["month", "best_of", "competition_tier", "has_unknown_players", "low_experience", "new_organization"]
        if reconstruction:
            for field in ("history_stat_coverage", "csd_xpd_coverage"):
                frame[field + "_bucket"] = np.select(
                    [frame[field].isna(), frame[field].lt(.9)], ["unknown", "<0.9"], default=">=0.9",
                )
                dimensions.append(field + "_bucket")
            dimensions.extend(["tournament_name", "competition_family"])
            frame["a0_confidence_bucket"] = pd.cut(
                np.maximum(frame.p_a0_full, 1 - frame.p_a0_full),
                [.5, .6, .7, .8, .9, 1.], include_lowest=True, right=False,
            )
            dimensions.append("a0_confidence_bucket")
        for dimension in dimensions:
            for value, group in frame.groupby(dimension, dropna=False, observed=True):
                for model in ["p_a0_full", *[f"{variant}__full__full" for variant in variants]]:
                    profiles.append({"cohort": name, "dimension": dimension, "value": str(value),
                                     "model": model, **_summary(group, model)})
        if reconstruction:
            for model in ["p_a0_full", *[f"{variant}__full__full" for variant in variants]]:
                confidence = np.maximum(frame[model], 1 - frame[model])
                bins = pd.cut(confidence, [.5, .6, .7, .8, .9, 1.], include_lowest=True, right=False)
                for value, group in frame.groupby(bins, observed=True):
                    conf = np.maximum(group[model], 1 - group[model])
                    accuracy = ((group[model] >= .5) == group.y).mean()
                    profiles.append({"cohort": name, "dimension": "own_forecast_confidence",
                                     "value": str(value), "model": model, **_summary(group, model),
                                     "mean_confidence": float(conf.mean()),
                                     "confidence_gap": float(conf.mean() - accuracy)})
    pd.DataFrame(profiles).to_csv(report_root / "profiles.csv", index=False)
    if reconstruction:
        primary = contrasts["all_2026"]["reconstruction__full__full_minus_supervised_18"]
        broader = primary
        market_signal = None
        all_purpose_guard = bool(primary["brier_monthly"]["delta"] <= 0)
        primary_success = bool(primary["log_loss_monthly"]["ci_upper"] < 0 and all_purpose_guard)
        scoreline_rows = []
        for name, frame in cohorts.items():
            for variant in variants:
                for kind in ("neural", "raw_neural"):
                    for mode in MODES:
                        scoreline_rows.append({"cohort": name, "variant": variant, "kind": kind, "mode": mode,
                                               "rows": len(frame), "terminal_scoreline_nll": float(
                                                   frame[f"{variant}__nll_{kind}_{mode}"].mean())})
        pd.DataFrame(seed_rows).to_csv(report_root / "seed_metrics.csv", index=False)
        pd.DataFrame(scoreline_rows).to_csv(report_root / "scoreline_metrics.csv", index=False)
    else:
        primary = contrasts["market_T1"]["market_weighted_dropout__full__full_minus_control"]
        broader = contrasts["all_2026"]["market_weighted_dropout__full__full_minus_control"]
        market_signal = bool(primary["log_loss_monthly"]["ci_upper"] < 0 and primary["brier_monthly"]["delta"] <= 0)
        all_purpose_guard = bool(broader["log_loss_monthly"]["delta"] <= 0 and broader["brier_monthly"]["delta"] <= 0)
        primary_success = market_signal and all_purpose_guard
    result = {
        "schema": "a0_data_focus_evaluation_v1", "run_id": run_id, "status": "COMPLETED_DEVELOPMENT_ONLY",
        "protocol_sha256": _sha256(protocol_path), "training_receipt_sha256": _sha256(report_root / "completed.json"),
        "metric_policy": "Unclipped finite probabilities strictly inside (0,1), same for summaries/deltas/profiles/tails",
        "cohort_rows": {key: len(value) for key, value in cohorts.items()},
        "primary": primary, "broader": broader, "primary_market_signal": market_signal,
        "all_purpose_nonregression_point_guard": all_purpose_guard,
        "primary_success": primary_success,
        "decision": "DEVELOPMENT_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION" if primary_success
                    else "PRESPECIFIED_PRIMARY_SUCCESS_NOT_ESTABLISHED",
        "comparisons": contrasts, "limitations": protocol["evaluation"]["limitations"] + [
            "Only nine 2026 calendar-month blocks; intervals are descriptive under repeated development.",
            "Tournament-name sensitivity is a proxy grouping, not independently audited tournament editions.",
            "CAL fits both native calibrators and the regularized mixture, not an independent CAL score.",
        ],
        "source_sha256": {str(Path(__file__).relative_to(PROJECT_ROOT)): _sha256(Path(__file__)),
                           "src/analysis/model_benchmark.py": _sha256(PROJECT_ROOT / "src/analysis/model_benchmark.py"),
                           "src/analysis/research_benchmark.py": _sha256(PROJECT_ROOT / "src/analysis/research_benchmark.py"),
                           "src/analysis/production_market_benchmark.py": _sha256(PROJECT_ROOT / "src/analysis/production_market_benchmark.py")},
        "artifacts": {str(path.relative_to(PROJECT_ROOT)): _sha256(path)
                      for path in sorted(report_root.iterdir()) if path.is_file()},
    }
    _write_json_exclusive(report_root / "evaluation.json", safe(result))
    print(summary_table.loc[summary_table.model.isin(["p_a0_full", "q_consensus", *[f"{variant}__full__full" for variant in variants]]),
                            ["cohort", "model", "sample_size", "log_loss", "brier_score", "tail_ge_2_5"]].to_string(index=False))
    print(json.dumps({"decision": result["decision"], "primary": result["primary"]}, indent=2))
    return result
