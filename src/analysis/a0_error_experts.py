"""Capture the actual frozen mixture inputs on repaired primary origins."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import ev_asof_replay as replay
from src.analysis.a0_error_audit import ROOT, save_json, sha256
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.probability_metrics import binary_log_loss_vector

EXPERTS = ("neural", "rating", "mlp")


def run(tables: dict, protocol: dict, output: Path) -> dict:
    primary = tables["opportunities"].loc[lambda frame:
        frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price")
        & frame.horizon.eq("T-1h") & frame.audit_primary_eligible].copy()
    source = pd.read_parquet(protocol["paths"]["repair_predictions"])
    source = source.loc[source.origin_id.isin(primary.origin_id)].copy()
    if len(source) != len(primary) or source.origin_id.duplicated().any():
        raise ValueError("Expert capture must cover every primary origin exactly once")
    origin_fields = ["cohort", "fixture_id", "horizon", "origin_id", "golgg_match_id", "row_id", "scenario_decision_at", "decision_at", "information_status", "scenario_timestamp_assumption", "start_at"]
    root = protocol["resolved_research_root"]
    legacy = replay._verify_protocol(ROOT / "data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json")
    bank, releases, bank_receipt, bank_hashes = replay._verify_corrected_bank(root, legacy, protocol["paths"]["raw_history"])
    original, bank_arrays, bank_temporal = replay._read_bank_validation_data(bank, bank_receipt)
    arrays = replay._build_replay_arrays(
        source[origin_fields], original[["golgg_match_id", "team1_id", "team2_id", "best_of", "date"]],
        root, protocol["paths"]["raw_history"], releases,
        validation_controls=replay._select_oracle_controls(original), bank_arrays=bank_arrays, bank_temporal=bank_temporal,
    )
    del original, bank_arrays, bank_temporal
    gc.collect()
    targets = arrays.targets
    if not targets.feature_available.all() or set(targets.origin_id) != set(source.origin_id):
        raise ValueError("Expert replay changed repaired input availability")
    if set(targets.model_year) != {2026}:
        raise ValueError("This diagnostic is scoped to the fixed 2026 replay")
    predictor = replay._activate_native_predictor(root).CausalA0Predictor(2026)
    baseline = json.loads(protocol["paths"]["repair_receipt"].read_text())
    prefix = f"{replay.CAUSAL_ARCHIVE_RELATIVE}/data/06_models/causal_a0/2026/"
    expected = {str(path).split(prefix, 1)[-1]: value for path, value in baseline["models"]["2026"]["artifact_hashes"].items()}
    observed = {str(path).split(prefix, 1)[-1]: value for path, value in predictor.hashes.items()}
    if expected != observed:
        raise ValueError("Frozen expert artifacts changed")
    captures = [[] for _ in predictor.mixes]
    mixtures = []
    for index, mixture in enumerate(predictor.mixes):
        original_predict = mixture.predict
        def capture(experts, context, *, original_predict=original_predict, index=index):
            captures[index].append(np.array(experts, copy=True))
            return original_predict(experts, context)
        mixture.predict = capture
        context = np.zeros((1, len(mixture.mean)))
        _, weights = mixture.distribution(mixture.theta, np.full((1, 3), .5), mixture.design(context))
        if mixture.kind == "equal_mixture":
            weights = np.full((1, 3), 1/3)
        if weights is None or not mixture.kind.endswith("mixture"):
            raise ValueError("Expected frozen probability-mixture artifact")
        mixtures.append({"kind":mixture.kind, "weights":weights[0].tolist(), "context":"actual native inference uses a constant zero context"})
    predictions = []
    for start in range(0, len(targets), 128):
        indices = np.arange(start, min(start+128, len(targets)))
        raw, history = replay._batch_features(arrays.raw, arrays.hist, indices)
        predictions.append(predictor.predict(raw, history, targets.iloc[indices].best_of.to_numpy(int)))
    predicted = np.concatenate(predictions)
    captured = np.concatenate(captures[0])
    expert_frame = targets[["origin_id", "fixture_id", "golgg_match_id", "horizon", "date", "best_of"]].copy()
    expert_frame["p_reproduced"] = predicted[:, 0]
    for index, expert in enumerate(EXPERTS):
        expert_frame[f"p_{expert}"] = captured[:, index]
    weights = np.asarray(mixtures[0]["weights"])
    if not np.allclose(captured @ weights, predicted[:, 0], atol=1e-12, rtol=0):
        raise ValueError("Captured experts do not reconstruct the frozen final probability")
    compare = expert_frame.merge(primary[["origin_id", "p", "y", "q_consensus_a", "month", "edition_proxy", "competition_family"]], on="origin_id", validate="one_to_one")
    max_difference = float(abs(compare.p-compare.p_reproduced).max())
    if max_difference > 2e-6:
        raise ValueError(f"Expert capture changed repaired forecasts: {max_difference}")
    compare.to_parquet(output / "expert_predictions.parquet", index=False)
    scores = []
    full_loss = binary_log_loss_vector(compare.y, compare.p, epsilon=1e-12)
    market_loss = binary_log_loss_vector(compare.y, compare.q_consensus_a, epsilon=1e-12)
    for expert in EXPERTS:
        loss = binary_log_loss_vector(compare.y, compare[f"p_{expert}"], epsilon=1e-12)
        scores.append({"expert":expert, "n":len(compare), "log_loss":float(loss.mean()), "full_log_loss":float(full_loss.mean()),
            "market_log_loss":float(market_loss.mean()),
            "minus_full_interval":block_ratio_interval(loss-full_loss, np.ones(len(compare)), compare.month, 5000, 20260930),
            "minus_market_interval":block_ratio_interval(loss-market_loss, np.ones(len(compare)), compare.month, 5000, 20260930)})
    bet_results, bet_tables = [], []
    for policy in ("raw_ev_gt_5", "current_application_filters"):
        bets = tables["bets"].loc[lambda frame:
            frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price") & frame.horizon.eq("T-1h")
            & frame.quality.eq("exclude_q25_conflicts") & frame.policy.eq(policy)].merge(
                compare[["origin_id", *[f"p_{name}" for name in EXPERTS]]], on="origin_id", validate="one_to_one")
        contributions = []
        for index, expert in enumerate(EXPERTS):
            probability = np.where(bets.side.eq("A"), bets[f"p_{expert}"], 1-bets[f"p_{expert}"])
            error = (probability-bets.won.astype(float).to_numpy()) * bets.odds.to_numpy(float) * .88
            contribution = weights[index]*error
            bets[f"p_{expert}_side"] = probability
            bets[f"ev_error_{expert}_weighted"] = contribution
            contributions.append(contribution)
            aligned = primary[["origin_id", "month"]].merge(pd.DataFrame({"origin_id":bets.origin_id,"value":error}), on="origin_id", how="left", validate="one_to_one")
            present = aligned.value.notna()
            bet_results.append({"policy":policy, "expert":expert, "n":len(bets), "mixture_weight":float(weights[index]),
                "mean_selected_probability":float(probability.mean()), "win_rate":float(bets.won.mean()),
                "ev_error":float(error.mean()), "weighted_ev_error":float(contribution.mean()),
                "ev_error_interval":block_ratio_interval(aligned.value.fillna(0), present.astype(float), aligned.month, 5000, 20260930)})
        final_error = .88*bets.odds.to_numpy(float)*(bets.p_bet.to_numpy(float)-bets.won.to_numpy(float))
        if not np.allclose(np.sum(contributions, axis=0), final_error, atol=2e-7, rtol=0):
            raise ValueError("Weighted expert selected-EV decomposition failed")
        bets["all_experts_above_market"] = np.logical_and.reduce([bets[f"p_{name}_side"].to_numpy(float)>bets.q_bet.to_numpy(float) for name in EXPERTS])
        bet_tables.append(bets)
    saved_bets = pd.concat(bet_tables, ignore_index=True)
    saved_bets.to_parquet(output / "expert_fixed_bets.parquet", index=False)
    pd.json_normalize(scores).to_csv(output / "expert_scores.csv", index=False)
    pd.json_normalize(bet_results).to_csv(output / "expert_selected_errors.csv", index=False)
    source_hashes = {"src/analysis/ev_asof_replay.py":sha256(ROOT / "src/analysis/ev_asof_replay.py"),
        "native_predictor":sha256(root / replay.CAUSAL_ARCHIVE_RELATIVE / "code/a0_predictor.py")}
    result = {"status":"COMPLETE_FROZEN_EXPERT_CAPTURE", "n":len(compare), "max_abs_final_probability_difference":max_difference,
        "mixtures":mixtures, "scores":scores, "selected":bet_results,
        "all_experts_above_market_counts":saved_bets.groupby("policy").all_experts_above_market.agg(["sum","count"]).to_dict("index"),
        "source_hashes":source_hashes, "model_hashes":observed, "bank_hashes":bank_hashes, "replay_validation":arrays.validation_receipt,
        "limitations":["No expert fitting or new selections; this captures the native frozen predictor's actual three group inputs.",
            "Expert weights and calibrators were trained together; replacing a group is not equivalent to retraining its competitors.",
            "Shared-error agreement does not establish that the experts are statistically independent or identify a missing causal variable."]}
    save_json(output / "capture_receipt.json", result)
    return result
