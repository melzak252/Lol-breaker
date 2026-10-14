#!/usr/bin/env python3
"""Freeze the authorized EMEA reconstruction population and diagnostic controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "data/05_model_input/model_bookmaker_audit/model_bookmaker_deep_audit_20261009_213835Z/protocol.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(value: object) -> str:
    return str(value)


def support_class(value: int) -> int:
    return 0 if value < 10 else 1 if value < 20 else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--research-root", type=Path)
    args = parser.parse_args()
    scope_path = args.scope.resolve()
    scope = json.loads(scope_path.read_text())
    paths = dict(scope["paths"])
    inputs = ROOT / paths["input_root"]
    protocol_path = inputs / "protocol.json"
    if protocol_path.exists() or (inputs / "selection.parquet").exists():
        raise FileExistsError("Prepared namespaces are immutable; use a fresh authorized run")
    research = args.research_root
    if research is None:
        research = Path(os.environ.get("ENSEMBLE_RESEARCH_ROOT") or (ROOT / "data/research_root.txt").read_text().strip())
    research = research.expanduser().resolve()
    bank_root = research / paths["bank_relative"]
    prior = json.loads(PRIOR.read_text())
    input_pins = dict(prior["input_sha256"])
    for relative, expected in input_pins.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Previous audit input changed: {relative}")
    native = pd.read_parquet(ROOT / paths["native_control"])
    bank = pd.read_parquet(bank_root / "metadata.parquet")
    panel = pd.read_parquet(ROOT / paths["previous_audit_panel"])
    primary = panel.loc[panel.comparison_cohort.eq("Polish2plus")].copy()
    for frame in (native, bank, primary):
        frame["golgg_match_id"] = frame.golgg_match_id.map(key)
        if frame.golgg_match_id.duplicated().any():
            raise ValueError("Duplicate native/bank/primary target identity")
    if len(native) != 3103 or len(primary) != 1467:
        raise ValueError("Frozen native or primary cohort differs from authorized population")
    bank["bank_row_index"] = np.arange(len(bank), dtype=np.int64)
    bank_by_id = bank.set_index("golgg_match_id", drop=False)
    selection = bank_by_id.loc[native.golgg_match_id].copy().reset_index(drop=True)
    native_by_id = native.set_index("golgg_match_id", drop=False)
    for column in bank.columns.difference(["bank_row_index"]):
        for ident, left, right in zip(selection.golgg_match_id, selection[column],
                                      native_by_id.loc[selection.golgg_match_id, column], strict=True):
            if column in ("roster_a", "roster_b"):
                equal = list(left) == list(right)
            elif pd.isna(left) and pd.isna(right):
                equal = True
            else:
                equal = left == right
            if not equal:
                raise ValueError(f"Native/bank metadata mismatch: {ident}/{column}")
    emea = primary.loc[primary.league.eq("EMEA Masters")].copy()
    if len(emea) != 89:
        raise ValueError(f"Expected all89 EMEA primary fixtures, observed {len(emea)}")
    primary_by_id = primary.set_index("golgg_match_id", drop=False)
    for row in primary.itertuples(index=False):
        n = native_by_id.loc[row.golgg_match_id]
        if row.y != n.y or row.p_C0 != n.p_full_full or row.native_orientation_reversed:
            raise ValueError(f"Primary/native orientation or probability drift: {row.golgg_match_id}")
    selection["p_c0"] = native_by_id.loc[selection.golgg_match_id, "p_full_full"].to_numpy()
    selection["native_date"] = selection.date
    selection["is_polish_primary"] = selection.golgg_match_id.isin(primary.golgg_match_id)
    selection["is_emea_primary"] = selection.golgg_match_id.isin(emea.golgg_match_id)
    selection["league"] = selection.golgg_match_id.map(primary_by_id.league).fillna("not_in_primary")
    selection["support_class"] = selection.min_player_series.map(support_class)
    selection["unknown_class"] = selection.unknown_players.gt(0).astype(int)
    selection["native_month"] = pd.to_datetime(selection.date).dt.strftime("%Y-%m")
    dates = pd.to_datetime(selection.date)
    controls = selection.loc[selection.is_polish_primary & ~selection.is_emea_primary].copy()
    links = []
    for target in selection.loc[selection.is_emea_primary].itertuples(index=False):
        same_bo = controls.best_of.eq(target.best_of)
        same_month = controls.native_month.eq(target.native_month)
        same_support = controls.support_class.eq(target.support_class) & controls.unknown_class.eq(target.unknown_class)
        eligible = controls.loc[same_bo & same_month & same_support].copy()
        level = "L1_MONTH_BO_SUPPORT_UNKNOWN"
        if eligible.empty:
            eligible = controls.loc[same_bo & same_month].copy()
            level = "L2_MONTH_BO"
        if eligible.empty:
            distances = (pd.to_datetime(controls.date) - pd.Timestamp(target.date)).dt.days.abs()
            eligible = controls.loc[same_bo & distances.le(31)].copy()
            level = "L3_BO_WITHIN31D"
        if eligible.empty:
            links.append({"emea_golgg_match_id": target.golgg_match_id, "control_golgg_match_id": None,
                          "match_status": "UNMATCHED", "match_level": "NO_ELIGIBLE_CONTROL", "candidate_count": 0})
            continue
        eligible["support_distance"] = (np.log1p(eligible.min_player_series) - np.log1p(target.min_player_series)).abs()
        eligible["confidence_distance"] = (np.maximum(eligible.p_c0, 1 - eligible.p_c0) - max(target.p_c0, 1 - target.p_c0)).abs()
        eligible["day_distance"] = (pd.to_datetime(eligible.date) - pd.Timestamp(target.date)).dt.days.abs()
        eligible["numeric_id"] = eligible.golgg_match_id.map(lambda value: int(value) if value.isdigit() else float("inf"))
        chosen = eligible.sort_values(["support_distance", "confidence_distance", "day_distance", "numeric_id", "golgg_match_id"], kind="stable").iloc[0]
        links.append({"emea_golgg_match_id": target.golgg_match_id, "control_golgg_match_id": chosen.golgg_match_id,
                      "match_status": "MATCHED", "match_level": level, "candidate_count": len(eligible),
                      "support_distance": float(chosen.support_distance), "confidence_distance": float(chosen.confidence_distance),
                      "day_distance": int(chosen.day_distance)})
    link_frame = pd.DataFrame(links)
    selection["is_matched_control"] = selection.golgg_match_id.isin(link_frame.control_golgg_match_id.dropna())
    selection.to_parquet(inputs / "selection.parquet", index=False)
    link_frame.to_csv(inputs / "control_links.csv", index=False)
    model_root = ROOT / paths["native_model_root"]
    protected_paths = [ROOT / paths["previous_audit_seal"], ROOT / paths["previous_audit_panel"],
                       ROOT / "conf/base/research_benchmark.json", model_root / "execution.json",
                       model_root / "shared/model.joblib", model_root / "shared/model.joblib.receipt.json"]
    for name in ("calibration", "seed_20260910", "seed_20260911", "seed_20260912"):
        protected_paths.extend((model_root / "control" / name / "model.joblib", model_root / "control" / name / "model.joblib.receipt.json"))
    for name in ("protocol.json", "source_contracts.json", "verification_recipe.json", "basis.json", "saved_goal.json"):
        protected_paths.append(ROOT / "data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022" / name)
    for name in ("matrix.json", "receipt.json"):
        protected_paths.append(ROOT / "data/08_reporting/a0_context_reliability/source_activation_20261004_01" / name)
    for path in protected_paths:
        input_pins[path.relative_to(ROOT).as_posix()] = sha256(path)
    completion = json.loads((bank_root / "completed.json").read_text())
    bank_pins = dict(completion["outputs_sha256"])
    bank_pins["completed.json"] = sha256(bank_root / "completed.json")
    for name, expected in bank_pins.items():
        actual = sha256(bank_root / name)
        if actual != expected:
            raise ValueError(f"Corrected bank changed: {name}")
        input_pins["research:" + (bank_root / name).relative_to(research).as_posix()] = expected
    for name in ("selection.parquet", "control_links.csv", "scope.json"):
        input_pins[(inputs / name).relative_to(ROOT).as_posix()] = sha256(inputs / name)
    input_pins["scripts/research/prepare_c0_emea_reconstruction.py"] = sha256(Path(__file__))
    paths.update({"bank_metadata": "research:" + (bank_root / "metadata.parquet").relative_to(research).as_posix(),
                  "reconstructed_arrays": paths["feature_root"] + "/tracks/reconstruction/reconstructed_arrays.npz",
                  "reconstructed_targets": paths["feature_root"] + "/tracks/reconstruction/targets.parquet",
                  "reconstructed_probabilities": paths["output_root"] + "/tracks/reconstruction/probabilities.parquet"})
    protocol = {"schema": "c0_emea_reconstruction_v1", "run_id": scope["run_id"], "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "goal": scope["goal"], "paths": paths, "input_sha256": input_pins, "scope_sha256": sha256(scope_path),
                "matching": scope["matching"], "control_match_levels": {"L1_MONTH_BO_SUPPORT_UNKNOWN": "month + BO + supportclass + unknownclass",
                  "L2_MONTH_BO": "relax BOTH supportclass andunknownclass; exactmonth andBO", "L3_BO_WITHIN31D": "exactBO and31calendar days"},
                "native_target_n": 3103, "primary_n": 1467, "emea_primary_n": 89,
                "selection": {"selected_controls": int(selection.is_matched_control.sum()), "control_links": len(link_frame),
                  "matched_links": int(link_frame.match_status.eq("MATCHED").sum()), "levels": link_frame.match_level.value_counts().to_dict(),
                  "outcome_blind": "Selection program neverreads y/market/error to match; nativeC0confidence is observedprematch covariate, not causal identification."},
                "verification": scope["verification"], "conditional_experiment": scope["conditional_experiment"], "boundaries": scope["boundaries"],
                "model_identities": {"C0": "c0-native-2026-w32-e12-v1", "LR": "lr_ts_standard_seed42", "MLP8": "mlp_ts_standard_seed42"},
                "fit_gate_at_freeze": "NOT_EVALUATED_NO_FITS_AUTHORIZED_WITHOUT_SUPPORTED_CAUSE", "new_fits_at_freeze": 0,
                "information_origin": "OriginalC0bankpredictionday; effectivewhole-series release<day; actual firstmap lineups conditional scenario; odds source-last endpoint retained, not authenticated same-origin issuance."}
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    receipt = {"status": "PASS_INPUTS_AND_OUTCOME_BLIND_CONTROL_BINDING", "protocol_sha256": sha256(protocol_path),
               "native_targets": len(selection), "primary": len(primary), "all_emea": len(emea),
               "unique_controls": int(selection.is_matched_control.sum()), "matched_control_links": int(link_frame.match_status.eq("MATCHED").sum()),
               "levels": link_frame.match_level.value_counts().to_dict(), "frozen_inputs": len(input_pins),
               "fits": 0, "native_inference": 0, "source_date_is_not_bank_cutoff": True}
    (inputs / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
