"""Execution runner and protocol orchestration for champion-pool experimentation (IDEA-031)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from src.analysis.bootstrap import block_ratio_interval
from src.features.champion_pool_features import (
    CHAMPION_FEATURE_NAMES,
    DIFF_FEATURE_NAMES,
    extract_champion_pool_features,
    verify_champion_features_invariants,
)
from src.models.a0_champion_model import (
    SPECIALIST_FEATURE_NAMES,
    ChampionResidualConnector,
    ChampionSpecialist,
    build_specialist_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAN08_CONTROL_PATH = PROJECT_ROOT / "data/07_model_output/model_reliability/model_reliability_20261001_055321/neural/2026/control.parquet"
C0_CAPACITY_PRED_PATH = PROJECT_ROOT / "data/07_model_output/a0_capacity/native_capacity_20261004_01/predictions.parquet"
C0_CAPACITY_CAL_PATH = PROJECT_ROOT / "data/07_model_output/a0_capacity/native_capacity_20261004_01/calibration_predictions.parquet"
ANNUAL_A0_PRED_PATH = PROJECT_ROOT / "data/07_model_output/a0_sports_correction/c0_raw_connector_20261002_033658/predictions.parquet"

PINNACLE_ACCEPTED_PATH = PROJECT_ROOT / "data/05_model_input/betting_ev_reliability/c0_betting_20261002_085551/opportunities_v2.parquet"
PINNACLE_SELECTED_PATH = PROJECT_ROOT / "data/05_model_input/betting_ev_reliability/c0_betting_20261002_085551/selected_v2.parquet"
PINNACLE_INVENTORY_PATH = PROJECT_ROOT / "data/05_model_input/betting_ev_reliability/c0_betting_20261002_085551/opportunity_inventory_v2.parquet"

ALPHA = 0.05
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261005


def resolve_research_root(explicit: Path | None = None) -> Path:
    """Resolve research root path following AGENTS.md conventions."""
    if explicit is not None:
        return explicit.resolve()
    env_val = os.environ.get("ENSEMBLE_RESEARCH_ROOT")
    if env_val:
        return Path(env_val).resolve()
    root_file = PROJECT_ROOT / "data" / "research_root.txt"
    if root_file.is_file():
        text = root_file.read_text(encoding="utf-8").strip()
        if text:
            return Path(text).resolve()
    raise FileNotFoundError("Could not resolve research root via --research-root, ENSEMBLE_RESEARCH_ROOT, or data/research_root.txt")


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/Pandas objects to JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        num = float(value)
        return num if math.isfinite(num) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def paired_monthly_bootstrap(
    delta_vector: np.ndarray,
    months: list[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Paired monthly-block bootstrap matching canonical research standards.

    Computes both match-weighted and equal-month estimands.
    """
    df = pd.DataFrame({"delta": delta_vector, "month": months})
    unique_months = sorted(df["month"].unique())
    n_blocks = len(unique_months)
    n_total = len(df)

    if n_blocks == 0 or n_total == 0:
        return {
            "delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "draws": draws,
            "seed": seed,
            "blocks": n_blocks,
            "n": n_total,
            "equal_month_delta": 0.0,
            "equal_month_ci_lower": 0.0,
            "equal_month_ci_upper": 0.0,
        }

    grouped = df.groupby("month")["delta"].agg(d_sum="sum", d_count="count", d_mean="mean").loc[unique_months]
    sums = grouped["d_sum"].to_numpy(dtype=np.float64)
    counts = grouped["d_count"].to_numpy(dtype=np.float64)
    means = grouped["d_mean"].to_numpy(dtype=np.float64)

    observed_match_weighted = float(sums.sum() / counts.sum())
    observed_equal_month = float(means.mean())

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n_blocks, size=(draws, n_blocks))

    # Match-weighted draws
    draw_sums = sums[sampled_indices].sum(axis=1)
    draw_counts = counts[sampled_indices].sum(axis=1)
    mw_draws = draw_sums / draw_counts

    # Equal-month draws
    em_draws = means[sampled_indices].mean(axis=1)

    alpha_tail = (1.0 - confidence_level) / 2.0
    return {
        "delta": observed_match_weighted,
        "ci_lower": float(np.quantile(mw_draws, alpha_tail)),
        "ci_upper": float(np.quantile(mw_draws, 1.0 - alpha_tail)),
        "draws": draws,
        "seed": seed,
        "blocks": n_blocks,
        "n": n_total,
        "confidence_level": confidence_level,
        "equal_month_delta": observed_equal_month,
        "equal_month_ci_lower": float(np.quantile(em_draws, alpha_tail)),
        "equal_month_ci_upper": float(np.quantile(em_draws, 1.0 - alpha_tail)),
    }


def freeze_run(run_id: str, research_root: Path | None = None) -> Path:
    """Freeze protocol for champion-augmented experiment."""
    root = resolve_research_root(research_root)
    bank_dir = root / "a0-phase-walkforward-20260914" / "data" / "04_feature" / "release_corrected_bank"
    meta_path = bank_dir / "metadata.parquet"
    champs_path = bank_dir / "champions.npy"
    mask_path = bank_dir / "mask.npy"
    arrays_path = bank_dir / "arrays.npz"

    features_npz_path = PROJECT_ROOT / "data" / "04_feature" / "champion_pool" / "features.npz"
    if not features_npz_path.is_file():
        raise FileNotFoundError(f"Champion features not precomputed at {features_npz_path}. Run scripts/prepare_champion_pool_features.py first.")

    # Check baseline files
    for p in (PLAN08_CONTROL_PATH, C0_CAPACITY_PRED_PATH, C0_CAPACITY_CAL_PATH, ANNUAL_A0_PRED_PATH):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required baseline file: {p}")

    meta = pd.read_parquet(meta_path)
    dates = meta["date"].values
    train_mask = (dates < "2025-07-01")
    cal_mask = (dates >= "2025-07-01") & (dates < "2026-01-01")
    assess_mask = (dates >= "2026-01-01") & (dates <= "2026-09-08")

    train_rows = int(train_mask.sum())
    cal_rows = int(cal_mask.sum())
    assess_rows = int(assess_mask.sum())
    assert train_rows == 36205, f"Expected 36205 train rows, got {train_rows}"
    assert cal_rows == 1328, f"Expected 1328 cal rows, got {cal_rows}"
    assert assess_rows == 3103, f"Expected 3103 assess rows, got {assess_rows}"

    source_hashes = {
        str(meta_path): compute_sha256(meta_path),
        str(champs_path): compute_sha256(champs_path),
        str(mask_path): compute_sha256(mask_path),
        str(arrays_path): compute_sha256(arrays_path),
        str(features_npz_path): compute_sha256(features_npz_path),
        str(PLAN08_CONTROL_PATH): compute_sha256(PLAN08_CONTROL_PATH),
        str(C0_CAPACITY_PRED_PATH): compute_sha256(C0_CAPACITY_PRED_PATH),
        str(C0_CAPACITY_CAL_PATH): compute_sha256(C0_CAPACITY_CAL_PATH),
        str(ANNUAL_A0_PRED_PATH): compute_sha256(ANNUAL_A0_PRED_PATH),
    }

    if PINNACLE_SELECTED_PATH.is_file():
        source_hashes[str(PINNACLE_SELECTED_PATH)] = compute_sha256(PINNACLE_SELECTED_PATH)
    if PINNACLE_ACCEPTED_PATH.is_file():
        source_hashes[str(PINNACLE_ACCEPTED_PATH)] = compute_sha256(PINNACLE_ACCEPTED_PATH)

    protocol = {
        "schema": "a0_champion_experiment_v1",
        "run_id": run_id,
        "status": "FROZEN",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "partitions": {
            "train": {"end": "2025-07-01", "rows": train_rows},
            "cal": {"start": "2025-07-01", "end": "2026-01-01", "rows": cal_rows},
            "assess": {"start": "2026-01-01", "end": "2026-09-08", "rows": assess_rows},
        },
        "model_spec": {
            "specialist": {
                "algorithm": "LightGBM",
                "num_leaves": 15,
                "max_depth": 4,
                "learning_rate": 0.03,
                "min_child_samples": 50,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
                "n_estimators": 150,
                "random_state": 20261005,
                "features_count": 42,
            },
            "connector": {
                "offset": 0.0,
                "gamma_bounds": [0.0, 0.5],
                "optimizer": "bounded_brent_minimize_scalar",
            },
            "modes": ["full", "no_w20", "no_organization"],
        },
        "decision_gates": {
            "primary_contrast": {
                "metric": "log_loss",
                "ci_upper_threshold": 0.0,
                "brier_point_threshold": 0.181034493,
                "confidence_level": 0.95,
            },
            "sports_guards": {
                "noninferiority_threshold": 0.005,
                "subsets": ["best_of_3", "best_of_5", "major", "source_unseen"],
                "comparators": ["c0", "a0"],
            },
            "side_symmetry_tolerance": 1.0e-6,
            "cold_start_neutral_tolerance": 1.0e-4,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "blocking": "paired_calendar_month_blocks",
        },
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    out_dir = PROJECT_ROOT / "data" / "05_model_input" / "a0_champion" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_path = out_dir / "protocol.json"
    with open(proto_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    print(f"Protocol frozen at {proto_path}")
    return proto_path


def run_protocol(
    protocol_path: Path | str,
    smoke: bool = False,
    research_root: Path | None = None,
) -> dict[str, Any]:
    """Execute training, calibration blending, assessment, and reporting."""
    t0 = time.time()
    proto_file = Path(protocol_path).resolve()
    if not proto_file.is_file():
        raise FileNotFoundError(f"Protocol file not found: {proto_file}")

    with open(proto_file, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    run_id = protocol["run_id"]
    if smoke:
        run_id = f"{run_id}_smoke"
        print(f"Running SMOKE mode for {run_id}...")

    root = resolve_research_root(research_root)
    bank_dir = root / "a0-phase-walkforward-20260914" / "data" / "04_feature" / "release_corrected_bank"
    meta_path = bank_dir / "metadata.parquet"
    arrays_path = bank_dir / "arrays.npz"
    features_npz_path = PROJECT_ROOT / "data" / "04_feature" / "champion_pool" / "features.npz"

    # Step 1: Verify baseline C0 outputs on 2026 assessment
    print("Verifying baseline C0 parity...")
    c0_cap_df = pd.read_parquet(C0_CAPACITY_PRED_PATH)
    plan08_df = pd.read_parquet(PLAN08_CONTROL_PATH)

    p_c0_cap = c0_cap_df["w32_e12__full"].to_numpy(dtype=np.float64)
    p_c0_plan08 = plan08_df["p_full_full"].to_numpy(dtype=np.float64)
    parity_diff = float(np.abs(p_c0_cap - p_c0_plan08).max())
    if parity_diff > 2.0e-6:
        raise ValueError(f"Baseline C0 parity violation: max diff = {parity_diff} > 2.0e-6")
    print(f"Baseline C0 parity verified: max diff = {parity_diff:.2e}")

    # Step 2: Load feature arrays and metadata
    print("Loading feature banks and metadata...")
    meta = pd.read_parquet(meta_path)
    dates = meta["date"].values
    y = meta["y"].to_numpy(dtype=np.float64)

    train_mask = (dates < "2025-07-01")
    cal_mask = (dates >= "2025-07-01") & (dates < "2026-01-01")
    assess_mask = (dates >= "2026-01-01") & (dates <= "2026-09-08")

    features_npz = np.load(features_npz_path)
    diff_champ = features_npz["diff_features"]

    arr = np.load(arrays_path)
    players = arr["players"]

    print("Building specialist features (30 champion diffs + 12 rating diffs)...")
    X_all = build_specialist_features(diff_champ, players)

    if smoke:
        # Reduced toy slices for smoke
        train_idx = np.where(train_mask)[0][:500]
        cal_idx = np.where(cal_mask)[0][:100]
        assess_idx = np.where(assess_mask)[0][:100]
    else:
        train_idx = np.where(train_mask)[0]
        cal_idx = np.where(cal_mask)[0]
        assess_idx = np.where(assess_mask)[0]

    X_train, y_train = X_all[train_idx], y[train_idx]
    X_cal, y_cal = X_all[cal_idx], y[cal_idx]
    X_assess, y_assess = X_all[assess_idx], y[assess_idx]

    # Step 3: Train ChampionSpecialist
    print(f"Training ChampionSpecialist on {len(X_train)} series with symmetric reflection...")
    spec_params = protocol["model_spec"]["specialist"]
    if smoke:
        spec_params = dict(spec_params)
        spec_params["n_estimators"] = 10

    specialist = ChampionSpecialist(
        num_leaves=spec_params["num_leaves"],
        max_depth=spec_params["max_depth"],
        learning_rate=spec_params["learning_rate"],
        min_child_samples=spec_params["min_child_samples"],
        colsample_bytree=spec_params["colsample_bytree"],
        subsample=spec_params["subsample"],
        n_estimators=spec_params["n_estimators"],
        random_state=spec_params["random_state"],
    )
    specialist.fit(X_train, y_train)

    # Step 4: Symmetrized prediction on CAL and Assessment
    print("Generating specialist predictions...")
    p_champ_cal = specialist.predict_proba_sym(X_cal)
    p_champ_assess = specialist.predict_proba_sym(X_assess)

    # Step 5: Load C0 baseline probabilities for CAL and Assessment
    cal_preds_df = pd.read_parquet(C0_CAPACITY_CAL_PATH)
    p_c0_cal = {
        "full": cal_preds_df["w32_e12__full"].to_numpy(dtype=np.float64)[cal_idx - cal_mask.argmax() if smoke else slice(None)],
        "no_w20": cal_preds_df["w32_e12__no_w20"].to_numpy(dtype=np.float64)[cal_idx - cal_mask.argmax() if smoke else slice(None)],
        "no_organization": cal_preds_df["w32_e12__no_organization"].to_numpy(dtype=np.float64)[cal_idx - cal_mask.argmax() if smoke else slice(None)],
    }

    assess_c0_df = pd.read_parquet(C0_CAPACITY_PRED_PATH)
    p_c0_assess = {
        "full": assess_c0_df["w32_e12__full"].to_numpy(dtype=np.float64)[assess_idx - assess_mask.argmax() if smoke else slice(None)],
        "no_w20": assess_c0_df["w32_e12__no_w20"].to_numpy(dtype=np.float64)[assess_idx - assess_mask.argmax() if smoke else slice(None)],
        "no_organization": assess_c0_df["w32_e12__no_organization"].to_numpy(dtype=np.float64)[assess_idx - assess_mask.argmax() if smoke else slice(None)],
    }

    # Step 6: Fit ChampionResidualConnector on CAL
    print("Fitting ChampionResidualConnector gamma on CAL...")
    connector = ChampionResidualConnector(offset=protocol["model_spec"]["connector"]["offset"])
    connector.fit_cal(p_c0_cal, p_champ_cal, y_cal)
    print(f"Optimal gammas on CAL: {connector.gammas_}")
    print(f"CAL log losses: {connector.cal_losses_} vs baseline: {connector.baseline_cal_losses_}")

    # Step 7: Predict blended candidate on Assessment
    print("Blending candidate predictions on Assessment...")
    p_cand_assess = connector.predict(p_c0_assess, p_champ_assess)

    # Step 8: Symmetry & Cold Start verification
    print("Checking symmetry and cold start invariants...")
    p_pos_test = specialist.predict_proba_sym(X_assess[:50])
    p_neg_test = specialist.predict_proba_sym(-X_assess[:50])
    symm_err = float(np.abs(p_pos_test + p_neg_test - 1.0).max())
    assert symm_err <= protocol["decision_gates"]["side_symmetry_tolerance"], f"Symmetry violation: {symm_err}"

    p_cold = specialist.predict_proba_sym(np.zeros((10, 42), dtype=np.float32))
    cold_err = float(np.abs(p_cold - 0.5).max())
    assert cold_err <= protocol["decision_gates"]["cold_start_neutral_tolerance"], f"Cold start violation: {cold_err}"

    # Step 9: Build assessment DataFrame and metrics
    meta_assess = meta.iloc[assess_idx].copy().reset_index(drop=True)
    assess_dates = meta_assess["date"].to_numpy()
    assess_months = [pd.to_datetime(d).strftime("%Y-%m") for d in assess_dates]
    y_true = y_assess

    # Load annual A0 for comparisons
    annual_a0_df = pd.read_parquet(ANNUAL_A0_PRED_PATH)
    annual_a0_2026 = annual_a0_df[annual_a0_df["date"] >= "2026-01-01"].reset_index(drop=True)
    a0_col = "p_full" if "p_full" in annual_a0_2026.columns else "p_a0_full"
    p_a0_full = annual_a0_2026[a0_col].to_numpy(dtype=np.float64)[assess_idx - assess_mask.argmax() if smoke else slice(None)]
    # Compute log loss and Brier for candidate and C0
    p_cand_full = p_cand_assess["full"]
    p_c0_full = p_c0_assess["full"]

    ll_cand = float(-(y_true * np.log(p_cand_full) + (1.0 - y_true) * np.log(1.0 - p_cand_full)).mean())
    ll_c0 = float(-(y_true * np.log(p_c0_full) + (1.0 - y_true) * np.log(1.0 - p_c0_full)).mean())
    ll_a0 = float(-(y_true * np.log(p_a0_full) + (1.0 - y_true) * np.log(1.0 - p_a0_full)).mean())

    brier_cand = float(((p_cand_full - y_true) ** 2).mean())
    brier_c0 = float(((p_c0_full - y_true) ** 2).mean())
    brier_a0 = float(((p_a0_full - y_true) ** 2).mean())

    # Bootstrap primary contrast: cand - C0
    delta_ll_vec = (-(y_true * np.log(p_cand_full) + (1.0 - y_true) * np.log(1.0 - p_cand_full))) - (
        -(y_true * np.log(p_c0_full) + (1.0 - y_true) * np.log(1.0 - p_c0_full))
    )
    delta_brier_vec = ((p_cand_full - y_true) ** 2) - ((p_c0_full - y_true) ** 2)

    bs_ll = paired_monthly_bootstrap(delta_ll_vec, assess_months, draws=BOOTSTRAP_DRAWS if not smoke else 500)
    bs_brier = paired_monthly_bootstrap(delta_brier_vec, assess_months, draws=BOOTSTRAP_DRAWS if not smoke else 500)

    # Primary contrast gate checks:
    # 1. 95% bootstrap CI upper bound strictly below 0
    # 2. Point Brier score strictly lower than C0 (0.181034493)
    primary_pass = (bs_ll["ci_upper"] < 0.0) and (brier_cand < protocol["decision_gates"]["primary_contrast"]["brier_point_threshold"])

    # Step 10: Compute sports guards on subsets
    # Subsets: best_of_3, best_of_5, major, source_unseen
    # Load tier and unknown_players
    tiers = assess_c0_df["competition_tier"].to_numpy()[assess_idx - assess_mask.argmax() if smoke else slice(None)]
    best_ofs = assess_c0_df["best_of"].to_numpy()[assess_idx - assess_mask.argmax() if smoke else slice(None)]
    unknowns = assess_c0_df["unknown_players"].to_numpy()[assess_idx - assess_mask.argmax() if smoke else slice(None)]

    subset_masks = {
        "best_of_3": (best_ofs == 3),
        "best_of_5": (best_ofs == 5),
        "major": (tiers == "major"),
        "source_unseen": (unknowns > 0),
    }

    guards: dict[str, Any] = {}
    guards_all_pass = True
    guard_threshold = protocol["decision_gates"]["sports_guards"]["noninferiority_threshold"]

    for s_name, s_mask in subset_masks.items():
        n_sub = int(s_mask.sum())
        if n_sub == 0:
            guards[s_name] = {"pass": True, "n": 0}
            continue

        s_months = [assess_months[i] for i in range(len(s_mask)) if s_mask[i]]
        # Candidate vs C0
        d_c0_sub = delta_ll_vec[s_mask]
        bs_c0_sub = paired_monthly_bootstrap(d_c0_sub, s_months, draws=BOOTSTRAP_DRAWS if not smoke else 500)
        pass_c0 = bs_c0_sub["ci_upper"] <= guard_threshold

        # Candidate vs A0
        d_a0_sub = (-(y_true[s_mask] * np.log(p_cand_full[s_mask]) + (1.0 - y_true[s_mask]) * np.log(1.0 - p_cand_full[s_mask]))) - (
            -(y_true[s_mask] * np.log(p_a0_full[s_mask]) + (1.0 - y_true[s_mask]) * np.log(1.0 - p_a0_full[s_mask]))
        )
        bs_a0_sub = paired_monthly_bootstrap(d_a0_sub, s_months, draws=BOOTSTRAP_DRAWS if not smoke else 500)
        pass_a0 = bs_a0_sub["ci_upper"] <= guard_threshold

        sub_pass = pass_c0 and pass_a0
        if not sub_pass:
            guards_all_pass = False

        guards[s_name] = {
            "n": n_sub,
            "pass": sub_pass,
            "comparators": {
                "c0": {
                    "pass": pass_c0,
                    "threshold": guard_threshold,
                    "log_loss": bs_c0_sub,
                },
                "a0": {
                    "pass": pass_a0,
                    "threshold": guard_threshold,
                    "log_loss": bs_a0_sub,
                },
            },
        }

    # Step 11: Decoupled EV Selection Audit
    market_diagnostic = None
    ev_decomp_rows = []
    if not smoke:
        print("Executing decoupled EV selection audit on Pinnacle fixtures...")
        cap_proto_path = PROJECT_ROOT / "data/05_model_input/a0_capacity/native_capacity_20261004_01/protocol.json"
        if cap_proto_path.is_file():
            with open(cap_proto_path, "r", encoding="utf-8") as f:
                cap_proto = json.load(f)
            dm_spec = cap_proto.get("diagnostic_market", {})
            sel_rows = dm_spec.get("selection_rows", [])
            inv_rows = dm_spec.get("inventory_rows", [])
        else:
            sel_rows, inv_rows = [], []

        if sel_rows and inv_rows:
            sel_df = pd.DataFrame(sel_rows)
            inv_df = pd.DataFrame(inv_rows)

            pred_map = dict(zip(meta_assess["golgg_match_id"].astype(str), p_cand_full))
            c0_map = dict(zip(meta_assess["golgg_match_id"].astype(str), p_c0_full))

            sel_df["candidate_a"] = sel_df["golgg_match_id"].astype(str).map(pred_map)
            sel_df["c0_a"] = sel_df["golgg_match_id"].astype(str).map(c0_map)

            is_side_a = sel_df["side"].astype(str).str.upper() == "A"
            cand_selected = np.where(is_side_a, sel_df["candidate_a"], 1.0 - sel_df["candidate_a"])
            c0_selected = np.where(is_side_a, sel_df["c0_a"], 1.0 - sel_df["c0_a"])

            odds = sel_df["odds"].to_numpy(dtype=float)
            q = sel_df["market_q_chosen"].to_numpy(dtype=float)
            won = sel_df["won"].to_numpy(dtype=float)

            tau = 0.12
            # Candidate
            est_ev_cand = (1.0 - tau) * odds * cand_selected - 1.0
            real_ret = (1.0 - tau) * odds * won - 1.0
            ev_err_cand = est_ev_cand - real_ret
            mod_mkt_cand = (1.0 - tau) * odds * (cand_selected - q)
            mkt_res = (1.0 - tau) * odds * (q - won)
            assert np.allclose(ev_err_cand, mod_mkt_cand + mkt_res, atol=1e-10)

            # C0
            est_ev_c0 = (1.0 - tau) * odds * c0_selected - 1.0
            ev_err_c0 = est_ev_c0 - real_ret
            mod_mkt_c0 = (1.0 - tau) * odds * (c0_selected - q)
            assert np.allclose(ev_err_c0, mod_mkt_c0 + mkt_res, atol=1e-10)

            sel_df["estimated_ev"] = est_ev_cand
            sel_df["realized_return"] = real_ret
            sel_df["term_model_market"] = mod_mkt_cand
            sel_df["term_market_residual"] = mkt_res
            sel_df["ev_error"] = ev_err_cand

            inv_dates = pd.to_datetime(inv_df["date"]).dt.to_period("M").astype(str).to_numpy()
            row_to_pos = {r: i for i, r in enumerate(inv_df["row_id"].astype(str))}
            pos = sel_df["row_id"].astype(str).map(row_to_pos).to_numpy(int)
            near_mask = sel_df["near_band"].values.astype(bool)

            def calc_intervals(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
                num = np.zeros(len(inv_df), dtype=float)
                den = np.zeros(len(inv_df), dtype=float)
                num[pos[mask]] = values[mask]
                den[pos[mask]] = 1.0
                return block_ratio_interval(num, den, inv_dates, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)

            all_mask = np.ones(len(sel_df), dtype=bool)
            market_diagnostic = {
                "status": "COMPLETED_DIAGNOSTIC",
                "g0_status": "BLOCKED",
                "inventory_count": len(inv_df),
                "accepted_fixture_count": 1615,
                "raw_ev_selected_count": len(sel_df),
                "near_band_count": int(near_mask.sum()),
                "policy_tuned": False,
                "by_model": {
                    "champion_specialist_c0": {
                        "selected": {
                            "ev_error": calc_intervals(ev_err_cand, all_mask),
                            "model_market_component": calc_intervals(mod_mkt_cand, all_mask),
                            "market_residual_component": calc_intervals(mkt_res, all_mask),
                        },
                        "selected_near_5_10": {
                            "ev_error": calc_intervals(ev_err_cand, near_mask),
                            "model_market_component": calc_intervals(mod_mkt_cand, near_mask),
                            "market_residual_component": calc_intervals(mkt_res, near_mask),
                        },
                    },
                    "baseline_c0": {
                        "selected": {
                            "ev_error": calc_intervals(ev_err_c0, all_mask),
                            "model_market_component": calc_intervals(mod_mkt_c0, all_mask),
                            "market_residual_component": calc_intervals(mkt_res, all_mask),
                        },
                        "selected_near_5_10": {
                            "ev_error": calc_intervals(ev_err_c0, near_mask),
                            "model_market_component": calc_intervals(mod_mkt_c0, near_mask),
                            "market_residual_component": calc_intervals(mkt_res, near_mask),
                        },
                    },
                },
            }
            ev_decomp_rows = sel_df.to_dict(orient="records")
    # Step 12: Overall decision
    adopted = bool(primary_pass and guards_all_pass and (parity_diff <= 2.0e-6))
    decision = {
        "status": "ADOPTED" if adopted else "NOT_ADOPTED",
        "adopted": adopted,
        "complete_assessment_coverage": bool(len(p_cand_full) == 3103 if not smoke else True),
        "primary_contrast": {
            "pass": primary_pass,
            "delta_ll": bs_ll,
            "delta_brier": bs_brier,
        },
        "sports_guards": {
            "all_pass": guards_all_pass,
            "subsets": guards,
        },
        "candidate_gates": {
            "champion_specialist_c0": {
                "adopted": adopted,
                "gammas": connector.gammas_,
                "log_loss": ll_cand,
                "brier": brier_cand,
            }
        },
    }

    # Format outputs
    metrics_summary = {
        "candidate": {
            "log_loss": ll_cand,
            "brier": brier_cand,
        },
        "baseline_c0": {
            "log_loss": ll_c0,
            "brier": brier_c0,
        },
        "baseline_a0": {
            "log_loss": ll_a0,
            "brier": brier_a0,
        },
        "contrasts": {
            "champ_minus_c0": {
                "log_loss": bs_ll,
                "brier": bs_brier,
                "pass": primary_pass,
            }
        },
        "guards": guards,
        "decision": decision,
    }

    if not smoke:
        # Save model artifacts
        model_out_dir = PROJECT_ROOT / "data" / "06_models" / "a0_champion" / run_id
        model_out_dir.mkdir(parents=True, exist_ok=True)
        specialist.save(model_out_dir / "specialist.joblib")
        connector.save(model_out_dir / "connector.json")

        # Save predictions
        pred_out_dir = PROJECT_ROOT / "data" / "07_model_output" / "a0_champion" / run_id
        pred_out_dir.mkdir(parents=True, exist_ok=True)
        pred_df = meta_assess[["golgg_match_id", "date", "y", "best_of", "tournament_name"]].copy()
        pred_df["p_champ_full"] = p_cand_assess["full"]
        pred_df["p_champ_no_w20"] = p_cand_assess["no_w20"]
        pred_df["p_champ_no_organization"] = p_cand_assess["no_organization"]
        pred_df["p_c0_full"] = p_c0_assess["full"]
        pred_df["p_specialist_standalone"] = p_champ_assess
        pred_df.to_parquet(pred_out_dir / "predictions.parquet", index=False)

        # Save reporting outputs
        rep_out_dir = PROJECT_ROOT / "data" / "08_reporting" / "a0_champion" / run_id
        rep_out_dir.mkdir(parents=True, exist_ok=True)

        with open(rep_out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(metrics_summary), f, indent=2)

        with open(rep_out_dir / "decision.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe({"decision": decision, "market_diagnostic": market_diagnostic}), f, indent=2)

        # Contrasts CSV
        contrasts_df = pd.DataFrame([
            {
                "contrast": "champ_minus_c0",
                "candidate": "champion_specialist_c0",
                "reference": "baseline_c0",
                "primary_contrast_pass": primary_pass,
                "candidate_adopted": adopted,
                "log_loss_delta": bs_ll["delta"],
                "log_loss_ci_lower": bs_ll["ci_lower"],
                "log_loss_ci_upper": bs_ll["ci_upper"],
                "brier_delta": bs_brier["delta"],
                "brier_ci_lower": bs_brier["ci_lower"],
                "brier_ci_upper": bs_brier["ci_upper"],
            }
        ])
        contrasts_df.to_csv(rep_out_dir / "contrasts.csv", index=False)

        # EV decomposition CSV
        if ev_decomp_rows:
            pd.DataFrame(ev_decomp_rows).to_csv(rep_out_dir / "selected_ev_decomposition.csv", index=False)

        # Receipt
        receipt = {
            "status": "COMPLETED",
            "run_id": run_id,
            "elapsed_seconds": round(time.time() - t0, 3),
            "decision": decision["status"],
            "adopted": adopted,
            "primary_contrast_ci_upper": bs_ll["ci_upper"],
            "brier_candidate": brier_cand,
            "brier_c0": brier_c0,
            "gammas": connector.gammas_,
            "output_artifacts": {
                "model": str(model_out_dir),
                "predictions": str(pred_out_dir / "predictions.parquet"),
                "reporting": str(rep_out_dir),
            },
        }
        with open(rep_out_dir / "receipt.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(receipt), f, indent=2)

    elapsed = time.time() - t0
    print(f"Protocol run complete in {elapsed:.2f}s! Decision: {decision['status']}")
    return _json_safe(metrics_summary)
