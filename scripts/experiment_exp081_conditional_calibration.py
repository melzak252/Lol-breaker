#!/usr/bin/env python3
"""EXP081 conditional and bounded payoff-moment calibration, research only.

Run with --output-dir NEW_DIRECTORY. No tuning or production adoption occurs.
For raw-sports selected side s in {-1,+1}, quoted odds o_s, and
D=max(1, .88*odds_a, .88*odds_b), use w=s*.88*o_s/D. For each
fixed raw-policy group g, M_g(beta)=sum_g w*(sigmoid(X beta)-y)/n_g.
The payoff objective adds .5*sum_g M_g(beta)**2 to mean BCE+.0005||beta||².
Empty groups contribute zero and are explicitly reported. All moment tests are
bounded by one; unnormalized currency/unit-payoff diagnostics are not tests.
This finite, overlapping moment penalty is NOT multiaccuracy certification or
a profit guarantee; historical inputs have no certified point-in-time status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import experiment_exp081_market_calibration as base
from src.models.competition_tiers import CompetitionTier

MODELS = ("raw_sports", "benter", "conditional", "conditional_payoff", "market_calibrated")
PHASES = ("open", "close")
FEATURE_NAMES = (
    "logit_sports", "logit_market", "logit_difference_times_absolute_disagreement",
    "logit_difference_times_overround",
)
MOMENT_GROUPS = ("raw_selected", "raw_selected_odds_lt_3.5", "raw_selected_odds_ge_3.5")
MIN_OPTIONAL_TRAIN_N = 50
GROUP_DEFINITIONS = {
    "overall": "all phase-eligible opportunities",
    "disagreement_lt_0.10": "abs(raw_sports-raw_market)<0.10",
    "disagreement_ge_0.10": "abs(raw_sports-raw_market)>=0.10",
    "opposite_favorites": "(raw_sports-.5)*(raw_market-.5)<0",
    "raw_chosen_odds_lt_2": "raw-sports maximum-EV side odds<2, including abstentions",
    "raw_chosen_odds_2_to_3.5": "2<=raw-sports maximum-EV side odds<3.5",
    "raw_chosen_odds_3.5_to_5": "3.5<=raw-sports maximum-EV side odds<=5",
    "raw_chosen_odds_gt_5": "raw-sports maximum-EV side odds>5",
    **{f"format_bo{n}": f"snapshot best_of=={n}" for n in (1, 3, 5)},
    "format_other": "snapshot best_of not in {1,3,5}",
    "raw_selected": "fixed raw-sports policy selects, independent of fitted candidate and outcome",
    "raw_selected_odds_lt_3.5": "fixed raw-sports policy selects and selected odds<3.5",
    "raw_selected_odds_ge_3.5": "fixed raw-sports policy selects and selected odds>=3.5",
    **{f"tier_{tier.value}": f"snapshot competition_tier=={tier.value}" for tier in CompetitionTier},
    "roster_rookie": "snapshot roster_min_prior_series<10; existing corrected_history_evaluation definition",
    "roster_stable": "snapshot roster_min_prior_series>=10; existing corrected_history_evaluation definition",
}
LIMITATIONS = [
    item for item in base.LIMITATIONS if not item.startswith("STS close is")
] + [
    "STS open is primary and close is sensitivity; neither has certified decision-time provenance.",
    "Finite normalized payoff moments are overlapping research diagnostics, not certified multiaccuracy or profit.",
    "Squared nonlinear residual moments need not be convex; fixed zero-initialized BFGS finds no guaranteed global optimum.",
    "Group comparisons and interval signs are exploratory, unadjusted for multiplicity, and cannot promote a model.",
    "Earlier-label-only calibration is verified by dates; upstream sports OOF provenance remains conditional on supplied artifacts.",
]


def conditional_design(frame):
    logits = base.design(frame, ("sports", "market"))
    difference = logits[:, 0] - logits[:, 1]
    disagreement = np.abs(frame.sports.to_numpy(float) - frame.market.to_numpy(float))
    overround = 1 / frame.odds_a.to_numpy(float) + 1 / frame.odds_b.to_numpy(float) - 1
    return np.column_stack((logits, difference * disagreement, difference * overround))


def side_context(frame, probabilities):
    selected, side_a = base.policy(probabilities, frame.odds_a, frame.odds_b)
    sign = np.where(side_a, 1.0, -1.0)
    oa, ob = frame.odds_a.to_numpy(float), frame.odds_b.to_numpy(float)
    price = np.where(side_a, oa, ob)
    denominator = np.maximum(1.0, (1 - base.TAX) * np.maximum(oa, ob))
    weights = sign * (1 - base.TAX) * price / denominator
    return selected, sign, price, denominator, weights


def audit_masks(frame):
    selected, _, price, _, _ = side_context(frame, frame.sports)
    disagreement = np.abs(frame.sports.to_numpy(float) - frame.market.to_numpy(float))
    masks = {
        "overall": np.ones(len(frame), dtype=bool),
        "disagreement_lt_0.10": disagreement < .10,
        "disagreement_ge_0.10": disagreement >= .10,
        "opposite_favorites": ((frame.sports - .5) * (frame.market - .5) < 0).to_numpy(),
        "raw_chosen_odds_lt_2": price < 2,
        "raw_chosen_odds_2_to_3.5": (price >= 2) & (price < 3.5),
        "raw_chosen_odds_3.5_to_5": (price >= 3.5) & (price <= 5),
        "raw_chosen_odds_gt_5": price > 5,
        **{f"format_bo{n}": frame.best_of.eq(n).to_numpy() for n in (1, 3, 5)},
        "format_other": (~frame.best_of.isin((1, 3, 5))).to_numpy(),
        "raw_selected": selected,
        "raw_selected_odds_lt_3.5": selected & (price < 3.5),
        "raw_selected_odds_ge_3.5": selected & (price >= 3.5),
    }
    if "competition_tier" in frame:
        masks.update({f"tier_{tier.value}": frame.competition_tier.eq(tier.value).to_numpy()
                      for tier in CompetitionTier})
    if "roster_min_prior_series" in frame:
        masks["roster_rookie"] = frame.roster_min_prior_series.lt(10).to_numpy()
        masks["roster_stable"] = frame.roster_min_prior_series.ge(10).to_numpy()
    return masks


def moment_matrix(frame):
    masks = audit_masks(frame)
    _, _, _, denominator, weights = side_context(frame, frame.sports)
    rows, support = [], []
    for name in MOMENT_GROUPS:
        mask = masks[name]
        n = int(mask.sum())
        rows.append(np.where(mask, weights, 0) / max(n, 1))
        support.append({
            "group": name, "n": n, "denominator_n": n,
            "empty_group_term": "zero" if n == 0 else None,
            "normalizer_min": float(denominator[mask].min()) if n else None,
            "normalizer_max": float(denominator[mask].max()) if n else None,
            "maximum_absolute_weight": float(np.abs(weights[mask]).max()) if n else None,
        })
    return np.stack(rows), support


def fit_payoff(frame):
    x, y = conditional_design(frame), frame.y_true.to_numpy(float)
    if not len(x) or not np.isfinite(x).all() or not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("payoff calibration requires finite design and both binary classes")
    matrix, support = moment_matrix(frame)

    def objective(beta):
        z = x @ beta
        p = expit(z)
        residual = p - y
        moments = matrix @ residual
        loss = np.mean(np.logaddexp(0, z) - y * z) + .5 * base.L2 * (beta @ beta) + .5 * (moments @ moments)
        gradient = x.T @ (residual / len(y) + p * (1 - p) * (matrix.T @ moments)) + base.L2 * beta
        return float(loss), gradient

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="BFGS",
                      options={"gtol": 1e-8, "maxiter": 2000})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun) or not np.isfinite(result.jac).all():
        raise RuntimeError(f"conditional payoff optimization failed: {result.message}")
    p = expit(x @ result.x)
    moments = matrix @ (p - y)
    return result.x, {
        "success": bool(result.success), "status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "objective": float(result.fun), "gradient_inf_norm": float(np.max(np.abs(result.jac))),
        "mean_bce": float(np.mean(np.logaddexp(0, x @ result.x) - y * (x @ result.x))),
        "l2_penalty": float(.5 * base.L2 * (result.x @ result.x)),
        "moment_penalty": float(.5 * (moments @ moments)),
        "training_moments": [{**item, "moment": float(value)} for item, value in zip(support, moments)],
    }


def predict(frame, model, beta):
    if model == "raw_sports":
        return frame.sports.to_numpy(float)
    x = conditional_design(frame) if model.startswith("conditional") else base.design(
        frame, ("sports", "market") if model == "benter" else ("market",))
    return expit(x @ beta)


def symmetry_error(frame, model, beta):
    swapped = frame.copy()
    swapped["sports"], swapped["market"] = 1 - frame.sports, 1 - frame.market
    swapped["odds_a"], swapped["odds_b"] = frame.odds_b, frame.odds_a
    error = float(np.max(np.abs(predict(frame, model, beta) + predict(swapped, model, beta) - 1)))
    if error > 1e-9:
        raise RuntimeError(f"{model}: prediction-based swapped-input symmetry failed: {error}")
    return error


def masked_uncertainty(frame, mask, first, second, first_name, second_name):
    """Shared ratio bootstrap over the full opportunity frame, including empty months."""
    result = base.selected_uncertainty(frame, mask, first, second)
    for source, target in (("probability_optimism", first_name), ("payoff_optimism", second_name)):
        if source in result:
            result[target] = result.pop(source)
    return result


def residual_report(frame, mask, p, sign, price, denominator, weights):
    residual = p - frame.y_true.to_numpy(float)
    side_residual, normalized_payoff = sign * residual, weights * residual
    n = int(mask.sum())
    return {
        "n": n, "opportunity_n": len(frame), "denominator_n": n,
        "status": "ok" if n >= MIN_OPTIONAL_TRAIN_N else "small_support" if n else "empty",
        "selected_a_n": int((mask & (sign > 0)).sum()), "selected_b_n": int((mask & (sign < 0)).sum()),
        "match_probability_residual": float(residual[mask].mean()) if n else None,
        "side_probability_residual": float(side_residual[mask].mean()) if n else None,
        "bounded_signed_price_residual": float(normalized_payoff[mask].mean()) if n else None,
        "unnormalized_unit_payoff_optimism_diagnostic": float(((1 - base.TAX) * price * side_residual)[mask].mean()) if n else None,
        "normalizer_min": float(denominator[mask].min()) if n else None,
        "normalizer_max": float(denominator[mask].max()) if n else None,
        "maximum_absolute_test_weight": float(np.abs(weights[mask]).max()) if n else None,
        "side_and_payoff_uncertainty": masked_uncertainty(
            frame, mask, side_residual, normalized_payoff, "side_probability_residual", "bounded_signed_price_residual"),
        "match_uncertainty": masked_uncertainty(
            frame, mask, residual, side_residual, "match_probability_residual", "side_probability_residual"),
    }


def audit_evaluation(frame, enabled_groups, header):
    masks, rows = audit_masks(frame), []
    raw_selected, raw_sign, raw_price, raw_denominator, raw_weights = side_context(frame, frame.sports)
    y = frame.y_true.to_numpy(float)
    for model in MODELS:
        p = frame[model].to_numpy(float)
        own_selected, own_sign, own_price, own_denominator, own_weights = side_context(frame, p)
        for name in enabled_groups:
            mask = masks[name]
            n = int(mask.sum())
            paired = {}
            scored = np.clip(p, base.EPS, 1 - base.EPS)
            ll = base.binary_log_loss_vector(y, scored)
            for reference in ("benter", "raw_sports", "market_calibrated"):
                ref = np.clip(frame[reference].to_numpy(float), base.EPS, 1 - base.EPS)
                ll_delta = ll - base.binary_log_loss_vector(y, ref)
                brier_delta = (p - y) ** 2 - (frame[reference].to_numpy(float) - y) ** 2
                paired[reference] = {
                    "direction": "candidate minus reference; negative favors candidate",
                    "log_loss_delta": float(ll_delta[mask].mean()) if n else None,
                    "brier_delta": float(brier_delta[mask].mean()) if n else None,
                    "uncertainty": masked_uncertainty(frame, mask, ll_delta, brier_delta, "log_loss_delta", "brier_delta"),
                }
            rows.append({
                **header, "model": model, "group": name, "opportunity_n": len(frame), "group_n": n,
                "group_months": int(pd.to_datetime(frame.loc[mask, "date"]).dt.to_period("M").nunique()),
                "group_log_loss": float(ll[mask].mean()) if n else None,
                "group_brier": float(((p - y) ** 2)[mask].mean()) if n else None,
                "paired": paired,
                "all_group_matches_raw_side": residual_report(frame, mask, p, raw_sign, raw_price, raw_denominator, raw_weights),
                "own_policy": residual_report(frame, mask & own_selected, p, own_sign, own_price, own_denominator, own_weights),
                "fixed_raw_sports_policy": residual_report(frame, mask & raw_selected, p, raw_sign, raw_price, raw_denominator, raw_weights),
            })
    return rows


def load_context(args):
    columns = pd.read_csv(args.snapshots, nrows=0).columns
    optional = [name for name in ("competition_tier", "roster_min_prior_series") if name in columns]
    context = pd.read_csv(args.snapshots, dtype={"golgg_match_id": "string"}, usecols=["golgg_match_id", *optional])
    if not context.golgg_match_id.is_unique:
        raise ValueError("duplicate snapshot context IDs")
    if "roster_min_prior_series" in context:
        context["roster_min_prior_series"] = pd.to_numeric(context.roster_min_prior_series, errors="raise")
    if "competition_tier" in context:
        known = {tier.value for tier in CompetitionTier}
        if not set(context.competition_tier.dropna()).issubset(known):
            raise ValueError("snapshot has competition tier outside existing enum definitions")
    return context


def run_fold(frame, phase, year):
    train = frame.loc[frame.date.ge("2024-01-01") & frame.date.lt(f"{year}-01-01")].copy().reset_index(drop=True)
    test = frame.loc[frame.date.ge(f"{year}-01-01") & frame.date.lt(f"{year + 1}-01-01")].copy().reset_index(drop=True)
    if train.empty or test.empty or train.date.max() >= test.date.min():
        raise ValueError(f"{phase}/{year}: empty or noncausal fold")
    actual_years = set(train.date.str[:4].astype(int))
    if actual_years != set(range(2024, year)) or set(train.golgg_match_id) & set(test.golgg_match_id):
        raise ValueError(f"{phase}/{year}: earlier-label-only contract failed")
    train_masks, test_masks = audit_masks(train), audit_masks(test)
    enabled, support = [], []
    for name in GROUP_DEFINITIONS:
        optional = name.startswith(("tier_", "roster_"))
        n = int(train_masks[name].sum()) if name in train_masks else 0
        use = name in train_masks and (not optional or n >= MIN_OPTIONAL_TRAIN_N)
        if use:
            enabled.append(name)
        support.append({"group": name, "train_n": n,
                        "test_n": int(test_masks[name].sum()) if name in test_masks else 0,
                        "enabled": use, "rule": "training input support only; no outcomes or heldout counts used"})
    fits = []
    for model in MODELS:
        if model == "raw_sports":
            beta, optimization, names = None, {"status": "unchanged supplied sports OOF"}, ()
        elif model == "conditional_payoff":
            beta, optimization = fit_payoff(train)
            names = FEATURE_NAMES
        else:
            names = FEATURE_NAMES if model == "conditional" else ("sports", "market") if model == "benter" else ("market",)
            x = conditional_design(train) if model == "conditional" else base.design(train, names)
            beta, optimization = base.fit_calibrator(x, train.y_true)
        train[model], test[model] = predict(train, model, beta), predict(test, model, beta)
        fits.append({
            "phase": phase, "test_year": year, "model": model,
            "coefficients": {} if beta is None else dict(zip(names, map(float, beta))),
            "optimization": optimization, "train_n": len(train), "test_n": len(test),
            "train_years": sorted(actual_years), "train_date_min": str(train.date.min()),
            "train_date_max": str(train.date.max()), "test_date_min": str(test.date.min()), "test_date_max": str(test.date.max()),
            "earlier_label_only_verified": True, "train_test_ids_disjoint": True,
            "train_swapped_input_symmetry_max_error": symmetry_error(train, model, beta),
            "heldout_swapped_input_symmetry_max_error": symmetry_error(test, model, beta),
        })
    for dataset in (train, test):
        dataset["phase"], dataset["year"] = phase, dataset.date.str[:4].astype(int)
        dataset["evaluation_year"] = year
        dataset["eligibility_live"] = 0
    train["split"], test["split"] = "train", "heldout"
    audits = []
    for split, dataset in (("train", train), ("heldout", test)):
        audits.extend(audit_evaluation(dataset, enabled, {"phase": phase, "test_year": year, "split": split}))
    return train, test, fits, audits, {"phase": phase, "test_year": year, "groups": support}


def verify_first_benter(predictions, path):
    prior = pd.read_csv(path, dtype={name: "string" for name in ("golgg_match_id", "team1_id", "team2_id")})
    keys = ["golgg_match_id", "phase", "year"]
    if prior.duplicated(keys).any() or predictions.duplicated(keys).any():
        raise ValueError("duplicate annual phase forecast keys")
    joined = predictions.merge(prior, on=keys, suffixes=("", "__prior"), how="outer", indicator=True, validate="one_to_one")
    if not joined._merge.eq("both").all():
        raise ValueError("conditional cohort differs from immutable first Benter cohort")
    for name in ("date", "y_true", "team1_id", "team2_id", "best_of"):
        if not joined[name].eq(joined[name + "__prior"]).all():
            raise ValueError(f"first Benter metadata mismatch: {name}")
    errors = {}
    for name in ("odds_a", "odds_b", "raw_sports", "benter", "market_calibrated"):
        error = float(np.max(np.abs(joined[name] - joined[name + "__prior"])))
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"first Benter nested control mismatch: {name} ({error})")
        errors[name] = error
    return {"cohort_identical": True, "maximum_absolute_differences": errors}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/artifacts/corrected-historical-reruns-20260908/comparison/predictions.csv")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/artifacts/corrected039081-20260908/replay/snapshots.csv")
    parser.add_argument("--odds", type=Path, default=ROOT / "data/artifacts/exp081-full-audit-20260908/identity-only-v3/odds.csv")
    parser.add_argument("--benter-run", type=Path, default=ROOT / "data/artifacts/exp081-market-calibration-20260908/run01")
    parser.add_argument("--output-dir", type=Path, required=True, help="Required new exclusive directory; existing paths refused")
    return parser.parse_args()


def main():
    args = parse_args()
    for key in ("predictions", "snapshots", "odds", "benter_run", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    if (base.TAX, base.MIN_EV, base.L2, base.SEED, base.RESAMPLES) != (.12, .05, .001, 82, 5000):
        raise RuntimeError("shared helper constants differ from the fixed protocol")
    paths = [args.predictions, args.snapshots, args.odds, *[
        args.benter_run / name for name in ("oos_predictions.csv", "fits.json", "summary.json", "protocol.json")]]
    inputs, code, frozen = {str(path): base.sha256(path) for path in paths}, base.code_hashes(), base.frozen_hashes()
    protocol = {
        "experiment": "exp081-conditional-payoff-calibration", "status": "exploratory_pre_run_protocol",
        "created_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "question": "Do two fixed symmetric conditional terms and a small bounded payoff-moment penalty improve historical annual OOS calibration?",
        "input_hashes": inputs, "code_hashes": code, "frozen_hashes": frozen,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "python": sys.version, "platform": platform.platform(), "models": list(MODELS),
        "prediction_column": base.PREDICTION_COLUMN, "phase_primary": "open", "phase_sensitivity": "close",
        "pooling": "never pool phases; annual fits separate by phase; pooled scores combine heldout years only within phase",
        "splits": [{"train_years": [2024], "test_year": 2025}, {"train_years": [2024, 2025], "test_year": 2026}],
        "earlier_label_only": "Only train.y_true enters optimizer; every training date strictly earlier than Jan 1 test year; IDs disjoint. Supplied sports probabilities treated as upstream OOF, not independent PIT proof.",
        "design": ["logit(p)", "logit(q)", "(logit(p)-logit(q))*abs(p-q)", "(logit(p)-logit(q))*(1/odds_a+1/odds_b-1)"],
        "market_probability": "proportional de-vig q=odds_b/(odds_a+odds_b), same STS phase pair, before tax",
        "intercept": 0, "logit_clip": base.EPS, "coefficient_constraints": "unconstrained; all coefficients L2-penalized",
        "conditional_objective": "mean(logaddexp(0,X beta)-y*(X beta)) + .001/2*||beta||^2",
        "nested_benter": "same objective using first two design columns; cohort and predictions checked against first Benter run",
        "moment_groups": list(MOMENT_GROUPS),
        "moment_formula": "s=+1 for fixed raw-sports selected A else -1; D_i=max(1,.88*odds_a,.88*odds_b); w_i=s_i*.88*selected_odds_i/D_i; M_g=sum_i I_g(i)*w_i*(sigmoid(X_i beta)-y_i)/n_g; n_g=sum I_g",
        "boundedness": "|w_i|<=1, |w_i*(r_i-y_i)|<=1 and |M_g|<=1; group means divide by group selection count, not all opportunities. No train-derived maximum or outcome normalization.",
        "payoff_objective": "conditional objective + (1/2)*sum over the three fixed groups M_g(beta)^2; coefficient fixed at 1/2; empty group term zero",
        "payoff_gradient": "X.T@((r-y)/n + r*(1-r)*(A.T@(A@(r-y)))) + .001*beta; A_gi=I_g(i)*w_i/max(n_g,1)",
        "optimization": {"method": "BFGS", "initialization": "zeros", "analytic_gradient": True, "gtol": 1e-8, "maxiter": 2000, "failure": "raise without fallback or hyperparameter search", "global_optimum_guaranteed_for_payoff": False},
        "audit_groups": GROUP_DEFINITIONS,
        "optional_group_rule": {"groups": "tier_* and roster_*", "minimum_training_input_support": MIN_OPTIONAL_TRAIN_N, "missing_definition": "omit, never infer", "heldout_support": "report even if small; never used for inclusion", "definitions_source": "existing snapshots and corrected_history_evaluation.py; CompetitionTier enum"},
        "audits": "For each fit: train and heldout; all group matches with fixed raw side, own-policy selected matches/sides, fixed raw-policy selected matches/sides; match, side and normalized signed price residuals; raw unit-payoff optimism separately unbounded diagnostic.",
        "chosen_side_ties": "Shared policy abstains exact EV ties; raw-side price slices retain shared helper's B-side tie convention on unselected opportunities, no selected moment receives a tied event.",
        "paired_scores": "LL and Brier candidate-minus-Benter/raw-sports/calibrated-market on identical group opportunity rows; negative favors candidate",
        "bootstrap": {"unit": "observed calendar month in full phase/fold opportunity cohort", "resamples": 5000, "seed": 82, "interval": "percentile 95%", "include_zero_selection_months": True, "zero_selection_replicates": "count and omit undefined ratios, never impute", "refit": False, "diagnostic_only": True},
        "policy": {"tax": .12, "max_ev_strictly_greater_than": .05, "ties": "abstain", "pairs": "finite same-book same-phase odds both >1"},
        "symmetry": "Recompute predictions from swapped sports/market probabilities and swapped odds; verify r(inputs)+r(swapped_inputs)=1 on training and heldout inputs, not just negated output logits.",
        "hyperparameter_selection": False, "sports_training": False, "eligibility_live": 0,
        "decision_time_certified": False, "promotion_approved": False, "limitations": LIMITATIONS,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base.write_json(args.output_dir / "protocol.json", protocol)
    try:
        frame, odds, alignment = base.load_aligned(args)
        context = load_context(args)
        forecasts, training, fits, audits, supports = [], [], [], [], []
        alignment["phases"] = {}
        for phase in PHASES:
            eligible, alignment["phases"][phase] = base.phase_frame(frame, odds, phase)
            eligible = eligible.merge(context, on="golgg_match_id", how="left", validate="one_to_one")
            pair = eligible[["odds_a", "odds_b"]].to_numpy(float)
            if not np.isfinite(pair).all() or not (pair > 1).all():
                raise ValueError("phase helper returned invalid paired quotes")
            for year in base.YEARS:
                train, test, fold_fits, fold_audits, support = run_fold(eligible, phase, year)
                training.append(train)
                forecasts.append(test)
                fits.extend(fold_fits)
                audits.extend(fold_audits)
                supports.append(support)
        predictions = pd.concat(forecasts, ignore_index=True)
        alignment["first_benter_control"] = verify_first_benter(predictions, args.benter_run / "oos_predictions.csv")
        summaries = []
        for phase in PHASES:
            phase_rows = predictions.loc[predictions.phase.eq(phase)].reset_index(drop=True)
            for period in (2025, 2026, "pooled_2025_2026"):
                rows = phase_rows if isinstance(period, str) else phase_rows.loc[phase_rows.year.eq(period)].reset_index(drop=True)
                for model in MODELS:
                    summaries.append({"phase": phase, "period": period, "model": model, **base.summarize(rows, model)})
        for path, digest in {**inputs, **code}.items():
            if base.sha256(Path(path)) != digest:
                raise RuntimeError(f"immutable input/code changed during run: {path}")
        if base.frozen_hashes() != frozen:
            raise RuntimeError("frozen models changed during run")
        predictions.to_csv(args.output_dir / "oos_predictions.csv", index=False)
        pd.concat(training, ignore_index=True).to_csv(args.output_dir / "training_predictions.csv", index=False)
        base.write_json(args.output_dir / "fits.json", fits)
        base.write_json(args.output_dir / "audit.json", {"groups": supports, "residual_and_paired_audits": audits})
        base.write_json(args.output_dir / "alignment.json", alignment)
        base.write_json(args.output_dir / "summary.json", {
            "status": "complete_exploratory", "eligibility_live": 0, "eligible_live_rows": 0,
            "promotion_approved": False, "immutable_hashes_verified": True,
            "oos_rows_by_phase": predictions.groupby("phase").size().to_dict(), "summaries": summaries,
            "limitations": LIMITATIONS,
        })
        base.write_json(args.output_dir / "manifest.json", {
            "input_hashes": inputs, "code_hashes": code, "frozen_hashes": frozen,
            "output_hashes": {path.name: base.sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()},
        })
        print(f"complete_exploratory: {args.output_dir}; eligibility_live=0; promotion_approved=false")
    except Exception as exc:
        base.write_json(args.output_dir / "failure.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "eligibility_live": 0, "promotion_approved": False,
        })
        raise


if __name__ == "__main__":
    main()
