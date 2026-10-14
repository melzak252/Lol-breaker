#!/usr/bin/env python3
"""Offline chronological team-strength experiment; no live/PIT certification.

Writes a pre-computation protocol, fit ancestry, heldout match scores and runnable
locally declared bracket demonstrations into a NEW --output-dir. No scraping,
DB access, betting teachers, historical bracket invention or artifact replacement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import combinations
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting_app.services.tournament_service import BracketMatchNode, TournamentBracket
from scripts.benchmark_siamese_architectures import monthly_bootstrap, probability_metrics
from scripts.build_siamese_research_dataset import sha256
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.frozen_tournament import simulate_frozen_bracket
from src.models.tournament_uncertainty import fit_team_strength_posterior

OOF = ROOT / "data/artifacts/corrected-historical-reruns-20260908/comparison/predictions.csv"
SNAPSHOTS = ROOT / "data/artifacts/corrected039081-20260908/replay/snapshots.csv"
MODELS = ("exp039", "recipe__reported_regularization_bagging")
PRIOR_GRID = (0.1, 0.3, 0.7, 1.5)
META = ["golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y_true"]


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _load(oof, snapshots):
    frame = pd.read_csv(oof, usecols=META + list(MODELS), dtype={"team1_id": str, "team2_id": str})
    source = pd.read_csv(snapshots, usecols=META + ["tournament"], dtype={"team1_id": str, "team2_id": str})
    if frame.golgg_match_id.duplicated().any() or source.golgg_match_id.duplicated().any():
        raise ValueError("Exact source match IDs must be unique")
    frame["date"], source["date"] = pd.to_datetime(frame.date), pd.to_datetime(source.date)
    joined = frame.merge(source, on=META, how="left", validate="one_to_one", indicator=True)
    if (joined._merge != "both").any() or joined.tournament.isna().any():
        raise ValueError("OOF metadata must exactly match local sports snapshot identities and outcomes")
    joined = joined.drop(columns="_merge").sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    if joined[META + ["tournament"]].isna().any().any() or not joined.best_of.isin([1, 3, 5]).all():
        raise ValueError("Invalid series metadata")
    probabilities = joined[list(MODELS)].to_numpy()
    if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("Every predeclared model must cover the complete paired cohort")
    if not joined.y_true.isin([0, 1]).all() or (joined.team1_id == joined.team2_id).any():
        raise ValueError("Invalid outcome or self-paired evidence")
    return joined.loc[joined.date >= "2024-01-01"].reset_index(drop=True)


def _save_posterior(directory, label, posterior):
    _json(directory / f"{label}.json", posterior.provenance)
    np.savez_compressed(directory / f"{label}.npz", teams=np.asarray(posterior.teams),
                        mean=posterior.mean, covariance=posterior.covariance)


def _choose_prior(frame, model, directory):
    calibration = frame.loc[(frame.date >= "2024-07-01") & (frame.date < "2025-01-01")]
    if calibration.empty:
        raise ValueError("Predeclared 2024 H2 calibration cohort is unavailable")
    rows = []
    for prior in PRIOR_GRID:
        posterior = fit_team_strength_posterior(frame, model, cutoff="2024-07-01", prior_sd=prior)
        _save_posterior(directory, f"{model}__calibration_sd{prior}", posterior)
        prediction = posterior.predict(calibration.team1_id, calibration.team2_id, calibration[model])
        rows.append({"prior_sd": prior, "log_loss": float(binary_log_loss_vector(calibration.y_true, prediction).mean()),
                     "n": len(calibration)})
    # Stable ascending grid breaks exact ties toward stronger shrinkage.
    chosen = min(rows, key=lambda row: row["log_loss"])["prior_sd"]
    return chosen, {"candidate_scores": rows, "selected_prior_sd": chosen,
                    "calibration_match_ids": calibration.golgg_match_id.tolist(),
                    "training_cutoff_exclusive": "2024-07-01", "calibration_end_exclusive": "2025-01-01"}


def _paired(frame, candidate, reference):
    ll = binary_log_loss_vector(frame.y_true, frame[candidate]) - binary_log_loss_vector(frame.y_true, frame[reference])
    bs = (frame[candidate] - frame.y_true)**2 - (frame[reference] - frame.y_true)**2
    if frame.date.dt.to_period("M").nunique() < 2:
        return {"log_loss_mean": float(np.mean(ll)), "brier_mean": float(np.mean(bs)),
                "interval_unavailable": "fewer than two calendar months"}
    return {"log_loss": monthly_bootstrap(ll, frame.date, repetitions=5000),
            "brier": monthly_bootstrap(bs, frame.date, repetitions=5000)}


def _heldout(frame, priors, directory, *, cadence="annual"):
    if cadence not in ("annual", "monthly"):
        raise ValueError("Posterior cadence must be annual or monthly")
    pieces, fits = [], {}
    selected = frame.loc[frame.date.dt.year.isin((2025, 2026))]
    periods = selected.date.dt.to_period("Y" if cadence == "annual" else "M")
    for period, group in selected.groupby(periods, sort=True):
        cohort = group.copy()
        cutoff = period.start_time.strftime("%Y-%m-%d")
        for model in MODELS:
            posterior = fit_team_strength_posterior(frame, model, cutoff=cutoff, prior_sd=priors[model])
            label = f"{model}__{period}"
            _save_posterior(directory, label, posterior)
            # Only January fits can be reused by the fixed start-of-year graph.
            # Other monthly covariance matrices are persisted, not retained.
            if cadence == "annual" or period.month == 1:
                fits[(model, period.year)] = posterior
            cohort[model + "__mean_only"] = posterior.predict(cohort.team1_id, cohort.team2_id, cohort[model], integrate=False)
            cohort[model + "__integrated"] = posterior.predict(cohort.team1_id, cohort.team2_id, cohort[model])
            cohort[model + "__posterior_cutoff"] = cutoff
        # Every model fits the same earlier rows and thus the same exact IDs.
        cohort["posterior_known_teams"] = (
            cohort.team1_id.isin(posterior.teams).to_numpy(dtype=np.int8)
            + cohort.team2_id.isin(posterior.teams).to_numpy(dtype=np.int8)
        )
        cohort["eligibility_live"] = 0
        pieces.append(cohort)
    if not pieces:
        raise ValueError("No predeclared 2025/2026 heldout matches")
    return pd.concat(pieces, ignore_index=True), fits


def _local_pair_clique(frame):
    """Choose a four-team complete observed pair graph without inspecting outcomes.

    This is explicitly NOT a recovered historical bracket. Probabilities come
    from stale earlier observations, not hypothetical-pair current inference.
    """
    earlier = frame.loc[frame.date < "2025-01-01"].sort_values(["date", "golgg_match_id"])
    for best_of in (1, 3, 5):
        rows = earlier.loc[earlier.best_of == best_of]
        pairs, neighbors = {}, {}
        for row in rows.itertuples(index=False):
            a, b = sorted((row.team1_id, row.team2_id))
            pairs[(a, b)] = row
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
        # Lexicographic first clique, graph structure only: no outcome tuning.
        for a in sorted(neighbors):
            for b in sorted(t for t in neighbors[a] if t > a):
                common = neighbors[a] & neighbors[b]
                for c in sorted(t for t in common if t > b):
                    remaining = sorted(t for t in common & neighbors[c] if t > c)
                    if remaining:
                        teams = [a, b, c, remaining[0]]
                        records = [pairs[pair]._asdict() for pair in combinations(teams, 2)]
                        return teams, best_of, pd.DataFrame(records)
    raise ValueError("No four-team complete same-format local pair graph exists")


def _bracket(teams, best_of, double):
    def node(identifier, section="upper", **kwargs):
        return BracketMatchNode(id=identifier, name=identifier, round_name=identifier,
                                bracket_section=section, best_of=best_of, **kwargs)
    matches = {
        "semi1": node("semi1", team1=teams[0], team2=teams[1], next_match_winner_id="title", next_match_winner_slot=1),
        "semi2": node("semi2", team1=teams[2], team2=teams[3], next_match_winner_id="title", next_match_winner_slot=2),
        "title": node("title"),
    }
    if double:
        for number in (1, 2):
            matches[f"semi{number}"].next_match_winner_id = "upper"
            matches[f"semi{number}"].next_match_loser_id = "lower1"
            matches[f"semi{number}"].next_match_loser_slot = number
        matches["upper"] = node("upper", next_match_winner_id="title", next_match_winner_slot=2,
                                 next_match_loser_id="lower2", next_match_loser_slot=1)
        matches["lower1"] = node("lower1", "lower", next_match_winner_id="lower2", next_match_winner_slot=2)
        matches["lower2"] = node("lower2", "lower", next_match_winner_id="title", next_match_winner_slot=1)
    format_name = "double_elimination" if double else "single_elimination"
    return TournamentBracket(id=f"local_declared_{format_name}", name="Local declared demonstration",
                             region="research", format=format_name, teams=teams, matches=matches)


def _scenarios(frame, priors, fits, directory, simulations, seed):
    teams, best_of, pairs = _local_pair_clique(frame)
    pairs.to_csv(directory / "pair_probability_ancestry.csv", index=False)
    strength_seed, outcome_seed = (
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(seed).spawn(2)
    )
    summary = {"qualification": "locally_declared_graph_demonstration_not_historical_event_replay",
               "freeze_cutoff_exclusive": "2025-01-01", "eligibility_live": 0,
               "random_streams": {"root_seed": seed, "strength_seed": strength_seed,
                                  "outcome_seed": outcome_seed, "derivation": "SeedSequence.spawn(2)"},
               "interpretation": "Stale earlier SERIES forecasts; not contemporaneous hypothetical pair predictions",
               "selection": "lexicographic first four-team complete earlier pair graph; Bo1 then Bo3 then Bo5",
               "outcome_scoring": "none: no true frozen tournament outcomes", "scenarios": {}}
    for double in (False, True):
        bracket = _bracket(teams, best_of, double)
        _json(directory / f"{bracket.id}__bracket.json", asdict(bracket))
        outputs = {}
        for model in MODELS:
            posterior = fits.get((model, 2025))
            if posterior is None:
                posterior = fit_team_strength_posterior(frame, model, cutoff="2025-01-01", prior_sd=priors[model])
                _save_posterior(directory, f"{model}__scenario", posterior)
            keys = list(zip(pairs.team1_id, pairs.team2_id, pairs.best_of.astype(int)))
            fixed = dict(zip(keys, pairs[model].astype(float)))
            mean = dict(zip(keys, posterior.predict(pairs.team1_id, pairs.team2_id, pairs[model], integrate=False)))
            marginal = dict(zip(keys, posterior.predict(pairs.team1_id, pairs.team2_id, pairs[model])))
            draws = posterior.sample(teams, simulations, strength_seed)
            results = {
                "fixed": simulate_frozen_bracket(bracket, fixed, simulations, outcome_seed),
                "zero_draw_control": simulate_frozen_bracket(bracket, fixed, simulations, outcome_seed,
                                                             team_strength_draws={team: np.zeros(simulations) for team in teams}),
                "mean_only": simulate_frozen_bracket(bracket, mean, simulations, outcome_seed),
                "coherent_posterior": simulate_frozen_bracket(bracket, fixed, simulations, outcome_seed, team_strength_draws=draws),
                "independent_matched_marginals": simulate_frozen_bracket(bracket, marginal, simulations, outcome_seed),
            }
            if results["fixed"] != results["zero_draw_control"]:
                raise RuntimeError("Zero uncertainty control changed the fixed simulator")
            results["champion_deltas"] = {
                reference: {team: results["coherent_posterior"]["champion_prob"][team] - results[reference]["champion_prob"][team]
                            for team in teams}
                for reference in ("fixed", "mean_only", "independent_matched_marginals")}
            results["paired_series_probabilities"] = [
                {"team1_id": key[0], "team2_id": key[1], "best_of": int(key[2]),
                 "fixed": fixed[key], "mean_only": mean[key], "integrated": marginal[key]}
                for key in keys]
            outputs[model] = results
        summary["scenarios"][bracket.id] = outputs
    summary["dependence_control"] = (
        "independent_matched_marginals uses the same simulator without strength draws and quadrature-integrated "
        "pair probabilities: same unconditional SERIES marginals, independent round Bernoulli outcomes. "
        "Coherent advancement selects latent strengths; conditional later-round probabilities need not match. "
        "Monte Carlo tournament deltas are demonstrations, not event-bootstrap accuracy evidence."
    )
    _json(directory / "scenario_outputs.json", summary)
    return summary


def run(output_dir, *, oof=OOF, snapshots=SNAPSHOTS, simulations=20000, seed=82, cadence="annual"):
    output_dir, oof, snapshots = Path(output_dir), Path(oof), Path(snapshots)
    if simulations <= 0 or seed < 0:
        raise ValueError("Positive simulations and nonnegative seed required")
    if cadence not in ("annual", "monthly"):
        raise ValueError("Posterior cadence must be annual or monthly")
    output_dir.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__), ROOT / "src/models/tournament_uncertainty.py", ROOT / "src/models/frozen_tournament.py",
            ROOT / "betting_app/services/tournament_service.py"]
    ancestry_paths = [
        ROOT / "data/artifacts/corrected039081-20260908/evaluation-complete/summary.json",
        ROOT / "data/artifacts/corrected-historical-reruns-20260908/reported-recipe081/manifest.json",
        ROOT / "data/artifacts/corrected-historical-reruns-20260908/run_recipe.py",
    ] + [
        ROOT / f"data/artifacts/corrected-historical-reruns-20260908/reported-recipe081/{year}-calibration.json"
        for year in (2024, 2025, 2026)
    ]
    protocol = {
        "created_at": datetime.now(UTC).isoformat(), "qualification": "retrospective_research_protocol_before_this_computation",
        "question": "Does team-only posterior integration improve heldout marginal forecasts beyond mean correction, and change frozen graph outputs?",
        "models": list(MODELS), "data_sha256": {str(path): sha256(path) for path in (oof, snapshots)},
        "code_sha256": {str(path): sha256(path) for path in code},
        "canonical_source_ancestry_sha256": {str(path): sha256(path) for path in ancestry_paths},
        "canonical_input_paths": oof.resolve() == OOF.resolve() and snapshots.resolve() == SNAPSHOTS.resolve(),
        "source_column_ancestry": {
            "exp039": "evaluation-complete exp039-corrected-history-research-v1; symmetric ElasticNet followed by positive zero-intercept Platt fitted in prior full year",
            "recipe__reported_regularization_bagging": "reported-recipe081 p__reported_regularization_bagging; sigmoid of mean member-scaled antisymmetric logits; five member Platt slopes on prior full year; NO additional ensemble slope",
            "distinction": "recipe__reported_regularization_bagging_ensemble_cal is a separate source column and is not used",
            "source_training_split": "train from 2020 strictly before test year minus one; calibration prior full calendar year; test target year",
            "qualification": "canonical source declarations and hashes, not source-timestamp certification; custom input paths require their own upstream ancestry audit",
        },
        "cohort": "all exact shared OOF modern matches from 2024; no bookmaker filtering",
        "primary_metric": "paired heldout LogLoss integrated minus posterior-mean-only; Brier diagnostic",
        "prior_grid": list(PRIOR_GRID), "calibration_train_end_exclusive": "2024-07-01",
        "calibration_period": "2024-07-01 <= date < 2025-01-01",
        "selection": "minimum calibration marginal LogLoss, ties choose smallest prior_sd",
        "heldout": f"2025 and 2026 separately; {cadence} posterior refits use dates strictly before calendar period start; hyperparameters locked in 2024",
        "posterior_cadence": cadence,
        "cadence_comparison": "Annual and monthly use the same H1-fit/H2-select prior; no heldout tuning or team-name identity merging. Neither cadence is a tournament-start replay.",
        "controls": ["fixed", "zero residual draws", "posterior mean only", "joint posterior", "independent matched marginal series"],
        "posterior": "logistic sports OOF offset plus paired team effects; Gaussian shrinkage; full joint Laplace covariance",
        "posterior_assumptions": "static team identity; no explicit recency weighting; no shared-player, roster-change or patch drift covariance",
        "bootstrap": {"unit": "calendar month", "repetitions": 5000, "seed": 82, "paired": True},
        "scenario_selection": "pre-2025 first lexicographic 4-team clique, Bo1/Bo3/Bo5 priority; last prior observed pair forecasts",
        "scenario_status": "locally declared unplayed graphs, not reconstructed historical brackets; no tournament outcome scoring",
        "eligibility_live": 0, "source_timing": "not certified; earlier outcomes alone do not prove PIT availability",
        "seed": seed, "simulations": simulations, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
        "betting_inputs": False, "model_status": "exp039 frozen comparison; corrected recipe is not current operational model",
    }
    _json(output_dir / "protocol.json", protocol)
    frame = _load(oof, snapshots)
    fit_dir, scenario_dir = output_dir / "fits", output_dir / "scenarios"
    fit_dir.mkdir()
    scenario_dir.mkdir()
    priors, calibration = {}, {}
    for model in MODELS:
        priors[model], calibration[model] = _choose_prior(frame, model, fit_dir)
    _json(output_dir / "calibration_selection.json", calibration)
    heldout, fits = _heldout(frame, priors, fit_dir, cadence=cadence)
    heldout.to_csv(output_dir / "predictions.csv", index=False)
    coverage = [
        {"year": int(year), "n": len(group),
         **{name: int((group.posterior_known_teams == count).sum())
            for count, name in enumerate(("neither_known", "one_known", "both_known"))}}
        for year, group in heldout.groupby(heldout.date.dt.year)
    ]
    _json(output_dir / "posterior_coverage.json", coverage)
    scores = {}
    for label, group in [("all_heldout", heldout)] + [(str(year), part) for year, part in heldout.groupby(heldout.date.dt.year)]:
        scores[label] = {}
        for model in MODELS:
            mean, integrated = model + "__mean_only", model + "__integrated"
            scores[label][model] = {
                "metrics": {name: probability_metrics(group.y_true, group[name]) for name in (model, mean, integrated)},
                "integrated_minus_fixed": _paired(group, integrated, model),
                "mean_only_minus_fixed": _paired(group, mean, model),
                "integrated_minus_mean_only": _paired(group, integrated, mean),
            }
    _json(output_dir / "heldout_scores.json", scores)
    scenarios = _scenarios(frame, priors, fits, scenario_dir, simulations, seed)
    result = {"heldout_rows": len(heldout), "selected_prior_sd": priors,
              "posterior_cadence": cadence, "posterior_coverage": coverage,
              "scenario_count": len(scenarios["scenarios"]), "eligibility_live": 0,
              "interpretation": "Retrospective heldout marginal evidence and local tournament demonstrations; no PIT certificate or production promotion"}
    _json(output_dir / "summary.json", result)
    products = sorted(path for path in output_dir.rglob("*") if path.is_file())
    _json(output_dir / "manifest.json", {str(path.relative_to(output_dir)): sha256(path) for path in products})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oof", type=Path, default=OOF)
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOTS)
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--posterior-cadence", choices=("annual", "monthly"), default="annual")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, oof=args.oof, snapshots=args.snapshots,
                         simulations=args.simulations, seed=args.seed, cadence=args.posterior_cadence), indent=2))


if __name__ == "__main__":
    main()
