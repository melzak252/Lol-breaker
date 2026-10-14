"""Tournament Bracket Monte Carlo Simulation Engine.

Orchestrates full tournament bracket simulations (Swiss stage, GSL groups, double/single
elimination playoffs) using `src.models.tournament_formats.simulate_tournament`.
Produces structured, mass-conserving tournament probability distributions:
- Champion probability distribution: sum(P(Champ)) == 1.0.
- Finalist probability distribution: sum(P(Finalist)) == 2.0.
- Swiss stage qualification / record distributions: sum(P(Advance)) == 8.0.
- Side-symmetric conditional series scoreline distributions.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from typing import Any

import numpy as np

from src.models.tournament_formats import simulate_tournament, validate_tournament_spec
from src.analysis.tournament_numerics import summarize_tournament_numerics
from src.models.tournament_scorelines import build_score_distributions


SWISS_RECORD_BANDS: dict[str, tuple[int, int]] = {
    "3-0": (0, 2),    # ranks 1-2
    "3-1": (2, 5),    # ranks 3-5
    "3-2": (5, 8),    # ranks 6-8
    "2-3": (8, 11),   # ranks 9-11
    "1-3": (11, 14),  # ranks 12-14
    "0-3": (14, 16),  # ranks 15-16
}


def extract_swiss_record_probabilities(
    stage_rank_prob: Mapping[str, Mapping[str, Sequence[float]]],
    group: str = "all",
) -> dict[str, dict[str, float]]:
    """Extract Swiss record distributions (3-0, 3-1, 3-2, 2-3, 1-3, 0-3) from rank probabilities.

    Parameters
    ----------
    stage_rank_prob : Mapping
        Dictionary mapping group -> team -> vector of rank probabilities (0-indexed).
    group : str
        Group identifier, defaults to "all".

    Returns
    -------
    dict[str, dict[str, float]]
        team -> record -> probability.
    """
    records_by_team: dict[str, dict[str, float]] = {}
    ranks = stage_rank_prob.get(group, {})

    for team, rank_vector in ranks.items():
        if len(rank_vector) != 16:
            raise ValueError("Swiss record bands require a complete 16-rank distribution")
        _check_mass(rank_vector, 1.0, f"Swiss rank distribution for {team}")
        team_records: dict[str, float] = {}
        for record_label, (start_idx, end_idx) in SWISS_RECORD_BANDS.items():
            team_records[record_label] = float(math.fsum(rank_vector[start_idx:end_idx]))
        records_by_team[team] = team_records

    return records_by_team


def _check_mass(values: Sequence[float], expected: float, label: str) -> None:
    if any(isinstance(p, bool) or not isinstance(p, Real) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError(f"{label} contains an invalid probability")
    if not math.isclose(math.fsum(values), expected, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{label} violates probability mass conservation")


def _canonical_probabilities(spec, probabilities, required_formats):
    if not isinstance(probabilities, Mapping):
        raise ValueError("probabilities must be a mapping")
    order = {team: index for index, team in enumerate(spec["teams"])}
    clean = {}
    originals = {}
    seen = set()
    formats = set(required_formats)
    for key, p in probabilities.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("Probability keys must be (team_a, team_b, best_of)")
        a, b, bo = key
        if not isinstance(a, str) or not isinstance(b, str) or a not in order or b not in order or a == b:
            raise ValueError("Probability opponents must be distinct exact entrant IDs")
        if isinstance(bo, bool) or not isinstance(bo, int) or bo not in (1, 3, 5):
            raise ValueError("Probability best_of must be 1, 3 or 5")
        if key in seen:
            raise ValueError(f"Duplicate probability row {key!r}")
        seen.add(key)
        if isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid series probability for {key!r}")
        canonical = (a, b, bo) if order[a] < order[b] else (b, a, bo)
        value = float(p) if order[a] < order[b] else 1.0 - float(p)
        if canonical in clean:
            if not math.isclose(clean[canonical], value, rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Inconsistent reverse probability for {key!r}")
        else:
            clean[canonical] = value
        if canonical not in originals or float(p) < originals[canonical][1]:
            originals[canonical] = (key, float(p))
        formats.add(bo)
    missing = [(a, b, bo) for a, b in combinations(spec["teams"], 2)
               for bo in sorted(formats) if (a, b, bo) not in clean]
    if missing:
        raise ValueError(f"Missing complete pair probability coverage: {missing[0]!r}")
    return {key: p for key, p in originals.values()}


def _check_joint(rows, marginals, cardinality, teams, label):
    if not isinstance(rows, Mapping) or not rows:
        raise ValueError(f"{label} joint distribution is missing")
    _check_mass(list(rows.values()), 1.0, f"{label} joint distribution")
    implied = dict.fromkeys(teams, 0.0)
    for key, p in rows.items():
        try:
            members = json.loads(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} has an invalid member set") from exc
        if (not isinstance(members, list) or len(members) != cardinality or
                any(not isinstance(team, str) or team not in implied for team in members) or
                len(set(members)) != cardinality or members != sorted(members)):
            raise ValueError(f"{label} has an invalid member set")
        for team in members:
            implied[team] += p
    if any(not math.isclose(implied[team], marginals[team], rel_tol=0, abs_tol=1e-7)
           for team in teams):
        raise ValueError(f"{label} joint distribution contradicts its marginals")


def _validate_result_mass(spec, result):
    """Check original whole-phase marginals, never condition away non-reach."""
    teams = set(spec["teams"])

    def team_rows(rows, label):
        if not isinstance(rows, Mapping) or set(rows) != teams:
            raise ValueError(f"{label} must retain every original entrant")
        return rows

    champion, final = result["champion_prob"], result["final_prob"]
    if champion is not None:
        _check_mass(list(team_rows(champion, "Champion").values()), 1.0, "Champion")
    if final is not None:
        _check_mass(list(team_rows(final, "Final").values()), 2.0, "Final")
        _check_joint(result["joint_final_prob"], final, 2, teams, "Final")
        if champion is not None and any(champion[t] > final[t] + 1e-7 for t in teams):
            raise ValueError("Champion probability exceeds final participation")
    for stage in spec["stages"]:
        sid = stage["id"]
        groups = stage.get("groups", {"all": stage.get("entrants", [])})
        rank_groups = result["stage_rank_prob"][sid]
        if set(rank_groups) != set(groups):
            raise ValueError("Rank groups disagree with the declared topology")
        participation = dict.fromkeys(teams, 0.0)
        for group, refs in groups.items():
            ranks = team_rows(rank_groups[group], f"{sid}/{group} ranks")
            if any(len(vector) != len(refs) for vector in ranks.values()):
                raise ValueError("Rank vector disagrees with the declared topology")
            for index in range(len(refs)):
                _check_mass([ranks[t][index] for t in teams], 1.0, f"{sid}/{group} rank {index + 1}")
            for team, vector in ranks.items():
                participation[team] += math.fsum(vector)
        if any(value > 1.0 + 1e-7 for value in participation.values()):
            raise ValueError("Stage participation duplicates probability mass")
        if "advance" in stage:
            advance = team_rows(result["advance_prob"][sid], f"{sid} advancement")
            count = stage["advance"] * len(groups)
            _check_mass(list(advance.values()), count, f"{sid} advancement")
            for team in teams:
                expected = math.fsum(math.fsum(rank_groups[g][team][:stage["advance"]]) for g in groups)
                if not math.isclose(advance[team], expected, rel_tol=0, abs_tol=1e-7):
                    raise ValueError("Advancement contradicts stage rank probabilities")
            _check_joint(result["joint_advance_prob"][sid], advance, count, teams, sid)
    if spec.get("placement_bands"):
        categories = [band["label"] for band in spec["placement_bands"]]
        if result["placement_categories"] != categories:
            raise ValueError("Placement categories disagree with official bands")
        placements = team_rows(result["placement_prob"], "Placement")
        for team, rows in placements.items():
            if set(rows) != set(categories):
                raise ValueError("Placement distribution omits an official category")
            _check_mass(list(rows.values()), 1.0, f"Placement for {team}")
        for band in spec["placement_bands"]:
            _check_mass([placements[t][band["label"]] for t in teams],
                        len(band["placements"]), f"Placement band {band['label']}")


def _pair_scoreline_laws(probabilities, scorelines, conditional, strength_draws, copula_rho=None):
    """Marginalize the actual simulation law over frozen-origin latent draws.

    These hypothetical-pair diagnostics are unconditional on bracket reach and
    on previously sampled outcomes. In particular, averaging the winner p first
    and then evaluating its conditional score law would give the wrong mixture.
    """
    strengths = (None if strength_draws is None or copula_rho == 0 else
                 {team: np.asarray(values, dtype=np.longdouble)
                  for team, values in strength_draws.items()})
    if copula_rho is not None and copula_rho > 0:
        from scipy.special import ndtr, ndtri
        copula_scale = math.sqrt(copula_rho / 2)
        copula_denominator = math.sqrt(1 - copula_rho)
    conditional_pmfs = ({} if conditional is None else
                        {int(bo): np.asarray(pmfs, dtype=float)
                         for bo, pmfs in conditional["distributions"].items()})
    bins = None if conditional is None else np.asarray(conditional["probability_bins"], dtype=float)
    rows, unavailable = [], set()
    for (a, b, bo), base_p in probabilities.items():
        p = np.asarray([base_p])
        if strengths is not None and 0 < base_p < 1:
            delta = strengths[a] - strengths[b]
            if copula_rho is not None:
                z = (ndtri(base_p) + copula_scale * delta) / copula_denominator
                # The normal CDF already saturates long before float overflow;
                # bound only the intermediate conversion, never probabilities.
                np.clip(z, -np.finfo(float).max, np.finfo(float).max, out=z)
                p = ndtr(z.astype(float))
            else:
                z = math.log(base_p) - math.log1p(-base_p) + delta
                with np.errstate(over="ignore", under="ignore"):
                    exp_negative_abs = np.exp(-np.asarray(np.abs(z), dtype=float))
                shifted = np.where(z >= 0, 1 / (1 + exp_negative_abs),
                                   exp_negative_abs / (1 + exp_negative_abs))
                # The logit-offset core leaves exact p unchanged for zero offsets.
                p = np.where(delta == 0, base_p, shifted)
        marginal_p = float(p.mean())
        wins = bo // 2 + 1
        if bo == 1:
            a_scores, b_scores = np.asarray([marginal_p]), np.asarray([1 - marginal_p])
        elif conditional is not None and bo in conditional_pmfs:
            pmfs = conditional_pmfs[bo]
            if copula_rho is not None:
                a_bin = min(np.searchsorted(bins, base_p, side="right") - 1, len(bins) - 2)
                b_bin = min(np.searchsorted(bins, 1 - base_p, side="right") - 1, len(bins) - 2)
                a_scores = marginal_p * pmfs[a_bin]
                b_scores = (1 - marginal_p) * pmfs[b_bin]
            else:
                a_bins = np.minimum(np.searchsorted(bins, p, side="right") - 1, len(bins) - 2)
                b_bins = np.minimum(np.searchsorted(bins, 1 - p, side="right") - 1, len(bins) - 2)
                a_weights = np.bincount(a_bins, weights=p, minlength=len(bins) - 1) / p.size
                b_weights = np.bincount(b_bins, weights=1 - p, minlength=len(bins) - 1) / p.size
                a_scores, b_scores = a_weights @ pmfs, b_weights @ pmfs
        elif conditional is None and scorelines is not None and bo in scorelines:
            pmf = np.asarray(scorelines[bo], dtype=float)
            a_scores, b_scores = marginal_p * pmf, (1 - marginal_p) * pmf
        else:
            unavailable.add(bo)
            continue
        score_pmf = {f"{wins}-{losing}": float(value) for losing, value in enumerate(a_scores)}
        score_pmf.update({f"{losing}-{wins}": float(value) for losing, value in enumerate(b_scores)})
        length_pmf = {str(wins + losing): float(a_scores[losing] + b_scores[losing])
                      for losing in range(wins)}
        _check_mass(list(score_pmf.values()), 1.0, f"{a}/{b}/BO{bo} scoreline mixture")
        _check_mass(list(length_pmf.values()), 1.0, f"{a}/{b}/BO{bo} length mixture")
        rows.append({"team_a": a, "team_b": b, "best_of": bo, "p": marginal_p,
                     "scoreline_pmf": score_pmf, "series_length_pmf": length_pmf})
    return {
        "probability_unit": "series_win",
        "scope": "frozen_origin_unconditional_hypothetical_pair",
        "integration": "finite_supplied_latent_draws" if strengths is not None else "fixed_pair_probability",
        "draw_count": len(next(iter(strengths.values()))) if strengths is not None else 1,
        "score_law": ("engine_origin_probability_bins" if conditional is not None and copula_rho is not None
                      else "engine_probability_bins" if conditional is not None else "engine_pooled"),
        "continuous_score_model_used": False,
        "integration_exact_for_supplied_draws": True,
        "reach_conditioned": False,
        "rows": rows,
        "unavailable_reasons": {str(bo): "No supplied simulation score law for this best_of"
                                for bo in sorted(unavailable)},
    }


def simulate_tournament_bracket(
    spec: dict[str, Any],
    probabilities: Mapping[tuple[str, str, int], float],
    scorelines: Mapping[int, Sequence[float]] | None = None,
    conditional_score_distributions: Mapping[str, Any] | None = None,
    simulations: int = 10000,
    seed: int = 20261005,
    *,
    team_strength_draws: Mapping[str, Sequence[float]] | None = None,
    integration: str = "monte_carlo",
    copula_rho: float | None = None,
) -> dict[str, Any]:
    """Integrate the declared tournament law by Monte Carlo, exact or hybrid methods.

    Parameters
    ----------
    spec : dict[str, Any]
        Instantiated tournament format specification.
    probabilities : Mapping[tuple[str, str, int], float]
        Pairwise series-win probabilities ((team_a, team_b, best_of) -> float).
    scorelines : Mapping[int, Sequence[float]] | None
        Optional pooled losing-map score distributions {3: [...], 5: [...]}.
    conditional_score_distributions : Mapping[str, Any] | None
        Winner-probability-binned losing-map law. The returned metadata records
        this simulation approximation; exact continuous-law scoring is separate.
    team_strength_draws : Mapping[str, Sequence[float]] | None
        One series-logit shift per entrant per rollout, or standardized team
        states when copula_rho selects the conditional probit kernel.
    copula_rho : float | None
        Learned shared-player dependence in [0, .5], for pre-event MC only.
        Conditional score bins use original probabilities. Only a verified
        Gaussian-prior producer establishes marginal preservation and iid paths.
    simulations : int
        Number of Monte Carlo realizations (default 10,000).
    seed : int
        RNG seed for simulation repeatability.
    integration : str
        ``monte_carlo`` samples complete paths; ``exact`` integrates supported
        fixed routes; ``hybrid`` samples adaptive prefixes and integrates supported
        fixed suffixes. Unsupported explicit requests raise instead of falling back.

    Returns
    -------
    dict[str, Any]
        Structured simulation results containing:
        - champion_prob: dict[team, float] summing to 1.0
        - final_prob: dict[team, float] summing to 2.0 (if final stage exists)
        - advance_prob: dict[stage, dict[team, float]]
        - stage_rank_prob: dict[stage, dict[group, dict[team, list[float]]]]
        - swiss_record_prob: dict[team, dict[str, float]] (if Swiss stage exists)
        - series_score_distributions: dict[int, list[float]]
        - expected_series: float
        - hits: raw counts
    """
    t0 = time.perf_counter()

    formats = validate_tournament_spec(spec)
    clean_probs = _canonical_probabilities(spec, probabilities, formats)
    default_score_law = scorelines is None and conditional_score_distributions is None
    if default_score_law:
        scorelines = build_score_distributions(p_series=0.5, best_ofs=(3, 5))

    raw_result = simulate_tournament(
        spec=spec,
        probabilities=clean_probs,
        simulations=simulations,
        seed=seed,
        score_distributions=scorelines,
        conditional_score_distributions=conditional_score_distributions,
        team_strength_draws=team_strength_draws,
        integration=integration,
        copula_rho=copula_rho,
    )

    t_elapsed = time.perf_counter() - t0

    _validate_result_mass(spec, raw_result)
    swiss_record_prob = None
    for stage in spec["stages"]:
        if (stage["kind"] == "swiss" and len(stage.get("entrants", [])) == 16 and
                stage.get("wins_to_advance") == 3 and stage.get("losses_to_eliminate") == 3):
            swiss_record_prob = extract_swiss_record_probabilities(raw_result["stage_rank_prob"][stage["id"]])

    # Preserve every core distribution and raw count, including explicit absent
    # champion/final targets in qualification-only phases.
    result = dict(raw_result)
    result.update({
        "spec_id": spec.get("id"),
        "execution_time_seconds": round(t_elapsed, 4),
        "swiss_record_prob": swiss_record_prob,
        "series_score_distributions": scorelines,
        "conditional_score_distributions": conditional_score_distributions,
        "score_law_metadata": {
            "kind": ("conditional_probability_bins" if conditional_score_distributions is not None
                     else "fixed_markov_pooled" if default_score_law else "supplied_pooled"),
            "probability_unit": "sampled_winner_series_win",
            "score_unit": "losing_maps_given_winner",
            "simulation_approximation": conditional_score_distributions is not None,
            "bin_semantics": ("left_inclusive_last_right_inclusive"
                              if conditional_score_distributions is not None else None),
            "default_p_series": 0.5 if default_score_law else None,
            "default_beta_momentum": 0.35 if default_score_law else None,
        },
        "pair_table": {
            "probability_unit": "series_win",
            "rows": [{"team_a": a, "team_b": b, "best_of": bo, "p": p}
                     for (a, b, bo), p in clean_probs.items()],
        },
        "pair_scoreline_laws": _pair_scoreline_laws(
            clean_probs, scorelines, conditional_score_distributions, team_strength_draws, copula_rho,
        ),
    })
    result["numerical_uncertainty"] = summarize_tournament_numerics(result)
    return result


@dataclass(frozen=True, slots=True)
class TournamentBracketSimulator:
    """Configurable tournament integration wrapper."""

    simulations: int = 10000
    seed: int = 20261005
    integration: str = "monte_carlo"
    copula_rho: float | None = None

    def simulate(
        self,
        spec: dict[str, Any],
        probabilities: Mapping[tuple[str, str, int], float],
        scorelines: Mapping[int, Sequence[float]] | None = None,
        conditional_score_distributions: Mapping[str, Any] | None = None,
        *,
        team_strength_draws: Mapping[str, Sequence[float]] | None = None,
    ) -> dict[str, Any]:
        """Execute bracket simulations using configured defaults."""
        return simulate_tournament_bracket(
            spec=spec,
            probabilities=probabilities,
            scorelines=scorelines,
            conditional_score_distributions=conditional_score_distributions,
            team_strength_draws=team_strength_draws,
            simulations=self.simulations,
            seed=self.seed,
            integration=self.integration,
            copula_rho=self.copula_rho,
        )
