"""Multi-category scoring rules, ground-truth extraction, and network flow verification.

Implements strictly proper scoring rules for full tournament distributions:
1. Tournament Champion: Negative Log Loss and multi-class Brier score.
2. Grand Final Reach: Multi-class Brier score over finalists.
3. Swiss Stage Qualification: Advancement Brier score and Ranked Probability Score (RPS)
   over final records (3-0, 3-1, 3-2, 2-3, 1-3, 0-3).
4. Series Length & Scoreline Calibration: RPS on series games played and exact scoreline LogLoss.
5. Network Flow Conservation: Linear programming verification via SciPy `linprog(method='highs')`
   ensuring that marginal team reach probabilities obey bracket capacity constraints without mass leaks.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

from src.models.tournament_pair_adapter import normalize_team_id
from src.analysis.tournament_numerics import query_tournament_joint_numerics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_RESULTS_PATH = (
    PROJECT_ROOT / "data/artifacts/leaguepedia-tournament-rules-20260908/series_results.csv"
)

ORDERED_SWISS_RECORDS: tuple[str, ...] = ("3-0", "3-1", "3-2", "2-3", "1-3", "0-3")


@dataclass(frozen=True, slots=True)
class TournamentMatchOutcome:
    """Historical completed playoff or bracket match."""

    stage: str
    round_name: str
    team1: str
    team2: str
    score1: int
    score2: int
    best_of: int
    winner: str


@dataclass(frozen=True, slots=True)
class TournamentGroundTruth:
    """Historical ground-truth outcomes for a tournament edition."""

    event_title: str
    champion: str
    finalists: tuple[str, ...]
    swiss_qualifiers: tuple[str, ...] | None
    swiss_records: dict[str, str] | None
    playoff_matches: tuple[TournamentMatchOutcome, ...]


def extract_tournament_ground_truth(
    event_title: str,
    series_results_path: Path | str | None = None,
) -> TournamentGroundTruth:
    """Extract historical tournament outcomes from series_results.csv.

    Parameters
    ----------
    event_title : str
        Canonical event title in series_results.csv (e.g. '2023 Season World Championship/Main Event').
    series_results_path : Path | str | None
        Optional explicit path to series_results.csv.

    Returns
    -------
    TournamentGroundTruth
        Extracted champion, finalists, Swiss qualifiers, records, and matches.
    """
    path = Path(series_results_path) if series_results_path is not None else DEFAULT_SERIES_RESULTS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Series results file not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for r in reader:
            if r.get("event_title") == event_title and r.get("status") == "completed":
                rows.append(r)

    if not rows:
        raise ValueError(f"No completed match records found for event: {event_title!r}")

    # Track Swiss matches and records
    swiss_wins: dict[str, int] = defaultdict(int)
    swiss_losses: dict[str, int] = defaultdict(int)
    has_swiss = any(r.get("stage") == "swiss" for r in rows)

    playoff_matches: list[TournamentMatchOutcome] = []
    final_match: dict[str, str] | None = None

    for r in rows:
        t1 = normalize_team_id(r.get("team1", ""))
        t2 = normalize_team_id(r.get("team2", ""))
        w_side = r.get("winner_side", "")
        if not t1 or not t2 or w_side not in ("1", "2"):
            continue

        s1 = int(r.get("team1_score", 0))
        s2 = int(r.get("team2_score", 0))
        bo = int(r.get("best_of", 1))
        winner = t1 if w_side == "1" else t2
        stage = r.get("stage", "")
        section = r.get("section_path", "")

        if stage == "swiss":
            if w_side == "1":
                swiss_wins[t1] += 1
                swiss_losses[t2] += 1
            else:
                swiss_losses[t1] += 1
                swiss_wins[t2] += 1

        elif stage == "playoffs" or "playoff" in stage.lower() or "Final" in section:
            match_outcome = TournamentMatchOutcome(
                stage=stage,
                round_name=section,
                team1=t1,
                team2=t2,
                score1=s1,
                score2=s2,
                best_of=bo,
                winner=winner,
            )
            playoff_matches.append(match_outcome)
            if "Finals" in section or section == "Finals":
                final_match = r

    # Determine champion and finalists
    if final_match is not None:
        f_t1 = normalize_team_id(final_match.get("team1", ""))
        f_t2 = normalize_team_id(final_match.get("team2", ""))
        f_w = f_t1 if final_match.get("winner_side") == "1" else f_t2
        champion = f_w
        finalists = (min(f_t1, f_t2), max(f_t1, f_t2))
    elif playoff_matches:
        last = playoff_matches[-1]
        champion = last.winner
        finalists = (min(last.team1, last.team2), max(last.team1, last.team2))
    else:
        raise ValueError(f"Could not determine champion/finals for {event_title!r}")

    # Process Swiss records if present
    swiss_records: dict[str, str] | None = None
    swiss_qualifiers: tuple[str, ...] | None = None
    if has_swiss:
        swiss_records = {}
        qualifiers_list = []
        for team in sorted(set(swiss_wins.keys()) | set(swiss_losses.keys())):
            w = swiss_wins[team]
            l = swiss_losses[team]
            record_str = f"{w}-{l}"
            swiss_records[team] = record_str
            if w >= 3:
                qualifiers_list.append(team)
        swiss_qualifiers = tuple(sorted(qualifiers_list))

    return TournamentGroundTruth(
        event_title=event_title,
        champion=champion,
        finalists=finalists,
        swiss_qualifiers=swiss_qualifiers,
        swiss_records=swiss_records,
        playoff_matches=tuple(playoff_matches),
    )


def _probabilities(values: Mapping, *, mass: float | None = None,
                   categories: Sequence | None = None, name: str = "distribution") -> dict:
    """Validate the original law; absent categories are not structural zeros."""
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a nonempty probability mapping")
    if categories is not None and set(values) != set(categories):
        raise ValueError(f"{name} has missing or unexpected categories")
    result = {}
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} has a nonnumeric probability")
        try:
            p = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} has a nonnumeric probability") from exc
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"{name} probabilities must be finite and in [0,1]")
        result[key] = p
    if mass is not None and abs(math.fsum(result.values()) - mass) > 1e-7:
        raise ValueError(f"{name} mass must equal {mass}")
    return result


def _log_loss(p: float) -> float:
    return math.inf if p == 0 else -math.log(p)


def _rps(probabilities: Mapping, actual: Any, categories: Sequence) -> float:
    law = _probabilities(probabilities, mass=1, categories=categories)
    if actual not in categories:
        raise ValueError("Observed category is absent from the declared law")
    if len(categories) == 1:
        return 0.0
    cdf = 0.0
    actual_index = categories.index(actual)
    errors = []
    for index, category in enumerate(categories[:-1]):
        cdf += law[category]
        errors.append((cdf - float(index >= actual_index)) ** 2)
    return math.fsum(errors) / (len(categories) - 1)


def compute_champion_log_loss(champion_prob: Mapping[str, float], actual_winner: str) -> float:
    """Unclipped winner loss; a genuine zero is infinite, not a missing member."""
    law = _probabilities(champion_prob, mass=1, name="champion law")
    if actual_winner not in law:
        raise ValueError("Actual champion is absent from the probability law")
    return _log_loss(law[actual_winner])


def compute_champion_brier(champion_prob: Mapping[str, float], actual_winner: str) -> float:
    law = _probabilities(champion_prob, mass=1, name="champion law")
    if actual_winner not in law:
        raise ValueError("Actual champion is absent from the probability law")
    return math.fsum((p - float(team == actual_winner)) ** 2 for team, p in law.items())


def _target_members(law: Mapping, observed: Sequence, name: str) -> set:
    if isinstance(observed, (str, bytes, Mapping)) or not isinstance(observed, Sequence):
        raise ValueError(f"{name} must be a sequence of entrant identities")
    try:
        members = set(observed)
    except TypeError as exc:
        raise ValueError(f"{name} contains malformed team members") from exc
    if len(members) != len(observed) or not members.issubset(law):
        raise ValueError(f"{name} contains duplicate or absent team members")
    return members


def compute_finalists_brier(final_prob: Mapping[str, float], actual_finalists: Sequence[str]) -> float:
    """Half the teamwise squared error for the two-finalist inclusion law."""
    law = _probabilities(final_prob, mass=2, name="final inclusion law")
    members = _target_members(law, actual_finalists, "Actual finalists")
    if len(members) != 2:
        raise ValueError("Exactly two actual finalists are required")
    return .5 * math.fsum((p - float(team in members)) ** 2 for team, p in law.items())


def compute_advancement_brier(advance_prob: Mapping[str, float],
                              actual_qualifiers: Sequence[str]) -> float:
    """Mean squared inclusion error over all team targets, not only qualifiers."""
    law = _probabilities(advance_prob, mass=len(actual_qualifiers), name="advancement law")
    members = _target_members(law, actual_qualifiers, "Actual qualifiers")
    return math.fsum((p - float(team in members)) ** 2 for team, p in law.items()) / len(law)


def compute_swiss_record_rps(swiss_record_prob: Mapping[str, Mapping[str, float]],
                             actual_records: Mapping[str, str]) -> float:
    if not actual_records or set(swiss_record_prob) != set(actual_records):
        raise ValueError("Swiss record target and forecast team sets must be identical")
    return float(np.mean([_rps(swiss_record_prob[team], record, ORDERED_SWISS_RECORDS)
                          for team, record in actual_records.items()]))


def _terminal_scores(best_of: int) -> tuple[tuple[int, int], ...]:
    if isinstance(best_of, bool) or best_of not in (1, 3, 5):
        raise ValueError("Score laws require best_of 1, 3, or 5")
    needed = best_of // 2 + 1
    return tuple((needed, lost) for lost in range(needed)) + tuple(
        (lost, needed) for lost in range(needed))


def compute_series_length_rps(length_prob: Mapping[int, float], actual_length: int,
                              best_of: int = 5) -> float:
    _terminal_scores(best_of)
    return _rps(length_prob, actual_length, tuple(range(best_of // 2 + 1, best_of + 1)))


def compute_scoreline_log_loss(score_pmf: Mapping[tuple[int, int], float],
                               actual_score: tuple[int, int]) -> float:
    if (not isinstance(actual_score, tuple) or len(actual_score) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in actual_score)):
        raise ValueError("Actual score must be a pair of integer map wins")
    needed = max(actual_score)
    law = _probabilities(score_pmf, mass=1, categories=_terminal_scores(2 * needed - 1),
                         name="terminal score law")
    if actual_score not in law:
        raise ValueError("Observed score is absent from the terminal score law")
    return _log_loss(law[actual_score])


def verify_network_flow_conservation(
    champion_prob: Mapping[str, float],
    final_prob: Mapping[str, float] | None = None,
    advance_prob: Mapping[str, float] | None = None,
    tolerance: float = 1e-4,
    *,
    advance_capacity: int | None = None,
) -> dict[str, Any]:
    """Verify ORIGINAL fixed marginals, including mass, membership and reach.

    HiGHS cannot repair the inputs: each variable is fixed to its supplied value.
    Capacity and reach constraints admit only the declared numerical tolerance;
    independent direct checks remain authoritative even if the solver tolerates
    a larger residual. An undeclared advancement capacity must already be an
    integer within tolerance, never a repaired or rescaled mass.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Flow tolerance must be finite and nonnegative")
    laws = {"champion": champion_prob}
    if final_prob is not None:
        laws["final"] = final_prob
    if advance_prob is not None:
        laws["advance"] = advance_prob
    teams = set(champion_prob)
    errors = []
    totals, mass_violations, bounds_violations, membership = {}, {}, {}, {}
    arrays = {}
    for stage, law in laws.items():
        missing, extra = teams - set(law), set(law) - teams
        membership[stage] = {"missing": sorted(missing), "extra": sorted(extra)}
        if missing or extra or not law:
            errors.append(f"{stage}: inconsistent or empty team set")
        try:
            values = np.asarray([float(value) for value in law.values()], dtype=float)
            finite = bool(np.all(np.isfinite(values)))
            bounds_violations[stage] = (
                float(max(0., np.max(-values), np.max(values - 1))) if finite and len(values) else None)
            totals[stage] = float(math.fsum(values)) if finite else None
            if not finite or any(isinstance(value, (bool, np.bool_)) for value in law.values()):
                errors.append(f"{stage}: nonfinite or nonnumeric probability")
            elif bounds_violations[stage] is not None and bounds_violations[stage] > 0:
                errors.append(f"{stage}: probability outside [0,1]")
            if finite and not missing and not extra:
                arrays[stage] = np.asarray([law[team] for team in sorted(teams)], dtype=float)
        except (TypeError, ValueError, OverflowError):
            totals[stage] = bounds_violations[stage] = None
            errors.append(f"{stage}: nonnumeric probability")
    capacities = {"champion": 1., "final": 2.}
    if advance_prob is not None:
        total = totals["advance"]
        if advance_capacity is None:
            # An integer hypothesis is checked against the original mass below.
            capacity = int(math.floor(total + .5)) if total is not None else None
        else:
            capacity = advance_capacity
        if (isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or
                not 0 <= capacity <= len(teams)):
            errors.append("advance: capacity must be an integer in [0,team_count]")
            capacity = None
        capacities["advance"] = capacity
    elif advance_capacity is not None:
        errors.append("advance: declared capacity without advancement marginals")
    for stage in laws:
        total, capacity = totals[stage], capacities[stage]
        violation = abs(total - capacity) if total is not None and capacity is not None else None
        mass_violations[stage] = violation
        if violation is None or violation > tolerance:
            errors.append(f"{stage}: original mass does not equal declared capacity")
    reach_violations = {}
    chain = [stage for stage in ("champion", "final", "advance") if stage in arrays]
    for lower, upper in zip(chain, chain[1:]):
        violation = float(max(0., np.max(arrays[lower] - arrays[upper]))) if teams else 0.
        reach_violations[f"{lower}_to_{upper}"] = violation
        if violation > tolerance:
            errors.append(f"{lower}_to_{upper}: reach inequality violated")
    max_viol = max(reach_violations.values(), default=0.)
    result = {"feasible": False, "status": 2, "max_flow_violation": max_viol,
              "tolerance": tolerance, "original_mass": totals, "capacities": capacities,
              "mass_violations": mass_violations, "bounds_violations": bounds_violations,
              "member_set_violations": membership, "reach_violations": reach_violations,
              "failure_reasons": errors}
    if errors:
        return result
    stages = list(laws)
    k, size = len(teams), len(teams) * len(stages)
    values = np.concatenate([arrays[stage] for stage in stages])
    constraints, rhs = [], []
    for index, stage in enumerate(stages):
        row = np.zeros(size)
        row[index * k:(index + 1) * k] = 1
        constraints.extend((row, -row))
        rhs.extend((capacities[stage] + tolerance, -capacities[stage] + tolerance))
    for index in range(len(stages) - 1):
        for team_index in range(k):
            row = np.zeros(size)
            row[index * k + team_index] = 1
            row[(index + 1) * k + team_index] = -1
            constraints.append(row)
            rhs.append(tolerance)
    lp = linprog(np.zeros(size), A_ub=np.asarray(constraints), b_ub=np.asarray(rhs),
                 bounds=[(value, value) for value in values], method="highs")
    result.update(feasible=bool(lp.success), status=int(lp.status), solver_message=lp.message)
    if not lp.success:
        errors.append("HiGHS rejected the fixed supplied marginals")
    return result


def _pair_probabilities(table: Mapping) -> dict[tuple[str, str, int], float]:
    if "rows" in table:
        if table.get("probability_unit") != "series_win":
            raise ValueError("Pair table must declare series_win probability units")
        items = [((row["team_a"], row["team_b"], row["best_of"]), row["p"])
                 for row in table["rows"]]
    else:
        items = list(table.items())
    probabilities = {}
    for key, value in items:
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("Pair probability key must contain team_a, team_b, best_of")
        a, b, bo = key
        _terminal_scores(bo)
        if a == b or key in probabilities:
            raise ValueError("Duplicate or self pair probability")
        p = _probabilities({key: value})[key]
        if (b, a, bo) in probabilities and abs(p + probabilities[b, a, bo] - 1) > 1e-7:
            raise ValueError("Reverse pair probabilities are inconsistent")
        probabilities[key] = p
    return probabilities


def _actual_score_law(model: Any, forecast: Mapping, p: float, bo: int) -> dict:
    if model is not None:
        if isinstance(model, Mapping):
            from src.models.tournament_scorelines import LearnedConditionalScorelineModel
            model = LearnedConditionalScorelineModel.from_dict(model)
        return _probabilities(model.scoreline_pmf(p, bo), mass=1,
                              categories=_terminal_scores(bo), name="actual conditional score law")
    # Legacy output contains the actual engine law, not necessarily the learned
    # continuous model. Reproduce its winner-probability binning exactly.
    if bo == 1:
        return {(1, 0): p, (0, 1): 1 - p}
    conditional = forecast.get("conditional_score_distributions")
    pooled = forecast.get("series_score_distributions")
    if conditional is None and pooled is None:
        raise LookupError("Actual conditional or pooled score law is unavailable")
    needed = bo // 2 + 1
    pmf = {}
    for winner_p, winner_a in ((p, True), (1 - p, False)):
        if conditional is not None:
            bins = conditional["probability_bins"]
            if (len(bins) < 2 or bins[0] != 0 or bins[-1] != 1 or
                    any(not math.isfinite(x) for x in bins) or
                    any(a >= b for a, b in zip(bins, bins[1:]))):
                raise ValueError("Malformed conditional score probability bins")
            rows = conditional["distributions"]
            rows = rows.get(bo, rows.get(str(bo)))
            if rows is None or len(rows) != len(bins) - 1:
                raise ValueError("Missing conditional score law format or bins")
            index = min(int(np.searchsorted(bins, winner_p, side="right")) - 1, len(bins) - 2)
            weights = rows[index]
        else:
            weights = pooled.get(bo, pooled.get(str(bo)))
            if weights is None:
                raise LookupError(f"Actual score law unavailable for BO{bo}")
        weights = _probabilities(dict(enumerate(weights)), mass=1, categories=range(needed))
        for lost, probability in weights.items():
            pmf[(needed, lost) if winner_a else (lost, needed)] = winner_p * probability
    return pmf


def _compiled_pair_laws(payload: Mapping) -> dict:
    """Validate the actual finite-draw engine law, not a law at mean strength."""
    if (payload.get("probability_unit") != "series_win" or
            payload.get("scope") != "frozen_origin_unconditional_hypothetical_pair"):
        raise ValueError("Actual pair-score laws require series-win units and frozen-origin scope")
    laws = {}
    for row in payload["rows"]:
        a, b, bo = row["team_a"], row["team_b"], row["best_of"]
        key = (a, b, bo)
        if a == b or key in laws:
            raise ValueError("Duplicate or self pair-score law")
        p = _probabilities({key: row["p"]})[key]
        try:
            scores = {tuple(int(x) for x in score.split("-")): value
                      for score, value in row["scoreline_pmf"].items()}
            lengths = {int(games): value for games, value in row["series_length_pmf"].items()}
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Malformed compiled pair-score categories") from exc
        if len(scores) != len(row["scoreline_pmf"]) or len(lengths) != len(row["series_length_pmf"]):
            raise ValueError("Duplicate encoded pair-score categories")
        scores = _probabilities(scores, mass=1, categories=_terminal_scores(bo))
        lengths = _probabilities(lengths, mass=1, categories=range(bo // 2 + 1, bo + 1))
        needed = bo // 2 + 1
        if abs(math.fsum(value for (wins_a, _), value in scores.items()
                         if wins_a == needed) - p) > 1e-7:
            raise ValueError("Compiled pair-score law does not preserve its integrated series probability")
        for games, probability in lengths.items():
            if abs(math.fsum(value for score, value in scores.items()
                             if sum(score) == games) - probability) > 1e-7:
                raise ValueError("Compiled length law disagrees with its scoreline law")
        reverse = laws.get((b, a, bo))
        if reverse is not None and (
                abs(p + reverse[0] - 1) > 1e-7 or
                any(abs(value - reverse[1][score[::-1]]) > 1e-7 for score, value in scores.items())):
            raise ValueError("Inconsistent reverse compiled pair-score laws")
        laws[key] = (p, scores, lengths)
    return laws


def _placement_rps(forecast: Mapping, observed: Mapping, teams: Sequence[str]) -> float:
    probabilities = forecast["placement_prob"]
    if set(probabilities) != set(teams) or set(observed) != set(teams):
        raise ValueError("Placement team sets must equal the event team set")
    categories = forecast.get("placement_categories")
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Official ordered placement categories are required")
    bands = []
    for label in categories:
        try:
            parts = tuple(int(x) for x in label.split("-"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("Malformed official placement band") from exc
        if len(parts) not in (1, 2):
            raise ValueError("Malformed official placement band")
        bands.append((parts[0], parts[-1]))
    cursor = 1
    for lo, hi in bands:
        if lo != cursor or hi < lo:
            raise ValueError("Official placement bands must be ordered and contiguous")
        cursor = hi + 1
    if cursor != len(teams) + 1:
        raise ValueError("Official placement bands must cover all entrant places")
    scores = []
    for team in teams:
        band = tuple(observed[team])
        if band not in bands:
            raise ValueError("Observed placement is not an official tied band")
        scores.append(_rps(probabilities[team], categories[bands.index(band)], categories))
    return float(np.mean(scores))


def _joint_set_law(law: Mapping, observed: Sequence[str], teams: Sequence[str],
                   name: str, cardinality: int | None = None) -> tuple[dict, frozenset]:
    """Validate sparse categories, without treating absent legal sets as impossible."""
    if isinstance(observed, (str, bytes)):
        raise ValueError(f"{name} target must be a sequence of entrant identities")
    _target_members(dict.fromkeys(teams), observed, f"{name} target")
    target = frozenset(observed)
    if cardinality is not None and len(target) != cardinality:
        raise ValueError(f"{name} target requires exactly {cardinality} members")
    probabilities = _probabilities(law, mass=1, name=f"{name} law")
    canonical = {}
    for key, p in probabilities.items():
        try:
            members = json.loads(key) if isinstance(key, str) else key
            if not isinstance(members, (list, tuple, set, frozenset)):
                raise ValueError("Category must encode an array of entrant identities")
            if any(not isinstance(member, str) for member in members):
                raise ValueError("Category members must be entrant identities")
            category = frozenset(members)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Malformed {name} category") from exc
        if (len(category) != len(members) or len(category) != len(target) or
                not category.issubset(teams) or category in canonical):
            raise ValueError(f"Malformed or duplicate {name} category")
        canonical[category] = p
    return canonical, target


def _joint_set_scores(law: Mapping, observed: Sequence[str], teams: Sequence[str],
                      name: str, cardinality: int | None = None) -> tuple[float, float]:
    canonical, target = _joint_set_law(law, observed, teams, name, cardinality)
    probability = canonical.get(target, 0.)
    # Unlisted legal categories have zero issued probability. Include the actual
    # target's one-hot mass even when it was not hit; do not add a probability floor.
    brier = math.fsum(p * p for p in canonical.values()) - 2 * probability + 1
    return _log_loss(probability), brier


def compute_joint_finalists_log_loss(law: Mapping, finalists: Sequence[str],
                                    teams: Sequence[str]) -> float:
    return _joint_set_scores(law, finalists, teams, "joint finalist", 2)[0]


def compute_joint_finalists_brier(law: Mapping, finalists: Sequence[str],
                                 teams: Sequence[str]) -> float:
    return _joint_set_scores(law, finalists, teams, "joint finalist", 2)[1]


def compute_joint_qualification_log_loss(law: Mapping, qualifiers: Sequence[str],
                                        teams: Sequence[str]) -> float:
    return _joint_set_scores(law, qualifiers, teams, "joint qualification")[0]


def compute_joint_qualification_brier(law: Mapping, qualifiers: Sequence[str],
                                     teams: Sequence[str]) -> float:
    return _joint_set_scores(law, qualifiers, teams, "joint qualification")[1]


def score_research_event(event: Mapping[str, Any], forecast: Mapping[str, Any],
                         labels: Mapping[str, Any], pair_table: Mapping | None = None,
                         scoreline_model: Any = None) -> dict[str, Any]:
    """Score only real declared targets and actual laws; preserve infinite losses.

    Missing supported forecasts remain named unavailable. Malformed supplied
    laws raise, never silently shrink a target set or invent uniform predictions.
    Serialization of nonfinite losses belongs to the reporting boundary.
    """
    event_id, edition_id = event["id"], event["edition_id"]
    if labels.get("event_id") != event_id or labels.get("edition_id") != edition_id:
        raise ValueError("Observed labels belong to a different event or edition")
    teams = event["teams"]
    if not teams or len(set(teams)) != len(teams):
        raise ValueError("Event team set must be nonempty and unique")
    metrics = ("advance_brier", "placement_rps", "champion_log_loss", "champion_brier",
               "finalists_brier", "joint_finalists_log_loss", "joint_finalists_brier",
               "swiss_record_rps", "joint_qualification_log_loss", "joint_qualification_brier",
               "series_length_rps", "scoreline_log_loss")
    result = dict.fromkeys(metrics)
    reasons, counts = {}, dict.fromkeys(metrics, 0)
    if labels.get("champion") is not None:
        if labels["champion"] not in teams:
            raise ValueError("Observed champion is not an entrant")
        counts.update(champion_log_loss=1, champion_brier=1)
    if labels.get("finalists") is not None:
        members = _target_members(dict.fromkeys(teams), labels["finalists"], "Observed finalists")
        if len(members) != 2:
            raise ValueError("Exactly two observed finalists are required")
        counts["finalists_brier"] = len(teams)
        counts.update(joint_finalists_log_loss=1, joint_finalists_brier=1)
    for observed in (labels.get("qualifiers") or {}).values():
        _target_members(dict.fromkeys(teams), observed, "Observed qualifiers")
        counts["advance_brier"] += len(teams)
        counts["joint_qualification_log_loss"] += 1
        counts["joint_qualification_brier"] += 1
    if labels.get("placement_bands") is not None:
        counts["placement_rps"] = len(labels["placement_bands"])
    if labels.get("swiss_records") is not None:
        counts["swiss_record_rps"] = len(labels["swiss_records"])
    for match in labels.get("matches") or []:
        a, b, bo = match["team_a"], match["team_b"], match["best_of"]
        score = (match["wins_a"], match["wins_b"])
        if (a not in teams or b not in teams or a == b or
                any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in score) or
                score not in _terminal_scores(bo)):
            raise ValueError("Observed series has invalid teams or a nonterminal score")
        counts["series_length_rps"] += 1
        counts["scoreline_log_loss"] += 1

    def unavailable(keys, reason):
        for key in keys:
            reasons[key] = reason

    champion = labels.get("champion")
    if champion is None:
        unavailable(("champion_log_loss", "champion_brier"), "No declared champion target")
    elif forecast.get("champion_prob") is None:
        unavailable(("champion_log_loss", "champion_brier"), "Champion forecast law unavailable")
    else:
        law = _probabilities(forecast["champion_prob"], mass=1, categories=teams)
        result["champion_log_loss"] = compute_champion_log_loss(law, champion)
        result["champion_brier"] = compute_champion_brier(law, champion)
    finalists = labels.get("finalists")
    if finalists is None:
        unavailable(("finalists_brier",), "No declared finalist target")
    elif forecast.get("final_prob") is None:
        unavailable(("finalists_brier",), "Finalist inclusion law unavailable")
    else:
        law = _probabilities(forecast["final_prob"], mass=2, categories=teams)
        result["finalists_brier"] = compute_finalists_brier(law, finalists)

    joint_final_keys = ("joint_finalists_log_loss", "joint_finalists_brier")
    if finalists is None:
        unavailable(joint_final_keys, "No declared finalist-pair target")
    elif forecast.get("joint_final_prob") is None:
        unavailable(joint_final_keys, "Joint finalist-pair law unavailable")
    else:
        losses = _joint_set_scores(forecast["joint_final_prob"], finalists, teams,
                                  "joint finalist", 2)
        result.update(zip(joint_final_keys, losses))
    qualifiers = labels.get("qualifiers") or {}
    advances = forecast.get("advance_prob") or {}
    if not qualifiers:
        unavailable(("advance_brier", "joint_qualification_log_loss", "joint_qualification_brier"),
                    "No declared qualification targets")
    elif any(stage not in advances or advances[stage] is None for stage in qualifiers):
        unavailable(("advance_brier",), "A declared stage qualification law is unavailable")
    else:
        scores = []
        for stage, observed in qualifiers.items():
            law = _probabilities(advances[stage], categories=teams)
            scores.append(compute_advancement_brier(law, observed))
        result["advance_brier"] = float(np.mean(scores))
    if qualifiers:
        joint = forecast.get("joint_advance_prob")
        if joint is None or any(not joint.get(stage) for stage in qualifiers):
            unavailable(("joint_qualification_log_loss", "joint_qualification_brier"),
                        "Joint qualification-set law unavailable")
        else:
            losses = [_joint_set_scores(joint[stage], observed, teams, "joint qualification")
                      for stage, observed in qualifiers.items()]
            result["joint_qualification_log_loss"] = float(np.mean([value[0] for value in losses]))
            result["joint_qualification_brier"] = float(np.mean([value[1] for value in losses]))
    observed_placements = labels.get("placement_bands")
    if observed_placements is None:
        unavailable(("placement_rps",), "No declared official placement-band targets")
    elif forecast.get("placement_prob") is None:
        unavailable(("placement_rps",), "Official placement-band forecast law unavailable")
    else:
        result["placement_rps"] = _placement_rps(forecast, observed_placements, teams)
    records = labels.get("swiss_records")
    if records is None:
        unavailable(("swiss_record_rps",), "No declared Swiss record targets")
    elif forecast.get("swiss_record_prob") is None:
        unavailable(("swiss_record_rps",), "Swiss record law unavailable")
    else:
        if set(records) != set(teams):
            raise ValueError("Swiss record labels must cover the event teams")
        result["swiss_record_rps"] = compute_swiss_record_rps(forecast["swiss_record_prob"], records)

    matches = labels.get("matches") or []
    table = pair_table if pair_table is not None else forecast.get(
        "pair_table", forecast.get("pair_probabilities"))
    model = scoreline_model if scoreline_model is not None else forecast.get("scoreline_model")
    compiled = forecast.get("pair_scoreline_laws")
    series_keys = ("series_length_rps", "scoreline_log_loss")
    if not matches:
        unavailable(series_keys, "No declared completed series targets")
    elif table is None and compiled is None:
        unavailable(series_keys, "Actual pair probability table unavailable")
    else:
        actual_laws = _compiled_pair_laws(compiled) if compiled is not None else None
        pairs = _pair_probabilities(table) if actual_laws is None else None
        length_scores, score_losses = [], []
        missing = []
        for match in matches:
            a, b, bo = match["team_a"], match["team_b"], match["best_of"]
            score = (match["wins_a"], match["wins_b"])
            if actual_laws is not None:
                if (a, b, bo) in actual_laws:
                    p, law, lengths = actual_laws[a, b, bo]
                elif (b, a, bo) in actual_laws:
                    reverse_p, reverse_law, lengths = actual_laws[b, a, bo]
                    p = 1 - reverse_p
                    law = {terminal[::-1]: probability for terminal, probability in reverse_law.items()}
                else:
                    missing.append(f"Missing actual finite-draw pair-score law for {a}/{b}/BO{bo}")
                    continue
            else:
                if (a, b, bo) in pairs:
                    p = pairs[a, b, bo]
                elif (b, a, bo) in pairs:
                    p = 1 - pairs[b, a, bo]
                else:
                    missing.append(f"Missing pair probability for {a}/{b}/BO{bo}")
                    continue
                try:
                    law = _actual_score_law(model, forecast, p, bo)
                except LookupError as exc:
                    missing.append(str(exc))
                    continue
                law = _probabilities(law, mass=1, categories=_terminal_scores(bo))
                needed = bo // 2 + 1
                if abs(math.fsum(value for (a_wins, _), value in law.items()
                                 if a_wins == needed) - p) > 1e-7:
                    raise ValueError("Actual conditional score law does not preserve its pair series probability")
                lengths = dict.fromkeys(range(bo // 2 + 1, bo + 1), 0.)
                for terminal, probability in law.items():
                    lengths[sum(terminal)] += probability
            length_scores.append(compute_series_length_rps(lengths, sum(score), bo))
            score_losses.append(compute_scoreline_log_loss(law, score))
        if missing:
            unavailable(series_keys, "; ".join(sorted(set(missing))))
        else:
            result["series_length_rps"] = float(np.mean(length_scores))
            result["scoreline_log_loss"] = float(np.mean(score_losses))
    for metric, value in result.items():
        if value is not None and not math.isfinite(value):
            reasons[metric] = "Observed target has zero forecast probability; unclipped log loss is infinite"
    result.update(event_id=event_id, edition_id=edition_id, unavailable_reasons=reasons,
                  target_counts=counts)
    result["series_score_scope"] = "frozen_origin_unconditional_hypothetical_pair"
    result["actual_joint_targets"] = {
        "finalists": (json.dumps(sorted(finalists), separators=(",", ":"), ensure_ascii=False)
                      if finalists is not None else None),
        "qualifiers": {stage: json.dumps(sorted(observed), separators=(",", ":"), ensure_ascii=False)
                       for stage, observed in qualifiers.items()},
    }
    integration = forecast.get("integration") or {}
    if isinstance(integration, Mapping) and integration.get("joint_target_design") is not None:
        design = integration["joint_target_design"]
        if not isinstance(design, Mapping) or set(design.get("teams") or []) != set(teams):
            raise ValueError("Joint numerical design entrants must equal the scored event entrants")
        # Label-selected diagnostics stay in scoring output, never forecasting
        # input. Numerical bounds do not floor issued zeros or replace losses.
        result["actual_joint_numerics"] = {
            "finalists": (query_tournament_joint_numerics(
                forecast, "joint_final_prob", finalists) if finalists is not None else None),
            "qualifiers": {stage: query_tournament_joint_numerics(
                forecast, "joint_advance_prob", observed, stage=stage)
                for stage, observed in qualifiers.items()},
        }
    result["descriptive_metrics"] = ["joint_finalists_log_loss", "joint_finalists_brier",
                                     "joint_qualification_log_loss", "joint_qualification_brier"]
    return result


def evaluate_tournament_edition(edition_id: str, simulation_results: Mapping[str, Any],
                                ground_truth: TournamentGroundTruth, *,
                                pair_table: Mapping | None = None,
                                scoreline_model: Any = None) -> dict[str, Any]:
    """Legacy report contract backed by actual pair probabilities and score law."""
    teams = list(simulation_results.get("champion_prob") or simulation_results.get("final_prob") or {})
    labels = {"event_id": edition_id, "edition_id": edition_id,
              "champion": ground_truth.champion, "finalists": ground_truth.finalists,
              "qualifiers": ({"swiss": ground_truth.swiss_qualifiers}
                             if ground_truth.swiss_qualifiers is not None else {}),
              "swiss_records": ground_truth.swiss_records, "placement_bands": None,
              "matches": [{"team_a": m.team1, "team_b": m.team2, "best_of": m.best_of,
                           "wins_a": m.score1, "wins_b": m.score2}
                          for m in ground_truth.playoff_matches]}
    scores = score_research_event({"id": edition_id, "edition_id": edition_id, "teams": teams},
                                 simulation_results, labels, pair_table, scoreline_model)
    flow = verify_network_flow_conservation(simulation_results["champion_prob"],
                                           simulation_results.get("final_prob"),
                                           (simulation_results.get("advance_prob") or {}).get("swiss"),
                                           advance_capacity=(len(ground_truth.swiss_qualifiers)
                                                             if ground_truth.swiss_qualifiers is not None else None))
    result = {key: scores[key] for key in ("champion_log_loss", "champion_brier", "finalists_brier",
                                         "advance_brier", "swiss_record_rps")}
    result.update(edition_id=edition_id, simulations=simulation_results.get("simulations"),
                  actual_champion=ground_truth.champion,
                  champion_prob=simulation_results["champion_prob"][ground_truth.champion],
                  actual_finalists=list(ground_truth.finalists),
                  playoff_series_length_rps=scores["series_length_rps"],
                  playoff_scoreline_log_loss=scores["scoreline_log_loss"],
                  unavailable_reasons=scores["unavailable_reasons"],
                  network_flow_feasible=flow["feasible"], max_flow_violation=flow["max_flow_violation"])
    return result


def tournament_study_decision(scores: Sequence[Mapping[str, Any]], series_guard: Mapping[str, Any],
                              minimum_editions: int = 8, bootstraps: int = 5000,
                              seed: int = 20261006) -> dict[str, Any]:
    """Paired edition-cluster inference, never event-row pseudoreplication.

    Within an edition, genuine phases are target-weighted; primary estimates
    weight editions equally. The sensitivity retains team-target weighting and
    resamples entire editions. Missing mandatory arms, laws, or equal target
    counts cannot create a handpicked passing intersection.
    """
    if (isinstance(minimum_editions, bool) or not isinstance(minimum_editions, int) or
            minimum_editions < 1 or isinstance(bootstraps, bool) or
            not isinstance(bootstraps, int) or bootstraps < 1):
        raise ValueError("Minimum editions and bootstrap draws must be positive integers")
    challenger = "residual_coherent"
    references = ("native_c0_unchanged", "residual_independent_matched_marginals")
    mandatory = (challenger, *references)
    primary = ("advance_brier", "placement_rps")
    rows = [row for row in scores if row.get("year") == 2026]
    support_reasons, gate_reasons = [], []
    by_event = defaultdict(dict)
    for row in rows:
        event, model = row["event_id"], row["model"]
        if model in by_event[event]:
            raise ValueError(f"Duplicate event/model score row: {event}/{model}")
        by_event[event][model] = row
        if row.get("eligible") is True:
            for gate in ("structural_checks_passed", "chronology_checks_passed"):
                if row.get(gate) is not True:
                    gate_reasons.append(f"{event}/{model}: {gate} did not pass")
            for metric, value in row.get("metrics", {}).items():
                if value is not None and not _finite_number(value):
                    gate_reasons.append(f"{event}/{model}/{metric}: available score is nonfinite")
    common = {}
    for event, models in by_event.items():
        if not any(model in models for model in mandatory):
            continue
        absent = [model for model in mandatory if model not in models or models[model].get("eligible") is not True]
        if absent:
            support_reasons.append(f"{event}: missing eligible mandatory arms {', '.join(absent)}")
            continue
        editions = {models[model].get("edition_id") for model in mandatory}
        if len(editions) != 1 or None in editions:
            raise ValueError(f"{event}: inconsistent edition identities across arms")
        common[event] = models
    metric_names = sorted(set(primary).union(
        metric for models in common.values() for model in mandatory
        for metric in models[model].get("metrics", {})))
    valid_primary = {metric: set() for metric in primary}
    comparisons = []
    rng = np.random.default_rng(seed)
    for reference in references:
        for metric in metric_names:
            edition_deltas = defaultdict(list)
            supported_editions, nonfinite_events = set(), []
            for event, models in common.items():
                relevant = [models[model] for model in mandatory] if metric in primary else [
                    models[challenger], models[reference]]
                values = [row.get("metrics", {}).get(metric) for row in relevant]
                counts = [row.get("target_counts", {}).get(metric) for row in relevant]
                # A truly absent target is excluded only when every mandatory
                # arm explicitly names its absence and declares zero targets.
                absent_target = (all(value is None for value in values) and
                                 all(count == 0 for count in counts) and
                                 all(row.get("unavailable_reasons", {}).get(metric) for row in relevant))
                if absent_target:
                    continue
                supported = (all(value is not None for value in values) and
                             all(_positive_count(count) for count in counts) and
                             len(set(counts)) == 1)
                if not supported:
                    if metric in primary:
                        support_reasons.append(f"{event}/{metric}: incomplete common target support")
                    else:
                        gate_reasons.append(f"{event}/{metric}: incomplete declared secondary target support")
                    continue
                edition = models[challenger]["edition_id"]
                supported_editions.add(edition)
                if not all(_finite_number(value) for value in values):
                    nonfinite_events.append(event)
                    continue
                delta = models[challenger]["metrics"][metric] - models[reference]["metrics"][metric]
                edition_deltas[edition].append((float(delta), int(counts[0])))
            editions = sorted(edition_deltas)
            if metric in primary and reference == references[0]:
                valid_primary[metric] = supported_editions
            if nonfinite_events:
                comparisons.append({"model": challenger, "reference": reference, "metric": metric,
                                    "mean_delta": None, "ci_lower": None, "ci_upper": None,
                                    "editions": len(supported_editions),
                                    "edition_ids": sorted(supported_editions),
                                    "unavailable_reason": "Nonfinite available target loss; no finite-cohort substitution",
                                    "nonfinite_events": sorted(nonfinite_events),
                                    "target_weighted_sensitivity": None})
                continue
            if not editions:
                continue
            weights = np.asarray([sum(count for _, count in edition_deltas[edition])
                                  for edition in editions], dtype=float)
            deltas = np.asarray([sum(delta * count for delta, count in edition_deltas[edition]) / weight
                                 for edition, weight in zip(editions, weights)], dtype=float)
            samples = rng.integers(0, len(editions), size=(bootstraps, len(editions)))
            draws = deltas[samples].mean(axis=1)
            weighted_draws = (deltas[samples] * weights[samples]).sum(axis=1) / weights[samples].sum(axis=1)
            low, high = np.quantile(draws, [.025, .975])
            weighted_low, weighted_high = np.quantile(weighted_draws, [.025, .975])
            comparisons.append({"model": challenger, "reference": reference, "metric": metric,
                                "mean_delta": float(deltas.mean()), "ci_lower": float(low),
                                "ci_upper": float(high), "editions": len(editions),
                                "edition_ids": editions,
                                "target_weighted_sensitivity": {
                                    "mean_delta": float(np.average(deltas, weights=weights)),
                                    "ci_lower": float(weighted_low), "ci_upper": float(weighted_high)}})
    eligible_editions = len(set.intersection(*(valid_primary[metric] for metric in primary)))
    required_minimum = max(8, minimum_editions)
    if eligible_editions < required_minimum:
        support_reasons.append(f"Only {eligible_editions} independent supported 2026 editions; "
                               f"{required_minimum} required")
    for metric in primary:
        contrast = next((row for row in comparisons if row["reference"] == references[0]
                         and row["metric"] == metric), None)
        matched = next((row for row in comparisons if row["reference"] == references[1]
                        and row["metric"] == metric), None)
        if contrast is None or matched is None:
            support_reasons.append(f"{metric}: mandatory paired contrast unavailable")
        elif contrast["ci_upper"] is None or contrast["ci_upper"] >= 0:
            gate_reasons.append(f"{metric}: paired C0 delta CI upper is not finite and below zero")
    ll = series_guard.get("log_loss_delta_ci_upper")
    brier = series_guard.get("brier_delta")
    if not _finite_number(ll) or ll > .002:
        gate_reasons.append("Canonical series log-loss delta CI upper must be finite and <=0.002")
    if not _finite_number(brier) or brier > 0:
        gate_reasons.append("Canonical series Brier delta must be finite and <=0")
    for key in ("coverage_complete", "structural_checks_passed", "chronology_checks_passed"):
        if series_guard.get(key) is not True:
            gate_reasons.append(f"Canonical series guard {key} did not pass")
    if support_reasons:
        decision = "NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE"
    elif gate_reasons:
        decision = "NOT_ADOPTED"
    else:
        decision = "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"
    return {"status": "COMPLETED", "decision": decision, "eligible_editions": eligible_editions,
            "comparisons": comparisons, "gate_reasons": sorted(set(support_reasons + gate_reasons)),
            "support_reasons": sorted(set(support_reasons)), "bootstraps": bootstraps, "seed": seed,
            "assessment_year": 2026, "minimum_editions": required_minimum,
            "primary_weighting": "equal_edition", "sensitivity_weighting": "team_target",
            "scope": "historical_development_only"}


def _finite_number(value: Any) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number))
            and math.isfinite(float(value)))


def _positive_count(value: Any) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
            and value > 0)
