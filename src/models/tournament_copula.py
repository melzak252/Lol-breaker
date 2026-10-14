"""Marginal-preserving shared-player Gaussian tournament winner responses.

Only complete sports responses along a legal, frozen-origin route are scored.
Organizer lottery terms and winner-conditional scoreline terms are not included:
this is not the probability of a fully authenticated tournament path. Static,
disjoint five-player rosters induce iid standard-normal team states, so the
Sobol team basis is equivalent to the player basis. This does not identify
player-specific learned variance, transfer benefits, posterior uncertainty, or
prospective calibration. Replicate SEs diagnose integration, not iid coverage.

Opposite-side responses must negate one authoritative pair probit, not invert a
rounded 1-p. Scalar callers cannot recover a rare tail already rounded away;
the tournament engine preserves the original probability orientation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import qmc


_SCHEMA = "gaussian_tournament_copula_v1"
_SAMPLING = {
    "algorithm": "numpy.PCG64.standard_normal",
    "identity_binding": "SHA256 canonical JSON [tournament-copula-player-v1, root_seed, player_id]",
    "root_seed": "explicit sample(seed) argument; no global RNG state",
    "player_law": "independent standard normal, reused throughout each complete rollout",
    "team_law": "sum of five player states / sqrt(5); disjoint entrants",
}
_CAVEATS = [
    "Static disjoint rosters have iid Gaussian team states; no learned player variance or transfer benefit.",
    "Sports-response likelihood omits organizer lottery and conditional scoreline terms.",
    "Replicate standard error is an integration diagnostic, not an iid confidence guarantee.",
    "Historical development-exposed evidence is not prospective calibration or promotion evidence.",
]


def _real(value, name, *, infinite=False):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        checked = np.longdouble(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if np.isnan(checked) or (not infinite and not np.isfinite(checked)):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _rho(value):
    _real(value, "rho")
    if not 0 <= value <= .5:
        raise ValueError("rho must be in [0, 0.5]")
    return float(value)


def _integer(value, name, *, minimum=0, maximum=None):
    if (not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be an integer in the declared bounds")
    return int(value)


def _probability(value):
    _real(value, "probability")
    if not 0 <= value <= 1:
        raise ValueError("probability must be in [0, 1]")
    return float(value)


def _probit_copula_probability(base_z, contrast, rho):
    """Conditional response from a cached base probit; infinities are certainties.

    A finite longdouble contrast is not prematurely narrowed to float64. Extreme
    valid contrasts saturate the CDF rather than overflowing during conversion.
    """
    rho = _rho(rho)
    _real(base_z, "base_z", infinite=True)
    _real(contrast, "contrast")
    if np.isinf(base_z):
        return float(base_z > 0)
    if rho == 0:
        return float(ndtr(float(base_z)))
    z = ((np.longdouble(base_z) + np.longdouble(math.sqrt(rho)) * contrast)
         / np.longdouble(math.sqrt(1 - rho)))
    # Outside this range double-precision Phi has already saturated. Avoid
    # narrowing finite extended-precision values beyond float64's range.
    if z >= 40:
        return 1.
    if z <= -40:
        return 0.
    return float(ndtr(float(z)))


def copula_win_probability(p, contrast, rho):
    """Phi((Phi^-1(p) + sqrt(rho)*contrast)/sqrt(1-rho)).

    Contrast is (U_A-U_B)/sqrt(2), not the raw team difference. Integrating its
    standard-normal law preserves exactly p. Certainties and rho=0 are exact.
    """
    p = _probability(p)
    rho = _rho(rho)
    _real(contrast, "contrast")
    if rho == 0 or p in (0., 1.):
        return p
    return _probit_copula_probability(float(ndtri(p)), contrast, rho)


def _identity(value, name):
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    raise ValueError(f"{name} must be a nonempty string or integer identifier")


def _rosters(rosters):
    if not isinstance(rosters, Mapping) or not rosters:
        raise ValueError("rosters must be a nonempty entrant mapping")
    result, seen = {}, set()
    for team, players in rosters.items():
        if not isinstance(team, str) or not team.strip():
            raise ValueError("team identifiers must be nonempty strings")
        if isinstance(players, (str, bytes, Mapping)):
            raise ValueError("each roster must contain five distinct player IDs")
        try:
            players = tuple(sorted(_identity(p, "player") for p in players))
        except TypeError as exc:
            raise ValueError("each roster must contain five distinct player IDs") from exc
        if len(players) != 5 or len(set(players)) != 5:
            raise ValueError("each roster must contain five distinct player IDs")
        if seen.intersection(players):
            raise ValueError("entrant rosters must be disjoint")
        seen.update(players)
        result[team] = players
    return result


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("provenance must be strict finite JSON") from exc


def _seed(domain, seed, identity):
    payload = json.dumps([domain, seed, identity], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


class GaussianTournamentCopula:
    """One bounded dependence parameter; no fitted per-player state or history."""

    def __init__(self, rho, provenance=None):
        self._rho = _rho(rho)
        if provenance is None:
            provenance = {}
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a JSON object")
        self._provenance = _json_copy(provenance)

    @property
    def rho(self):
        return self._rho

    @property
    def provenance(self):
        return _json_copy(self._provenance)

    def sample(self, rosters, simulations, seed):
        """Return standardized iid team vectors with identity-bound player streams.

        Team labels, roster ordering, and other entrants never enter a player's
        seed. The same player is reproducible across separate lineup scenarios;
        overlapping entrants within one scenario are explicitly unsupported.
        """
        rosters = _rosters(rosters)
        simulations = _integer(simulations, "simulations", minimum=1)
        seed = _integer(seed, "seed")
        result = {}
        for team, players in rosters.items():
            vector = np.zeros(simulations)
            for player in players:
                generator = np.random.Generator(np.random.PCG64(
                    _seed("tournament-copula-player-v1", seed, player)))
                vector += generator.standard_normal(simulations)
            vector /= math.sqrt(5)
            result[team] = vector
        return result

    def to_dict(self):
        return {"schema": _SCHEMA, "rho": self.rho, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) != {"schema", "rho", "provenance"}
                or value["schema"] != _SCHEMA):
            raise ValueError("invalid Gaussian tournament copula serialization")
        return cls(value["rho"], value["provenance"])


def _clock(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO UTC date or timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            if len(value) != 10:
                raise ValueError("timestamp requires a timezone")
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value}") from exc


@dataclass
class _PreparedPath:
    event_id: str
    cutoff: str
    teams: tuple
    series_count: int
    base_z: np.ndarray
    contrasts: tuple
    work: np.ndarray
    log_paths: np.ndarray
    independent_log_probability: float
    impossible: bool
    evidence_sha256: str
    replicate_seeds: tuple
    target_clock_audit: dict

    def evaluate(self, rho):
        if self.impossible:
            return np.full(len(self.contrasts), -np.inf)
        if rho == 0 or self.base_z.size == 0:
            return np.full(len(self.contrasts), self.independent_log_probability)
        logs = np.empty(len(self.contrasts))
        scale = math.sqrt(rho)
        denominator = math.sqrt(1 - rho)
        for i, contrasts in enumerate(self.contrasts):
            np.multiply(contrasts, scale, out=self.work)
            np.add(self.work, self.base_z, out=self.work)
            np.divide(self.work, denominator, out=self.work)
            log_ndtr(self.work, out=self.work)
            np.sum(self.work, axis=1, out=self.log_paths)
            logs[i] = logsumexp(self.log_paths) - math.log(len(self.log_paths))
        return logs


def _prepare(event, power, replicates, seed, training_cutoff=None, *, latent=True):
    # Importing route validation does not create a second tournament engine.
    from src.models.tournament_research_cohort import build_observed_labels

    if not isinstance(event, Mapping):
        raise ValueError("event must be a fixed-origin mapping")
    for field in ("event_id", "cutoff", "rosters", "probabilities", "matches", "spec"):
        if field not in event:
            raise ValueError(f"event requires {field}")
    event_id = _identity(event["event_id"], "event_id")
    origin = _clock(event["cutoff"], "cutoff")
    if _clock(event.get("probability_cutoff", event["cutoff"]), "probability_cutoff") != origin:
        raise ValueError("probabilities must be from the immutable event origin, not daily predictions")
    if training_cutoff is not None and event.get("partition", "TRAIN") != "TRAIN":
        raise ValueError("only TRAIN event labels can enter a fit")
    rosters = _rosters(event["rosters"])
    teams = tuple(sorted(rosters))
    spec = event["spec"]
    if not isinstance(spec, Mapping) or set(spec.get("teams", [])) != set(teams):
        raise ValueError("frozen spec and roster entrant membership must match")
    if not isinstance(event["probabilities"], Mapping):
        raise ValueError("probabilities must be an immutable-origin tuple-key mapping")
    probabilities = {}
    for key, value in event["probabilities"].items():
        if (not isinstance(key, tuple) or len(key) != 3 or key[0] == key[1]
                or key[0] not in rosters or key[1] not in rosters
                or type(key[2]) is not int or key[2] not in (1, 3, 5)):
            raise ValueError("probability keys must be distinct entrant (a,b,best_of) tuples")
        probabilities[key] = _probability(value)
    probabilities = MappingProxyType(probabilities)
    for (a, b, bo), p in probabilities.items():
        if (b, a, bo) in probabilities and not math.isclose(
                p, 1 - probabilities[b, a, bo], rel_tol=0, abs_tol=1e-12):
            raise ValueError("side-reversed origin probabilities must be complementary")
    if not isinstance(event["matches"], (list, tuple)) or not event["matches"]:
        raise ValueError("a complete nonempty observed sports path is required")
    rows = []
    clock_audit = {"date_only_targets": 0, "same_origin_calendar_day_scenarios": 0,
                   "authenticated_target_start_claim": False,
                   "policy": "UTC day-only targets may equal a midnight daily-scenario origin; exact timestamps must be later"}
    for record in event["matches"]:
        if not isinstance(record, Mapping):
            raise ValueError("each observed response must be a label mapping")
        row = dict(record)
        if set(row).intersection({"p", "probability", "p_series", "p_C0"}):
            raise ValueError("observed labels cannot supply later per-series probabilities")
        if "date" not in row:
            raise ValueError("each event target requires a date")
        target = _clock(row["date"], "label date")
        date_only = len(row["date"]) == 10
        same_day = date_only and target == origin
        if target < origin or (target == origin and not same_day):
            raise ValueError("event targets must follow the origin; same-day dates require a midnight daily scenario")
        clock_audit["date_only_targets"] += int(date_only)
        clock_audit["same_origin_calendar_day_scenarios"] += int(same_day)
        release = _clock(row.get("result_day", row["date"]), "label release")
        if release < _clock(row["date"], "label date"):
            raise ValueError("label release cannot precede the observed match")
        if training_cutoff is not None and release >= training_cutoff:
            raise ValueError("TRAIN labels must be strictly before training_cutoff")
        rows.append(row)
    try:
        labels = build_observed_labels(event_id, event.get("edition_id", event_id), rows, spec=spec)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed complete observed tournament route") from exc
    indices = {team: i for i, team in enumerate(teams)}
    base_z, pairs = [], []
    independent, impossible = 0., False
    for row in labels["matches"]:
        a, b, bo = row["team_a"], row["team_b"], row["best_of"]
        if (a, b, bo) in probabilities:
            p, orientation = probabilities[a, b, bo], 1
        elif (b, a, bo) in probabilities:
            p, orientation = probabilities[b, a, bo], -1
        else:
            raise ValueError(f"origin probability is missing for {a}/{b}/BO{bo}")
        response = (1 if row["wins_a"] > row["wins_b"] else -1) * orientation
        if p in (0., 1.):
            if (p == 1) != (response == 1):
                impossible = True
            continue
        independent += math.log(p) if response == 1 else math.log1p(-p)
        base_z.append(response * float(ndtri(p)))
        # Use winner-positive team contrasts, independent of table orientation.
        direction = 1 if row["wins_a"] > row["wins_b"] else -1
        pairs.append((indices[a], indices[b], direction))
    seeds = tuple(_seed("tournament-copula-sobol-v1", seed,
                        [event_id, origin.isoformat(), i]) for i in range(replicates))
    contrasts = []
    count = 1 << power if latent and base_z and not impossible else 1
    for replicate_seed in seeds:
        if latent and base_z and not impossible:
            uniforms = qmc.Sobol(d=len(teams), scramble=True, bits=52,
                                 seed=np.random.default_rng(replicate_seed)).random_base2(power)
            if np.any((uniforms <= 0) | (uniforms >= 1)):
                raise ValueError("Sobol Gaussian transform encountered an endpoint; no clipping is permitted")
            ndtri(uniforms, out=uniforms)
            matrix = np.empty((count, len(pairs)))
            for j, (a, b, direction) in enumerate(pairs):
                np.subtract(uniforms[:, a], uniforms[:, b], out=matrix[:, j])
                matrix[:, j] *= direction / math.sqrt(2)
            matrix.setflags(write=False)
        else:
            matrix = np.empty((count, 0))
        contrasts.append(matrix)
    inventory = {"event_id": event_id, "cutoff": origin.isoformat(), "rosters": rosters,
                 "probabilities": [[*key, p] for key, p in sorted(probabilities.items())],
                 "matches": rows, "spec": spec}
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, allow_nan=False,
                                      separators=(",", ":")).encode()).hexdigest()
    return _PreparedPath(event_id, origin.isoformat(), teams, len(rows), np.asarray(base_z),
                         tuple(contrasts), np.empty((count, len(base_z) if latent else 0)), np.empty(count),
                         -math.inf if impossible else independent, impossible, digest, seeds, clock_audit)


def _integration_args(sobol_power, replicates, seed):
    return (_integer(sobol_power, "sobol_power", minimum=1, maximum=24),
            _integer(replicates, "replicates", minimum=2), _integer(seed, "seed"))


def _diagnostic(path, rho, power, replicates, seed, logs=None):
    if logs is None:
        logs = path.evaluate(rho)
    log_probability = float(logsumexp(logs) - math.log(replicates))
    finite = math.isfinite(log_probability)
    if finite:
        # Delta-method SE on log(mean replicate likelihood), with a stable ratio.
        relative = np.exp(logs - log_probability)
        standard_error = float(np.std(relative, ddof=1) / math.sqrt(replicates) / path.series_count)
    else:
        standard_error = None
    sampled = rho > 0 and not path.impossible and path.base_z.size > 0
    return {
        "event_id": path.event_id, "cutoff": path.cutoff, "rho": rho,
        "series_count": path.series_count,
        "log_probability": log_probability if finite else None,
        "log_probability_status": "finite" if finite else "negative_infinity",
        "normalized_nll": -log_probability / path.series_count if finite else None,
        "normalized_nll_standard_error": standard_error,
        "replicate_log_probabilities": [float(x) if np.isfinite(x) else None for x in logs],
        "reason": (None if finite else "structurally_impossible_certain_response" if path.impossible
                   else "numerically_unresolved_response_probability"),
        "integration": {"method": ("scrambled_sobol_gaussian_team_basis_log_ndtr_logsumexp"
                                   if sampled else "analytic_independent_or_certain_responses"),
                        "sobol_power": power if sampled else None,
                        "replicates": replicates if sampled else 0, "seed": seed if sampled else None,
                        "replicate_seeds": list(path.replicate_seeds) if sampled else [],
                        "requested_sobol_power": power, "requested_replicates": replicates,
                        "aggregation": "mean likelihood, not mean log likelihood",
                        "basis": "iid teams equivalent to static disjoint five-player sums"},
        "evidence_sha256": path.evidence_sha256,
        "target_clock_audit": dict(path.target_clock_audit),
        "scope": "sports responses on fully consumed legal route; lottery terms omitted",
        "uncertainty_caveats": list(_CAVEATS),
    }


def tournament_copula_path_likelihood(event, rho, *, sobol_power=15, replicates=8, seed=20261007):
    """Evaluate a fixed model only; higher-power verification never performs fitting.

    Impossible log probabilities are internally -inf and encoded as JSON null
    with negative_infinity status and an explicit reason, never clipped/smoothed.
    """
    rho = _rho(rho)
    power, replicates, seed = _integration_args(sobol_power, replicates, seed)
    path = _prepare(event, power, replicates, seed, latent=rho > 0)
    return _diagnostic(path, rho, power, replicates, seed)


def fit_tournament_copula(events, *, max_rho=.5, sobol_power=15, replicates=8,
                          seed=20261007, training_cutoff=None):
    """Fit one global rho to equal-edition normalized complete-path NLL.

    Origin timestamps constrain forecast inputs. Date-only labels can share a
    midnight daily-scenario origin; this does not authenticate the target start.
    The optional training_cutoff separately constrains label availability
    before CAL; real experiment callers must supply the frozen CAL boundary.
    Prepared Sobol contrasts and arithmetic workspaces are reused across all 65
    grid points and every interior local-minimum refinement. No base model fits.
    """
    max_rho = _rho(max_rho)
    power, replicates, seed = _integration_args(sobol_power, replicates, seed)
    boundary = None if training_cutoff is None else _clock(training_cutoff, "training_cutoff")
    if isinstance(events, (str, bytes, Mapping)):
        raise ValueError("events must be a nonempty sequence of TRAIN paths")
    try:
        events = list(events)
    except TypeError as exc:
        raise ValueError("events must be a nonempty sequence of TRAIN paths") from exc
    if not events:
        raise ValueError("at least one complete TRAIN path is required")
    if any(isinstance(event, Mapping) and event.get("partition", "TRAIN") != "TRAIN"
           for event in events):
        raise ValueError("only TRAIN event labels can enter a fit")
    paths = [_prepare(event, power, replicates, seed, boundary, latent=max_rho > 0) for event in events]
    paths.sort(key=lambda path: path.event_id)
    if len({path.event_id for path in paths}) != len(paths):
        raise ValueError("TRAIN event identities must be unique")
    if any(path.impossible for path in paths):
        raise ValueError("TRAIN contains a structurally impossible certain response; no finite fit exists")
    evaluated = {}

    def objective(rho):
        rho = float(rho)
        if rho not in evaluated:
            values = [-(float(logsumexp(path.evaluate(rho))) - math.log(replicates))
                      / path.series_count for path in paths]
            evaluated[rho] = math.fsum(values) / len(paths)
        return evaluated[rho]

    grid = np.linspace(0., max_rho, 65)
    values = [objective(rho) for rho in grid]
    refinements = []
    if max_rho > 0:
        for i in range(1, 64):
            if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
                optimum = minimize_scalar(objective, bounds=(float(grid[i - 1]), float(grid[i + 1])),
                                          method="bounded", options={"xatol": 1e-6})
                refinements.append({"grid_index": i, "rho": float(optimum.x),
                                    "normalized_nll": float(optimum.fun),
                                    "success": bool(optimum.success), "evaluations": int(optimum.nfev)})
    minimum = min(evaluated.values())
    if not math.isfinite(minimum):
        raise ValueError("TRAIN has no numerically finite integrated likelihood")
    # A fixed numerical tie tolerance prevents optimizer noise inventing a
    # departure from the exact identity option. No CAL or ASSESS is inspected.
    rho = min(x for x, value in evaluated.items() if value <= minimum + 1e-12)
    diagnostics = [_diagnostic(path, rho, power, replicates, seed) for path in paths]
    provenance = {
        "sampling": dict(_SAMPLING), "uncertainty_caveats": list(_CAVEATS),
        "fit": {"objective": "equal-edition mean normalized sports-path NLL",
                "normalized_nll": objective(rho), "selected_rho": rho,
                "training_cutoff": None if boundary is None else boundary.isoformat(),
                "training_cutoff_enforced": boundary is not None,
                "event_ids": [path.event_id for path in paths],
                "max_rho": max_rho, "seed": seed, "sobol_power": power, "replicates": replicates,
                "grid": [{"rho": float(x), "normalized_nll": value} for x, value in zip(grid, values)],
                "refinements": refinements, "xatol": 1e-6,
                "tie_tolerance": 1e-12, "tie_policy": "lowest rho among tied TRAIN objectives",
                "evaluated_objectives": [{"rho": x, "normalized_nll": value}
                                         for x, value in sorted(evaluated.items())],
                "events": diagnostics, "base_fits": 0, "dependence_parameters": 1},
    }
    return GaussianTournamentCopula(rho, provenance)
