"""Tournament residual uncertainty: static-team Laplace and dated player ADF.

The existing static-team posterior uses a joint inverse-Hessian Laplace
approximation; the player filter uses dated diagonal assumed-density updates.
Both model uncertainty in complete-series logits, not arbitrary match jitter.
Neither approximation certifies posterior coverage or source publication timing;
chronological outcome exclusion is not point-in-time source authentication.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import ChainMap
from copy import deepcopy
import json
import math
import hashlib
from itertools import chain
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logit, log_expit, logsumexp


def _positive(value, name, *, zero=False):
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def _teams(values):
    result = []
    for value in values:
        if pd.isna(value) or isinstance(value, (bool, np.bool_)):
            raise ValueError("Team identifiers must be present")
        if isinstance(value, Real) and float(value).is_integer():
            value = int(value)
        identifier = str(value)
        if not identifier.strip():
            raise ValueError("Team identifiers must be nonempty")
        result.append(identifier)
    return result


@dataclass(frozen=True)
class TeamStrengthPosterior:
    """Joint team-only Gaussian Laplace approximation, in SERIES-logit units."""

    teams: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    prior_sd: float
    provenance: dict

    def _moments(self, teams):
        names = _teams(teams)
        if len(set(names)) != len(names):
            raise ValueError("Requested teams must be unique")
        index = {team: i for i, team in enumerate(self.teams)}
        mean = np.zeros(len(names))
        covariance = np.eye(len(names)) * self.prior_sd**2
        known = [(i, index[team]) for i, team in enumerate(names) if team in index]
        if known:
            target, source = map(np.asarray, zip(*known))
            mean[target] = self.mean[source]
            covariance[np.ix_(target, target)] = self.covariance[np.ix_(source, source)]
        return names, mean, covariance

    def sample(self, teams, simulations, seed, *, uncertainty_scale=1.0):
        """Draw each team once per tournament. Scale=0 is posterior-mean-only.

        Unseen teams receive the declared prior, not fabricated learned variance.
        The caller passes returned vectors unchanged to simulate_frozen_bracket.
        """
        if (not isinstance(simulations, Integral) or isinstance(simulations, (bool, np.bool_))
                or simulations <= 0):
            raise ValueError("simulations must be a positive integer")
        if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _positive(uncertainty_scale, "uncertainty_scale", zero=True)
        names, mean, covariance = self._moments(teams)
        if uncertainty_scale == 0:
            return {name: np.full(simulations, mean[i]) for i, name in enumerate(names)}
        generator = np.random.default_rng(seed)
        factor = np.linalg.cholesky(covariance)
        draws = mean + uncertainty_scale * (generator.standard_normal((simulations, len(names))) @ factor.T)
        return {name: draws[:, i] for i, name in enumerate(names)}

    def predict(self, first, second, probabilities, *, integrate=True):
        """Marginal complete-series probabilities using 32-node Gaussian quadrature."""
        first, second = _teams(first), _teams(second)
        p = np.asarray(probabilities, dtype=float)
        if (p.ndim != 1 or len(first) != len(second) or len(first) != len(p)
                or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
                or any(a == b for a, b in zip(first, second))):
            raise ValueError("Predictions require aligned distinct pairs and finite probabilities in [0, 1]")
        names = sorted(set(first) | set(second))
        _, mean, covariance = self._moments(names)
        index = {team: i for i, team in enumerate(names)}
        a = np.array([index[t] for t in first], dtype=int)
        b = np.array([index[t] for t in second], dtype=int)
        location = logit(p) + mean[a] - mean[b]
        if not integrate:
            return expit(location)
        variance = np.maximum(0, covariance[a, a] + covariance[b, b] - 2 * covariance[a, b])
        nodes, weights = np.polynomial.hermite.hermgauss(32)
        marginal = expit(location[:, None] + np.sqrt(2 * variance)[:, None] * nodes) @ (weights / np.sqrt(np.pi))
        # Quadrature weights may sum one ulp away from one; preserve certainties.
        return np.where(p == 0, 0.0, np.where(p == 1, 1.0, marginal))


def fit_team_strength_posterior(frame, probability_column, *, cutoff, prior_sd):
    """Fit only rows strictly earlier than cutoff; never tune prior_sd here.

    The probability column MUST be an externally audited sports-only OOF stream.
    Match IDs are unique; same-day/future rows are excluded before fitting. The
    optimizer uses two indexed incidence entries per match, not a dense design.
    """
    _positive(prior_sd, "prior_sd")
    required = ["golgg_match_id", "date", "team1_id", "team2_id", "y_true", probability_column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing evidence columns: {sorted(missing)}")
    dates = pd.to_datetime(frame.date, utc=True, errors="raise")
    boundary = pd.to_datetime(cutoff, utc=True, errors="raise")
    if pd.isna(boundary) or dates.isna().any():
        raise ValueError("Evidence and cutoff dates must be present")
    evidence = frame.loc[dates < boundary, required].copy()
    if evidence.empty:
        raise ValueError("No evidence strictly earlier than cutoff")
    if evidence.golgg_match_id.isna().any() or evidence.golgg_match_id.duplicated().any():
        raise ValueError("Evidence requires unique nonmissing match IDs")
    evidence["date"] = dates.loc[evidence.index].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence["team1_id"], evidence["team2_id"] = _teams(evidence.team1_id), _teams(evidence.team2_id)
    evidence = evidence.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    p = evidence[probability_column].to_numpy(dtype=float)
    y = evidence.y_true.to_numpy(dtype=float)
    if (not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1))
            or not np.isin(y, [0, 1]).all() or (evidence.team1_id == evidence.team2_id).any()):
        raise ValueError("Fit requires interior finite OOF probabilities, binary outcomes and distinct teams")
    teams = tuple(sorted(set(evidence.team1_id) | set(evidence.team2_id)))
    index = {team: i for i, team in enumerate(teams)}
    a = evidence.team1_id.map(index).to_numpy(dtype=int)
    b = evidence.team2_id.map(index).to_numpy(dtype=int)
    offset = logit(p)
    precision = 1 / prior_sd**2

    def objective(theta):
        z = offset + theta[a] - theta[b]
        residual = expit(z) - y
        loss = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * precision * np.dot(theta, theta)
        gradient = (np.bincount(a, weights=residual, minlength=len(teams))
                    - np.bincount(b, weights=residual, minlength=len(teams)) + precision * theta)
        return loss, gradient

    fit = minimize(objective, np.zeros(len(teams)), method="L-BFGS-B", jac=True,
                   options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7})
    if not fit.success:
        raise RuntimeError(f"Residual posterior optimization failed: {fit.message}")
    fitted = expit(offset + fit.x[a] - fit.x[b])
    weights = fitted * (1 - fitted)
    hessian = np.eye(len(teams)) * precision
    np.add.at(hessian, (a, a), weights)
    np.add.at(hessian, (b, b), weights)
    np.add.at(hessian, (a, b), -weights)
    np.add.at(hessian, (b, a), -weights)
    covariance = cho_solve(cho_factor(hessian, lower=True), np.eye(len(teams)))
    covariance = (covariance + covariance.T) * 0.5
    fit.x.setflags(write=False)
    covariance.setflags(write=False)
    provenance = {
        "method": "paired-team logistic-offset Gaussian-prior joint Laplace approximation",
        "probability_unit": "series_win", "probability_column": probability_column,
        "cutoff_exclusive": boundary.isoformat(), "fit_rows": len(evidence),
        "fit_min_date": evidence.date.min(), "fit_max_date": evidence.date.max(),
        "fit_match_ids": evidence.golgg_match_id.tolist(),
        "fit_evidence_sha256": hashlib.sha256(evidence.to_csv(index=False).encode()).hexdigest(),
        "teams": list(teams), "prior_sd": float(prior_sd),
        "optimizer_iterations": int(fit.nit), "optimizer_gradient_max": float(np.max(np.abs(fit.jac))),
        "covariance": "full inverse penalized observed Hessian; opponent-induced dependence retained",
        "unknown_teams": "independent zero-mean prior; no learned evidence",
        "limitations": ["Static team identity; shared-player covariance and roster changes unmodeled",
                        "Laplace approximation, not empirically certified posterior coverage",
                        "OOF source timing audited externally; eligibility_live=0 absent source proof"],
    }
    return TeamStrengthPosterior(teams, fit.x, covariance, float(prior_sd), provenance)


def _residual_day(value, name):
    if value is None or isinstance(value, (Real, bool, np.bool_)):
        raise ValueError(f"{name} must be a dated UTC day")
    try:
        day = pd.Timestamp(value)
        if pd.isna(day):
            raise ValueError
        day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
        return day.normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a dated UTC day") from exc


def _residual_year(value, day):
    if value is None:
        return day.year
    if (not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
            or int(value) != day.year):
        raise ValueError("model_year must equal the UTC decision/event year")
    return int(value)


def _residual_probability(p):
    _positive(p, "probability", zero=True)
    if p > 1:
        raise ValueError("probability must be in [0, 1]")
    return float(p)


def _residual_roster(values):
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("A roster must contain five unique player IDs")
    try:
        players = tuple(_teams(values))
    except (TypeError, ValueError) as exc:
        raise ValueError("A roster must contain five unique player IDs") from exc
    if len(players) != 5 or len(set(players)) != 5:
        raise ValueError("A roster must contain five unique player IDs")
    return players


def _residual_rosters(a, b):
    a, b = _residual_roster(a), _residual_roster(b)
    if set(a).intersection(b):
        raise ValueError("Opponent rosters must be disjoint")
    return a, b


class PlayerResidualFilter:
    """Dated diagonal Gaussian ADF around frozen native SERIES-win logits.

    A match conditions all ten players on one scalar logistic-normal posterior.
    Simultaneously released labels contribute additive natural-parameter changes
    from one shared prior. Only ``advance`` commits updates; ``predict`` and the
    tournament arm builder use read-only as-of snapshots. Player variances diffuse
    lazily, so a daily online prediction never copies the accumulated history.
    """

    _SCHEMA = "dated_player_residual_adf_v1"

    def __init__(self, process_variance_per_day=0.0, prior_sd=.35, quadrature_nodes=32):
        _positive(process_variance_per_day, "process_variance_per_day", zero=True)
        _positive(prior_sd, "prior_sd")
        if (not isinstance(quadrature_nodes, Integral)
                or isinstance(quadrature_nodes, (bool, np.bool_))
                or quadrature_nodes != 32):
            raise ValueError("The frozen residual recipe requires 32 quadrature nodes")
        self.process_variance_per_day = float(process_variance_per_day)
        self.prior_sd = float(prior_sd)
        self._prior_variance = self.prior_sd * self.prior_sd
        if not np.isfinite(self._prior_variance) or self._prior_variance <= 0:
            raise ValueError("prior_sd must have finite positive squared variance")
        self.quadrature_nodes = int(quadrature_nodes)
        self._nodes, self._weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
        self._weights /= np.sqrt(np.pi)
        self._players = {}
        self._pending = {}
        self._seen = set()
        self._history = []
        self._cutoff = None
        self._model_year = None

    def _boundary(self, cutoff_day, model_year):
        cutoff = _residual_day(cutoff_day, "cutoff_day")
        year = _residual_year(model_year, cutoff)
        if self._cutoff is not None and cutoff < self._cutoff:
            raise ValueError("Cannot rewind a persisted residual filter")
        if self._model_year is not None and year < self._model_year:
            raise ValueError("Cannot rewind an annual base-model anchor")
        return cutoff, year

    def _moment(self, players, player, day):
        entry = players.get(player)
        if entry is None:
            return 0., self._prior_variance
        mean, variance, origin = entry
        elapsed = (day - origin).days
        if elapsed < 0:
            raise ValueError("Cannot rewind a player state")
        variance += self.process_variance_per_day * elapsed
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Diffused player variance must be finite and positive")
        return mean, variance

    def _batch(self, players, labels, release):
        involved = sorted({p for row in labels for p in row["roster_a"] + row["roster_b"]})
        prior = {p: self._moment(players, p, release) for p in involved}
        increments = {p: ([], []) for p in involved}
        coefficients = np.array([.2] * 5 + [-.2] * 5)
        for row in sorted(labels, key=lambda r: r["match_id"]):
            p, y = row["p"], row["y"]
            if p in (0., 1.):
                # Compatible absorbing probabilities have constant likelihood.
                continue
            roster = row["roster_a"] + row["roster_b"]
            means = np.array([prior[player][0] for player in roster])
            variances = np.array([prior[player][1] for player in roster])
            location = float(logit(p) + coefficients @ means)
            variance = float(coefficients ** 2 @ variances)
            z = location + np.sqrt(2 * variance) * self._nodes
            log_weights = np.log(self._weights) + log_expit(z if y else -z)
            weights = np.exp(log_weights - logsumexp(log_weights))
            posterior_mean = float(weights @ z)
            posterior_variance = float(weights @ ((z - posterior_mean) ** 2))
            gain = coefficients * variances / variance
            updated_means = means + gain * (posterior_mean - location)
            # Conditional-normal residual plus scalar-posterior uncertainty.
            updated_variances = (variances - (coefficients * variances) ** 2 / variance
                                 + gain ** 2 * posterior_variance)
            if (not np.isfinite(updated_means).all()
                    or not np.isfinite(updated_variances).all()
                    or np.any(updated_variances <= 0)):
                raise ValueError("ADF posterior moments must be finite with positive variance")
            for player, mean, var in zip(roster, updated_means, updated_variances):
                old_mean, old_var = prior[player]
                precision, information = increments[player]
                precision.append(1 / var - 1 / old_var)
                information.append(mean / var - old_mean / old_var)
        for player in involved:
            mean, var = prior[player]
            precision_changes, information_changes = increments[player]
            precision = 1 / var + math.fsum(precision_changes)
            information = mean / var + math.fsum(information_changes)
            if not np.isfinite(precision) or precision <= 0 or not np.isfinite(information):
                raise ValueError("Batched posterior precision must be finite and positive")
            players[player] = (information / precision, 1 / precision, release)

    def _view(self, cutoff, year):
        reset = self._model_year != year
        cutoff_key = cutoff.date().isoformat()
        eligible = [row for row in self._pending.values()
                    if row["model_year"] == year
                    and row["result_day"] < cutoff_key]
        if not reset and not eligible:
            return self._players, eligible
        # Overlay only changed player entries; no copying labels or all history.
        players = {} if reset else ChainMap({}, self._players)
        batches = {}
        for row in eligible:
            release = _residual_day(row["result_day"], "result_day")
            batches.setdefault(release, []).append(row)
        for release, rows in sorted(batches.items()):
            self._batch(players, rows, release)
        return players, eligible

    def observe(self, p, y, roster_a, roster_b, event_day, result_day,
                match_id=None, model_year=None):
        """Queue one label without exposing it to same-release-day predictions."""
        p = _residual_probability(p)
        if (not isinstance(y, Real) or isinstance(y, (bool, np.bool_))
                or not np.isfinite(y) or y not in (0, 1)):
            raise ValueError("An observed outcome must be 0 or 1")
        if p in (0., 1.) and p != y:
            raise ValueError("An absorbing base probability assigns zero likelihood to this label")
        a, b = _residual_rosters(roster_a, roster_b)
        event, release = _residual_day(event_day, "event_day"), _residual_day(result_day, "result_day")
        year = _residual_year(model_year, event)
        if release < event:
            raise ValueError("result_day cannot precede event_day")
        if self._cutoff is not None and release < self._cutoff:
            raise ValueError("Cannot insert an already-released label behind the persisted cutoff")
        if self._model_year is not None and year < self._model_year:
            raise ValueError("Cannot queue a label from an obsolete annual base anchor")
        row = {"p": p, "y": int(y), "roster_a": list(a), "roster_b": list(b),
               "event_day": event.date().isoformat(), "result_day": release.date().isoformat(),
               "model_year": year}
        if match_id is None:
            identifier = "anonymous:" + hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        else:
            identifier = _teams([match_id])[0]
        if identifier in self._seen:
            raise ValueError(f"Duplicate residual match_id: {identifier}")
        row["match_id"] = identifier
        self._pending[identifier] = row
        self._seen.add(identifier)

    def advance(self, cutoff_day, model_year=None):
        """Commit prior-released labels, resetting at the annual base boundary."""
        cutoff, year = self._boundary(cutoff_day, model_year)
        players, eligible = self._view(cutoff, year)
        if isinstance(players, ChainMap):
            self._players.update(players.maps[0])
        else:
            self._players = players
        for row in sorted(eligible, key=lambda r: (r["result_day"], r["match_id"])):
            self._pending.pop(row["match_id"])
            self._history.append({**row, "disposition": "consumed",
                                  "decision_day": cutoff.date().isoformat()})
        for identifier, row in list(self._pending.items()):
            if row["model_year"] < year:
                self._pending.pop(identifier)
                self._history.append({**row, "disposition": "discarded_old_anchor",
                                      "decision_day": cutoff.date().isoformat()})
        self._cutoff, self._model_year = cutoff, year
        return self

    def _prediction(self, p, a, b, players, cutoff):
        ma = [self._moment(players, player, cutoff) for player in a]
        mb = [self._moment(players, player, cutoff) for player in b]
        mean_a, mean_b = float(np.mean([x[0] for x in ma])), float(np.mean([x[0] for x in mb]))
        variance_a, variance_b = sum(x[1] for x in ma) / 25, sum(x[1] for x in mb) / 25
        if p in (0., 1.):
            mean_probability = integrated = p
        else:
            location = float(logit(p) + mean_a - mean_b)
            mean_probability = float(expit(location))
            conditional = expit(location + np.sqrt(2 * (variance_a + variance_b)) * self._nodes)
            # Normalize quadrature mass, not probabilities: no output floor/cap.
            integrated = math.fsum(conditional * self._weights) / math.fsum(self._weights)
        return {"mean_probability": mean_probability, "integrated_probability": integrated,
                "mean_a": mean_a, "mean_b": mean_b,
                "variance_a": variance_a, "variance_b": variance_b}

    def predict(self, p, roster_a, roster_b, cutoff_day, model_year=None):
        p = _residual_probability(p)
        a, b = _residual_rosters(roster_a, roster_b)
        cutoff, year = self._boundary(cutoff_day, model_year)
        players, _ = self._view(cutoff, year)
        return self._prediction(p, a, b, players, cutoff)

    def to_dict(self):
        """Return detached numeric state plus pending/consumed/discarded chronology."""
        return {"schema": self._SCHEMA, "process_variance_per_day": self.process_variance_per_day,
                "prior_sd": self.prior_sd, "quadrature_nodes": self.quadrature_nodes,
                "cutoff_day": None if self._cutoff is None else self._cutoff.date().isoformat(),
                "model_year": self._model_year,
                "players": {p: {"mean": m, "variance": v, "day": day.date().isoformat()}
                            for p, (m, v, day) in sorted(self._players.items())},
                "pending_labels": deepcopy(sorted(self._pending.values(), key=lambda r: r["match_id"])),
                "history": deepcopy(self._history), "seen_match_ids": sorted(self._seen)}

    @classmethod
    def from_dict(cls, payload):
        """Restore validated numeric state; never trust unchecked saved chronology."""
        expected = {"schema", "process_variance_per_day", "prior_sd", "quadrature_nodes",
                    "cutoff_day", "model_year", "players", "pending_labels", "history", "seen_match_ids"}
        if not isinstance(payload, dict) or set(payload) != expected or payload["schema"] != cls._SCHEMA:
            raise ValueError("Malformed saved player residual filter")
        state = cls(payload["process_variance_per_day"], payload["prior_sd"], payload["quadrature_nodes"])
        if payload["cutoff_day"] is None:
            if payload["model_year"] is not None or payload["players"] or payload["history"]:
                raise ValueError("Unadvanced saved filter cannot have fitted states or history")
        else:
            state._cutoff = _residual_day(payload["cutoff_day"], "cutoff_day")
            if payload["model_year"] is None:
                raise ValueError("An advanced filter requires an annual model anchor")
            state._model_year = _residual_year(payload["model_year"], state._cutoff)
        if (not isinstance(payload["players"], dict)
                or not isinstance(payload["pending_labels"], list)
                or not isinstance(payload["history"], list)
                or not isinstance(payload["seen_match_ids"], list)):
            raise ValueError("Malformed saved filter collections")
        for identifier, entry in payload["players"].items():
            if _teams([identifier])[0] != identifier or not isinstance(entry, dict) or set(entry) != {"mean", "variance", "day"}:
                raise ValueError("Malformed saved player state")
            mean, variance = entry["mean"], entry["variance"]
            if not isinstance(mean, Real) or isinstance(mean, bool) or not np.isfinite(mean):
                raise ValueError("Saved player mean must be finite")
            _positive(variance, "saved player variance")
            day = _residual_day(entry["day"], "player day")
            if state._cutoff is None or day >= state._cutoff or day.year != state._model_year:
                raise ValueError("Saved player day lies outside the persisted annual chronology")
            state._players[identifier] = (float(mean), float(variance), day)
        label_fields = {"p", "y", "roster_a", "roster_b", "event_day", "result_day", "model_year", "match_id"}
        validator = cls(state.process_variance_per_day, state.prior_sd)
        labelled = chain(((False, row) for row in payload["pending_labels"]),
                         ((True, row) for row in payload["history"]))
        for historical, row in labelled:
            fields = label_fields | ({"disposition", "decision_day"} if historical else set())
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError("Malformed saved residual label")
            # Reuse the ingestion boundary checks for p/y/rosters/release/IDs.
            validator.observe(**{key: row[key] for key in label_fields})
            identifier = next(iter(validator._pending))
            if identifier != row["match_id"] or identifier in state._seen:
                raise ValueError("Saved residual label IDs must be unique canonical strings")
            canonical = validator._pending.pop(identifier)
            state._seen.add(identifier)
            if historical:
                decision = _residual_day(row["decision_day"], "decision_day")
                release = _residual_day(row["result_day"], "result_day")
                if state._cutoff is None or decision > state._cutoff:
                    raise ValueError("Saved history is ahead of its persisted cutoff")
                if row["disposition"] == "consumed":
                    if release >= decision or decision.year != row["model_year"]:
                        raise ValueError("Saved history consumed an ineligible label")
                elif row["disposition"] == "discarded_old_anchor":
                    if row["model_year"] >= decision.year:
                        raise ValueError("Saved history discarded a nonobsolete annual label")
                else:
                    raise ValueError("Unknown saved label disposition")
                state._history.append({**canonical, "disposition": row["disposition"],
                                       "decision_day": decision.date().isoformat()})
            else:
                release = _residual_day(row["result_day"], "result_day")
                if (state._cutoff is not None and
                        (release < state._cutoff or row["model_year"] < state._model_year)):
                    raise ValueError("Saved pending label is behind the persisted chronology")
                state._pending[identifier] = canonical
        seen = payload["seen_match_ids"]
        if any(not isinstance(identifier, str) for identifier in seen) or len(set(seen)) != len(seen) or set(seen) != state._seen:
            raise ValueError("Saved deduplication IDs do not match label chronology")
        return state


def build_residual_tournament_arms(base_table, event, filter_state, simulations=10000, seed=20261006):
    """Freeze four arms; matched marginals integrate the identical coherent draws.

    Each player's draw is reused for the entire bracket, with no organization
    identity or in-bracket label updates. Pair marginals match *finite* latent
    samples, not an asserted exact continuous-posterior or tournament joint law.
    """
    from src.models.tournament_formats import required_best_ofs

    if not isinstance(filter_state, PlayerResidualFilter):
        raise ValueError("filter_state must be a PlayerResidualFilter")
    if (not isinstance(simulations, Integral) or isinstance(simulations, (bool, np.bool_))
            or simulations <= 0):
        raise ValueError("simulations must be a positive integer")
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    teams = _teams(event["teams"])
    if len(teams) < 2 or len(set(teams)) != len(teams):
        raise ValueError("Tournament entrants must be distinct")
    rosters = {team: _residual_roster(event["rosters"][team]) for team in teams}
    all_players = [p for roster in rosters.values() for p in roster]
    if len(set(all_players)) != len(all_players):
        raise ValueError("Frozen entrant rosters must be disjoint")
    cutoff, year = filter_state._boundary(event["cutoff"], event.get("year"))
    players, _ = filter_state._view(cutoff, year)
    if not isinstance(base_table, dict) or base_table.get("probability_unit") != "series_win":
        raise ValueError("Residual arms require a standard series-win pair table")
    rows = base_table.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Residual arms require complete pair rows")
    team_order = {team: i for i, team in enumerate(teams)}
    seen_pairs, formats = set(), set(required_best_ofs(event["spec"]))
    normalized_pairs = []
    for row in rows:
        if not isinstance(row, dict) or "p" not in row:
            raise ValueError("Malformed residual base pair row")
        native_fields = ("A", "B", "BO")
        descriptive_fields = ("team_a", "team_b", "best_of")
        native = all(field in row for field in native_fields)
        descriptive = all(field in row for field in descriptive_fields)
        if not native and not descriptive:
            raise ValueError("Pair rows require A/B/BO or team_a/team_b/best_of")
        fields = native_fields if native else descriptive_fields
        a, b = _teams([row[fields[0]], row[fields[1]]])
        bo = row[fields[2]]
        if native and descriptive:
            other_a, other_b = _teams([row["team_a"], row["team_b"]])
            if (a, b, bo) != (other_a, other_b, row["best_of"]):
                raise ValueError("Conflicting native/descriptive pair fields")
        if (a not in team_order or b not in team_order or a == b
                or not isinstance(bo, Integral) or isinstance(bo, (bool, np.bool_)) or bo not in (1, 3, 5)):
            raise ValueError("Base pair row has unknown entrants or illegal format")
        _residual_probability(row["p"])
        key = (min(team_order[a], team_order[b]), max(team_order[a], team_order[b]), bo)
        if key in seen_pairs:
            raise ValueError("Duplicate or reversed duplicate residual base pair row")
        seen_pairs.add(key)
        formats.add(bo)
        normalized_pairs.append((a, b))
    expected = {(i, j, bo) for i in range(len(teams)) for j in range(i + 1, len(teams)) for bo in formats}
    if seen_pairs != expected:
        raise ValueError("Native base table must cover every entrant pair and required series format")
    rng = np.random.default_rng(seed)
    draws = {team: np.zeros(simulations) for team in teams}
    ownership = {player: team for team, roster in rosters.items() for player in roster}
    for player in sorted(all_players):
        mean, variance = filter_state._moment(players, player, cutoff)
        draws[ownership[player]] += (mean + np.sqrt(variance) * rng.standard_normal(simulations)) / 5
    mean_table, matched_table = deepcopy(base_table), deepcopy(base_table)
    max_quadrature_gap = 0.
    for original, (a, b), mean_row, matched_row in zip(
            rows, normalized_pairs, mean_table["rows"], matched_table["rows"]):
        p = float(original["p"])
        prediction = filter_state._prediction(p, rosters[a], rosters[b], players, cutoff)
        mean_row["p"] = prediction["mean_probability"]
        matched_row["p"] = (p if p in (0., 1.) else
                            float(expit(logit(p) + draws[a] - draws[b]).mean()))
        max_quadrature_gap = max(max_quadrature_gap,
                                 abs(matched_row["p"] - prediction["integrated_probability"]))
    metadata = {
        "method": "dated_player_residual_diagonal_adf", "model_year": year,
        "cutoff_day": cutoff.date().isoformat(), "prior_sd": filter_state.prior_sd,
        "process_variance_per_day": filter_state.process_variance_per_day,
        "quadrature_nodes": filter_state.quadrature_nodes, "simulations": int(simulations),
        "seed": int(seed), "latent_schedule": "one draw per player per whole bracket",
        "independent_marginals": "exact same finite coherent player draws, pairwise integrated",
        "finite_draw_max_abs_gap_from_series_quadrature": max_quadrature_gap,
        "limitations": ["Diagonal ADF is an approximation, not certified posterior coverage",
                       "Finite-sample matched marginals do not imply identical tournament joint distributions",
                       "No source publication/authentication or extrapolation claim"],
    }
    native, coherent_table = deepcopy(base_table), deepcopy(base_table)
    arms = {"native_c0_unchanged": {"table": native, "team_strength_draws": None},
            "residual_posterior_mean": {"table": mean_table, "team_strength_draws": None},
            "residual_independent_matched_marginals": {"table": matched_table, "team_strength_draws": None},
            "residual_coherent": {"table": coherent_table, "team_strength_draws": draws}}
    for name, arm in arms.items():
        if name != "native_c0_unchanged":
            arm["table"].setdefault("provenance", {})["residual_filter"] = {**metadata, "arm": name}
        arm["residual_metadata"] = {**metadata, "arm": name}
    return arms
