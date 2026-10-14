"""Glicko-derived proxy pair tables for historical bracket diagnostics.

Generates complete series-win probability tables across best-of formats (1, 3, 5)
for tournament entrants, strictly as-of pre-tournament cutoff timestamps.
Supported predictors (none loads a native neural checkpoint):
1. calibrated_regional_glicko_proxy: Regional proxy followed by a fixed
   temperature=1.12, dampening=0.08, cap=0.88 probability transform.
2. regional_glicko_proxy: Team Glicko ratings plus fixed regional offsets
   (gamma=0.70, temperature=1.08).
3. glicko_format: Prior team Glicko ratings with BO slopes fitted on prior-year CAL.
4. uniform: Reference null baseline assigning p = 0.50 to all pairs and formats.

Invariants:
- Exact side anti-symmetry: p(B, A, BO) == 1.0 - p(A, B, BO) with numerical error < 1.0e-7.
- Native-style glicko_format probabilities are unclipped; fixed proxies retain their stated clips.
- Zero temporal leakage: only series completed strictly prior to cutoff timestamp are consumed.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from itertools import combinations, groupby
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

from src.models.calibrated_tournament_model import calibrate_pairwise_probability
from src.ratings.glicko2_core import Glicko2Observation, Glicko2State, update


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_RESULTS_PATH = (
    PROJECT_ROOT / "data/artifacts/leaguepedia-tournament-rules-20260908/series_results.csv"
)

# Standard LoL team alias mapping to canonical tournament entrant IDs
TEAM_ALIASES: dict[str, str] = {
    "Dplus": "DK",
    "Dplus KIA": "DK",
    "DWG KIA": "DK",
    "DAMWON Gaming": "DK",
    "MADL LEC": "MAD",
    "MAD KOI": "MAD",
    "MAD Lions": "MAD",
    "MAD Lions KOI": "MAD",
    "GGS": "GG",
    "Golden Guardians": "GG",
    "PAIN": "PNG",
    "PaiN": "PNG",
    "paiN Gaming": "PNG",
    "PSGT": "PSG",
    "PSG Talon": "PSG",
    "TOP Esports": "TES",
    "Top Esports": "TES",
    "Bilibili Gaming": "BLG",
    "G2 Esports": "G2",
    "Fnatic": "FNC",
    "FlyQuest": "FLY",
    "Team Liquid": "TL",
    "Cloud9": "C9",
    "Gen.G": "GEN",
    "JD Gaming": "JDG",
    "KT Rolster": "KT",
    "Weibo Gaming": "WBG",
    "Hanwha Life Esports": "HLE",
    "Team BDS": "BDS",
    "GAM Esports": "GAM",
}

# Regional competition family alignment
TEAM_REGIONS: dict[str, str] = {
    # LCK (Korea)
    "GEN": "LCK", "T1": "LCK", "KT": "LCK", "DK": "LCK", "HLE": "LCK",
    # LPL (China)
    "BLG": "LPL", "JDG": "LPL", "WBG": "LPL", "LNG": "LPL", "TES": "LPL",
    # LEC (Europe)
    "G2": "LEC", "FNC": "LEC", "MAD": "LEC", "BDS": "LEC",
    # LCS (North America)
    "NRG": "LCS", "C9": "LCS", "TL": "LCS", "FLY": "LCS", "GG": "LCS",
    # Minor Tier 1 (VCS, PCS, CBLOL)
    "GAM": "Minor Tier 1", "PSG": "Minor Tier 1", "PNG": "Minor Tier 1",
}

# Calibrated regional strength offsets relative to LEC anchor = 0.0
REGIONAL_OFFSETS: dict[str, float] = {
    "LCK": 0.463,
    "LPL": 0.355,
    "LEC": 0.000,
    "LCS": 0.013,
    "Minor Tier 1": -0.456,
}

FORMAT_SLOPES: dict[int, float] = {1: 1.0, 3: 1.34, 5: 1.62}
GLICKO_SCALE: float = 173.7178
GLICKO_DEFAULT_RATING: float = 1500.0

REGIONAL_GAMMA: float = 0.70
REGIONAL_TEMPERATURE: float = 1.08
PAIRWISE_CAP: float = 0.88
TRANSFORM_TEMPERATURE: float = 1.12
ENTROPY_DAMPENING: float = 0.08
MODEL_KINDS = (
    "uniform", "glicko_format", "regional_glicko_proxy",
    "calibrated_regional_glicko_proxy",
)
PREDICTOR_CODE_PATHS = (
    "src/models/tournament_pair_adapter.py",
    "src/models/calibrated_tournament_model.py",
    "src/ratings/glicko2_core.py",
    "scripts/run_tournament_distribution_benchmark.py",
    "src/models/tournament_catalog.py",
    "src/models/tournament_formats.py",
    "src/models/tournament_scorelines.py",
    "src/analysis/tournament_metrics.py",
    "src/models/tournament_bracket_simulator.py",
)


def validate_model_kind(model_kind: str) -> None:
    """Refuse native names: these predictors have no native artifact bindings."""
    if model_kind not in MODEL_KINDS:
        raise ValueError(
            f"Unsupported model_kind {model_kind!r}; supported proxies: {MODEL_KINDS}. "
            "Native c0/a0 require a separate artifact-bound predictor."
        )


def normalize_team_id(name: str) -> str:
    """Normalize raw tournament team string to canonical entrant identifier."""
    if not name:
        return name
    clean = str(name).strip()
    return TEAM_ALIASES.get(clean, clean)


@dataclass(frozen=True, slots=True)
class HistoricalMatch:
    """Chronologically sorted series outcome."""

    scheduled_at: str
    team1: str
    team2: str
    score1: float  # 1.0 if team1 won, 0.0 if team2 won
    best_of: int


class TournamentPairAdapter:
    """Generates strictly prior-to-cutoff pairwise win probability tables."""

    def __init__(self, series_results_path: Path | str | None = None) -> None:
        self.series_results_path = (
            Path(series_results_path) if series_results_path is not None else DEFAULT_SERIES_RESULTS_PATH
        )
        self._matches: list[HistoricalMatch] = self._load_matches()
        self._format_calibrations: dict[int, dict[str, Any]] = {}

    def _load_matches(self) -> list[HistoricalMatch]:
        """Load and normalize completed historical matches from series_results.csv."""
        if not self.series_results_path.is_file():
            raise FileNotFoundError(f"Series results file not found: {self.series_results_path}")

        matches: list[HistoricalMatch] = []
        source_bytes = self.series_results_path.read_bytes()
        self._source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        with io.StringIO(source_bytes.decode("utf-8")) as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if row.get("status") != "completed":
                    continue
                sched = row.get("scheduled_at_utc") or row.get("source_date")
                if not sched:
                    continue
                w_side = row.get("winner_side")
                if w_side not in ("1", "2"):
                    continue
                t1 = normalize_team_id(row.get("team1", ""))
                t2 = normalize_team_id(row.get("team2", ""))
                if not t1 or not t2 or t1 == t2:
                    continue
                try:
                    bo = int(row.get("best_of", 1))
                except (ValueError, TypeError):
                    bo = 1
                score1 = 1.0 if w_side == "1" else 0.0
                matches.append(
                    HistoricalMatch(
                        scheduled_at=sched,
                        team1=t1,
                        team2=t2,
                        score1=score1,
                        best_of=bo,
                    )
                )

        matches.sort(key=lambda m: self._utc_clock(m.scheduled_at))
        return matches

    def compute_team_ratings(self, cutoff: str) -> dict[str, Glicko2State]:
        """Compute Glicko-2 ratings for all teams using series strictly prior to cutoff."""
        ratings: dict[str, Glicko2State] = {}
        boundary = self._utc_clock(cutoff)
        for match in self._matches:
            if self._utc_clock(match.scheduled_at) >= boundary:
                continue
            s1 = ratings.get(match.team1, Glicko2State())
            s2 = ratings.get(match.team2, Glicko2State())
            obs1 = [Glicko2Observation(opponent_rating=s2.rating, opponent_rd=s2.rd, score=match.score1)]
            obs2 = [Glicko2Observation(opponent_rating=s1.rating, opponent_rd=s1.rd, score=1.0 - match.score1)]
            ratings[match.team1] = update(s1, obs1)
            ratings[match.team2] = update(s2, obs2)
        return ratings

    @staticmethod
    def _utc_clock(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("rating and calibration clocks require explicit UTC offsets")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _rating_delta(sa: Glicko2State, sb: Glicko2State) -> float:
        return ((sa.rating - sb.rating) / GLICKO_SCALE) / math.sqrt(
            2.0 + (sa.rd / GLICKO_SCALE) ** 2 + (sb.rd / GLICKO_SCALE) ** 2
        )

    def fitted_format_slopes(self, cutoff: str, best_ofs: Sequence[int]) -> dict[str, Any]:
        """Fit zero-intercept BO slopes from strictly prior-year CAL labels only."""
        year = self._utc_clock(cutoff).year
        if year not in self._format_calibrations:
            start = datetime(year - 1, 7, 1, tzinfo=timezone.utc).date()
            stop = datetime(year, 1, 1, tzinfo=timezone.utc).date()
            ratings: dict[str, Glicko2State] = {}
            records: dict[int, list[tuple[float, float, str, str, str]]] = {1: [], 3: [], 5: []}
            for day, group in groupby(self._matches, key=lambda m: self._utc_clock(m.scheduled_at).date()):
                if day >= stop:
                    break
                matches = list(group)
                # All series released on one day see the same earlier-only state.
                if day >= start:
                    for match in matches:
                        if match.best_of in records:
                            delta = self._rating_delta(ratings.get(match.team1, Glicko2State()), ratings.get(match.team2, Glicko2State()))
                            records[match.best_of].append((delta, match.score1, match.scheduled_at, match.team1, match.team2))
                for match in matches:
                    sa, sb = ratings.get(match.team1, Glicko2State()), ratings.get(match.team2, Glicko2State())
                    ratings[match.team1] = update(sa, [Glicko2Observation(sb.rating, sb.rd, match.score1)])
                    ratings[match.team2] = update(sb, [Glicko2Observation(sa.rating, sa.rd, 1 - match.score1)])
            fitted = {}
            for bo, values in records.items():
                if not values:
                    continue
                x = np.asarray([row[0] for row in values])
                y = np.asarray([row[1] for row in values])
                if not np.any(x != 0):
                    continue
                def objective(theta):
                    z = theta[0] * x
                    return float(np.mean(np.logaddexp(0, z) - y * z)), np.asarray([np.mean((expit(z) - y) * x)])
                result = minimize(objective, np.asarray([1.0]), jac=True, method="L-BFGS-B", bounds=[(0.0, None)])
                if not result.success or not np.isfinite(result.x).all():
                    raise ValueError(f"prior CAL Glicko BO{bo} slope fitting failed: {result.message}")
                fitted[str(bo)] = {"slope": float(result.x[0]), "rows": len(values),
                                   "max_label_day": max(row[2] for row in values),
                                   "membership_sha256": hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()}
            self._format_calibrations[year] = {"model_year": year, "cal_start": start.isoformat(),
                                               "cal_end_exclusive": stop.isoformat(), "formats": fitted}
        receipt = self._format_calibrations[year]
        missing = [bo for bo in best_ofs if str(bo) not in receipt["formats"]]
        if missing:
            raise ValueError(f"unavailable_prior_calibrated_glicko_formats:{year}:{missing}")
        return json.loads(json.dumps(receipt))

    def predictor_receipt(
        self,
        teams: Sequence[str],
        cutoff: str,
        model_kind: str,
        best_ofs: Sequence[int] = (1, 3, 5),
    ) -> dict[str, Any]:
        """Bind the loaded source, actual fixed/fitted parameters and simulation code."""
        validate_model_kind(model_kind)
        configuration: dict[str, Any] = {"best_ofs": sorted(set(best_ofs))}
        calibration = (
            self.fitted_format_slopes(cutoff, configuration["best_ofs"])
            if model_kind == "glicko_format" else None
        )
        if model_kind == "uniform":
            formula = "p(A,B,BO)=0.5; p(B,A,BO)=1-p(A,B,BO)"
        else:
            configuration.update({
                "format_slopes": (
                    {str(bo): calibration["formats"][str(bo)]["slope"] for bo in configuration["best_ofs"]}
                    if calibration is not None else
                    {str(bo): FORMAT_SLOPES[bo] for bo in configuration["best_ofs"]}
                ),
                "glicko_scale": GLICKO_SCALE,
                "glicko_default_rating": GLICKO_DEFAULT_RATING,
                "initial_state": asdict(Glicko2State()),
                "team_aliases": dict(TEAM_ALIASES),
                "history_policy": "completed series strictly before cutoff; sequential Glicko2 updates",
                "probability_clip": [0.001, 0.999],
            })
            formula = (
                "mu_i=(rating_i-default_rating)/scale; phi_i=rd_i/scale; "
                "d=(mu_A-mu_B)/sqrt(2+phi_A^2+phi_B^2); s=format_slopes[BO]; "
            )
            if model_kind == "glicko_format":
                configuration["probability_clip"] = None
                configuration["calibration"] = calibration
                configuration["calibration_policy"] = {
                    "label_window": "previous year July 1 inclusive to model year January 1 exclusive",
                    "state_policy": "all completed series on a UTC day use ratings from earlier days only",
                    "objective": "mean zero-intercept series Bernoulli log loss per BO",
                    "fit_intercept": False,
                    "slope_bounds": [0.0, None],
                }
                formula += (
                    "p=sigmoid(s*d); "
                    "s=argmin_{s>=0} mean_prior_year_CAL("
                    "logaddexp(0,s*d_prior_day)-y*s*d_prior_day)"
                )
            else:
                configuration.update({
                    "regional_gamma": REGIONAL_GAMMA,
                    "regional_temperature": REGIONAL_TEMPERATURE,
                    "regional_offsets": dict(REGIONAL_OFFSETS),
                    "team_regions": dict(TEAM_REGIONS),
                    "unknown_region": "LEC",
                    "unknown_regional_offset": 0.0,
                })
                formula += (
                    "o=offset(region_A)-offset(region_B); "
                    "u=sigmoid(s*(d+regional_gamma*o)/regional_temperature); "
                )
                if model_kind == "regional_glicko_proxy":
                    formula += "p=clip(u,0.001,0.999)"
                else:
                    configuration.update({
                        "pairwise_cap": PAIRWISE_CAP,
                        "temperature": TRANSFORM_TEMPERATURE,
                        "entropy_dampening": ENTROPY_DAMPENING,
                        "transform_input_clip": [0.0001, 0.9999],
                    })
                    formula += (
                        "v=clip(u,0.0001,0.9999); "
                        "p=clip((1-entropy_dampening)*sigmoid(logit(v)/temperature)"
                        "+entropy_dampening*0.5,1-pairwise_cap,pairwise_cap)"
                    )
            formula += "; p(B,A,BO)=1-p(A,B,BO)"
        canonical_config = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        return {
            "schema_version": 1,
            "model_kind": model_kind,
            "predictor_kind": ("reference_null" if model_kind == "uniform" else
                               "prior_calibrated_glicko_format" if model_kind == "glicko_format" else "glicko_derived_proxy"),
            "teams": list(teams),
            "cutoff": cutoff,
            "source": {"path": str(self.series_results_path), "sha256": self._source_sha256},
            "formula": formula,
            "formula_sha256": hashlib.sha256(formula.encode("utf-8")).hexdigest(),
            "configuration": configuration,
            "configuration_sha256": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
            "code_sha256": {
                path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                for path in PREDICTOR_CODE_PATHS
            },
        }

    def predict_pair_table(
        self,
        teams: Sequence[str],
        cutoff: str,
        model_kind: str,
        best_ofs: Sequence[int] = (1, 3, 5),
    ) -> dict[tuple[str, str, int], float]:
        """Generate the complete pairwise probability table for entrant teams.

        Parameters
        ----------
        teams : Sequence[str]
            List of entrant team IDs.
        cutoff : str
            ISO UTC timestamp strictly bounding the training/rating evidence.
        model_kind : str
            One of MODEL_KINDS; native c0/a0 names are unsupported.
        best_ofs : Sequence[int]
            Best-of formats required (1, 3, 5).

        Returns
        -------
        dict[tuple[str, str, int], float]
            Mapping of (team_a, team_b, best_of) -> series win probability.
        """
        validate_model_kind(model_kind)

        team_list = list(teams)
        if len(team_list) < 2 or len(set(team_list)) != len(team_list):
            raise ValueError("teams must contain at least two unique identifiers")

        formats = sorted(set(best_ofs))
        for bo in formats:
            if bo not in (1, 3, 5):
                raise ValueError(f"best_of must be 1, 3, or 5, got {bo!r}")

        # Uniform reference model
        if model_kind == "uniform":
            probs: dict[tuple[str, str, int], float] = {}
            for a, b in combinations(team_list, 2):
                for bo in formats:
                    probs[(a, b, bo)] = 0.50
                    probs[(b, a, bo)] = 0.50
            return probs

        # Compute prior Glicko-2 ratings as-of cutoff
        ratings = self.compute_team_ratings(cutoff)
        slopes = ({bo: self.fitted_format_slopes(cutoff, formats)["formats"][str(bo)]["slope"] for bo in formats}
                  if model_kind == "glicko_format" else FORMAT_SLOPES)
        probs = {}

        for a, b in combinations(team_list, 2):
            sa = ratings.get(a, Glicko2State())
            sb = ratings.get(b, Glicko2State())

            mu_a = (sa.rating - GLICKO_DEFAULT_RATING) / GLICKO_SCALE
            mu_b = (sb.rating - GLICKO_DEFAULT_RATING) / GLICKO_SCALE
            phi_a = sa.rd / GLICKO_SCALE
            phi_b = sb.rd / GLICKO_SCALE

            # Base normalized rating differential with beta = 1.0
            denom = math.sqrt(2.0 + phi_a * phi_a + phi_b * phi_b)
            delta_ab = (mu_a - mu_b) / denom

            reg_a = TEAM_REGIONS.get(a, "LEC")
            reg_b = TEAM_REGIONS.get(b, "LEC")
            offset_diff = REGIONAL_OFFSETS.get(reg_a, 0.0) - REGIONAL_OFFSETS.get(reg_b, 0.0)

            for bo in formats:
                slope = slopes[bo]

                if model_kind == "glicko_format":
                    # Direct format-sloped Glicko-2
                    p_ab = float(expit(slope * delta_ab))
                    p_ba = 1.0 - p_ab

                elif model_kind == "regional_glicko_proxy":
                    # Fixed regional adjustment of the same team Glicko states.
                    z_regional = (delta_ab + REGIONAL_GAMMA * offset_diff) / REGIONAL_TEMPERATURE
                    p_ab = float(expit(slope * z_regional))
                    p_ab = float(np.clip(p_ab, 0.001, 0.999))
                    p_ba = 1.0 - p_ab

                elif model_kind == "calibrated_regional_glicko_proxy":
                    # Fixed transform of the regional proxy, not a native model.
                    z_regional = (delta_ab + REGIONAL_GAMMA * offset_diff) / REGIONAL_TEMPERATURE
                    p_base = float(expit(slope * z_regional))
                    p_ab = calibrate_pairwise_probability(
                        p_base,
                        cap=PAIRWISE_CAP,
                        temperature=TRANSFORM_TEMPERATURE,
                        dampening=ENTROPY_DAMPENING,
                    )
                    p_ba = 1.0 - p_ab

                # Anti-symmetry invariant assertion: |p(A,B) + p(B,A) - 1.0| < 1.0e-7
                assert abs(p_ab + p_ba - 1.0) < 1.0e-7, f"Anti-symmetry violated for ({a}, {b}, {bo})"

                probs[(a, b, bo)] = p_ab
                probs[(b, a, bo)] = p_ba

        return probs
