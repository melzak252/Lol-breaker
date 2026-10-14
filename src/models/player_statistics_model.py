"""Player Statistics Generative Model & In-Game Simulation.

Simulates in-game player statistics (kills, deaths, assists, creep score, objectives,
and pentakills) conditional on series match context, team rosters, drafted champions,
and game outcome.
Guarantees exact kill-death conservation invariants across teams.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
from typing import Any, Literal, Mapping, Sequence

import joblib
import numpy as np

from src.features.player_statistics_data import (
    DEFAULT_OBJECTIVE_RATES,
    DEFAULT_ROLE_ASSIST_FACTORS,
    DEFAULT_ROLE_CSM,
    DEFAULT_ROLE_DEATH_SHARES,
    DEFAULT_ROLE_KILL_SHARES,
    DEFAULT_WINNER_KILL_SHARE,
    ROLES,
    extract_player_stat_priors,
    get_default_priors,
    normalize_role,
)

WinnerKey = Literal["A", "B"]


@dataclass(frozen=True)
class PlayerGameStats:
    """Individual player statistics for a single simulated map."""

    player_id: str
    role: str
    champion_id: int
    kills: int
    deaths: int
    assists: int
    cs: int
    kda: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GameStatsResult:
    """Complete simulated map-level result including 10 player stat lines and team objectives."""

    duration_seconds: int
    team_a_kills: int
    team_b_kills: int
    team_a_dragons: int
    team_b_dragons: int
    team_a_barons: int
    team_b_barons: int
    player_stats: dict[str, PlayerGameStats]
    has_pentakill: bool = False
    pentakill_player: str | None = None
    @property
    def team_a_deaths(self) -> int:
        """Team A total deaths equals Team B total kills."""
        return self.team_b_kills

    @property
    def team_b_deaths(self) -> int:
        """Team B total deaths equals Team A total kills."""
        return self.team_a_kills

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "team_a_kills": self.team_a_kills,
            "team_b_kills": self.team_b_kills,
            "team_a_deaths": self.team_a_deaths,
            "team_b_deaths": self.team_b_deaths,
            "team_a_dragons": self.team_a_dragons,
            "team_b_dragons": self.team_b_dragons,
            "team_a_barons": self.team_a_barons,
            "team_b_barons": self.team_b_barons,
            "has_pentakill": self.has_pentakill,
            "pentakill_player": self.pentakill_player,
            "player_stats": {k: v.to_dict() for k, v in self.player_stats.items()},
        }


class PlayerStatsDistributionModel:
    """Hierarchical conditional generative model for League of Legends in-game player statistics."""

    def __init__(self, priors: dict[str, Any] | None = None) -> None:
        self.priors: dict[str, Any] = priors or get_default_priors()
        self._role_kill_shares = self.priors.get("role_kill_shares", DEFAULT_ROLE_KILL_SHARES)
        self._role_death_shares = self.priors.get("role_death_shares", DEFAULT_ROLE_DEATH_SHARES)
        self._role_assist_factors = self.priors.get("role_assist_factors", DEFAULT_ROLE_ASSIST_FACTORS)
        self._winner_kill_share = self.priors.get("winner_kill_share", DEFAULT_WINNER_KILL_SHARE)
        self._objective_rates = self.priors.get("objective_rates", DEFAULT_OBJECTIVE_RATES)
        self._role_csm = self.priors.get("role_csm", DEFAULT_ROLE_CSM)
        self._player_profiles = self.priors.get("player_profiles", {})

    def sample_game_stats(
        self,
        team_a_roster: Sequence[str] | Mapping[str, Any],
        team_b_roster: Sequence[str] | Mapping[str, Any],
        winner: WinnerKey,
        drafted_champions: Sequence[int] | None = None,
        rng: np.random.Generator | random.Random | None = None,
    ) -> GameStatsResult:
        """Sample a fully consistent set of in-game player stats and objectives for one map.

        Parameters
        ----------
        team_a_roster : Sequence[str] | Mapping[str, Any]
            Roster for Team A ordered by [TOP, JUNGLE, MID, ADC, SUPPORT] or keyed by role.
        team_b_roster : Sequence[str] | Mapping[str, Any]
            Roster for Team B ordered by [TOP, JUNGLE, MID, ADC, SUPPORT] or keyed by role.
        winner : Literal['A', 'B']
            Winning team on this map.
        drafted_champions : Sequence[int], optional
            10 drafted champion IDs (first 5 for Team A, next 5 for Team B).
        rng : np.random.Generator | random.Random, optional
            RNG for sampling.

        Returns
        -------
        GameStatsResult
            Sampled duration, objectives, and 10 player lines obeying exact conservation laws.
        """
        # Ensure numpy Generator
        if isinstance(rng, np.random.Generator):
            np_rng = rng
        elif isinstance(rng, random.Random):
            np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        else:
            np_rng = np.random.default_rng()

        # Parse rosters into ordered lists of 5 player IDs
        def extract_roster(r: Sequence[str] | Mapping[str, Any]) -> list[str]:
            if isinstance(r, Mapping):
                return [str(r.get(role, r.get(role.lower(), f"player_{role}"))) for role in ROLES]
            return [str(r[i]) if i < len(r) else f"player_{ROLES[i]}" for i in range(len(ROLES))]

        players_a = extract_roster(team_a_roster)
        players_b = extract_roster(team_b_roster)

        # Parse drafted champions
        if drafted_champions and len(drafted_champions) >= 10:
            champs_a = [int(c) for c in drafted_champions[:5]]
            champs_b = [int(c) for c in drafted_champions[5:10]]
        else:
            champs_a = [101 + i for i in range(5)]
            champs_b = [201 + i for i in range(5)]

        # 1. Sample Game Duration: Lognormal(ln(1920), 0.16^2) in [900, 3600]
        # ln(1920) ~ 7.56008
        mu_dur = math.log(1920.0)
        sigma_dur = 0.16
        raw_dur = float(np_rng.lognormal(mean=mu_dur, sigma=sigma_dur))
        duration_sec = int(round(np.clip(raw_dur, 900.0, 3600.0)))
        duration_min = duration_sec / 60.0

        # 2. Sample Total Kills: NegBinomial(mu = 26.5 * (T / 1920), alpha = 0.18)
        # Parameterization: n = 1 / alpha, p = 1 / (1 + alpha * mu)
        mu_kills = 26.5 * (duration_sec / 1920.0)
        disp_alpha = 0.18
        nb_n = 1.0 / disp_alpha
        nb_p = 1.0 / (1.0 + disp_alpha * mu_kills)
        total_kills = int(np_rng.negative_binomial(n=nb_n, p=nb_p))
        total_kills = max(2, total_kills)

        # 3. Kill Split between Winner and Loser
        alpha_w = float(self._winner_kill_share.get("alpha", 14.0))
        beta_w = float(self._winner_kill_share.get("beta", 6.0))
        winner_share = float(np_rng.beta(alpha_w, beta_w))
        winner_kills = max(1, int(round(winner_share * total_kills)))
        if winner_kills >= total_kills:
            winner_kills = total_kills - 1
        loser_kills = max(1, total_kills - winner_kills)
        total_kills = winner_kills + loser_kills

        if winner == "A":
            team_a_kills = winner_kills
            team_b_kills = loser_kills
        else:
            team_a_kills = loser_kills
            team_b_kills = winner_kills

        # Team deaths: Conservation law
        team_a_deaths = team_b_kills
        team_b_deaths = team_a_kills

        # 4. Player Kills Distribution (Dirichlet-Multinomial)
        def sample_player_kills(team_kills: int, player_ids: list[str]) -> list[int]:
            if team_kills <= 0:
                return [0, 0, 0, 0, 0]
            tot_alpha = sum(self._role_kill_shares.get(r, 20.0) for r in ROLES)
            weights = []
            for r, p_id in zip(ROLES, player_ids):
                base_w = self._role_kill_shares.get(r, 20.0) / tot_alpha
                prof = self._player_profiles.get(p_id) or self._player_profiles.get(p_id.lower())
                delta = prof.get("delta_ks", 0.0) if prof else 0.0
                weights.append(max(0.01, base_w + delta))
            w_sum = sum(weights)
            alpha_vec = [100.0 * (w / w_sum) for w in weights]
            p_kill = np_rng.dirichlet(alpha_vec)
            counts = np_rng.multinomial(team_kills, p_kill)
            return [int(c) for c in counts]

        kills_a = sample_player_kills(team_a_kills, players_a)
        kills_b = sample_player_kills(team_b_kills, players_b)

        # 5. Player Deaths Distribution (Dirichlet-Multinomial)
        def sample_player_deaths(team_deaths: int, player_ids: list[str]) -> list[int]:
            if team_deaths <= 0:
                return [0, 0, 0, 0, 0]
            tot_alpha = sum(self._role_death_shares.get(r, 20.0) for r in ROLES)
            weights = []
            for r, p_id in zip(ROLES, player_ids):
                base_w = self._role_death_shares.get(r, 20.0) / tot_alpha
                prof = self._player_profiles.get(p_id) or self._player_profiles.get(p_id.lower())
                mult = prof.get("death_multiplier", 1.0) if prof else 1.0
                weights.append(max(0.01, base_w * mult))
            w_sum = sum(weights)
            alpha_vec = [100.0 * (w / w_sum) for w in weights]
            p_death = np_rng.dirichlet(alpha_vec)
            counts = np_rng.multinomial(team_deaths, p_death)
            return [int(c) for c in counts]

        deaths_a = sample_player_deaths(team_a_deaths, players_a)
        deaths_b = sample_player_deaths(team_b_deaths, players_b)

        # 6. Player Assists: Kill-by-kill generation ensuring A_i <= TeamKills - Kills_i
        def sample_player_assists(team_kills: int, player_kills: list[int]) -> list[int]:
            if team_kills <= 0:
                return [0, 0, 0, 0, 0]
            assist_weights_dict = self._role_assist_factors.get("weights", {})
            raw_weights = [assist_weights_dict.get(r, 0.2) for r in ROLES]
            assists = [0, 0, 0, 0, 0]

            # Construct list of killer index for each team kill
            killer_indices: list[int] = []
            for idx, k in enumerate(player_kills):
                killer_indices.extend([idx] * k)

            # For each kill, sample assists among teammates
            for killer_idx in killer_indices:
                teammate_indices = [idx for idx in range(5) if idx != killer_idx]
                t_weights = [raw_weights[idx] for idx in teammate_indices]
                w_sum = sum(t_weights)
                norm_w = [w / w_sum for w in t_weights]

                # Poisson number of assists on this kill: lambda ~ 2.15, bounded by [0, 4]
                n_assist = min(4, int(np_rng.poisson(2.15)))
                if n_assist > 0:
                    chosen = np_rng.choice(teammate_indices, size=n_assist, replace=False, p=norm_w)
                    for c_idx in chosen:
                        assists[c_idx] += 1

            # Invariant safety assertion: assists[i] <= team_kills - player_kills[i]
            for idx in range(5):
                max_allowed = max(0, team_kills - player_kills[idx])
                assists[idx] = min(assists[idx], max_allowed)

            return assists

        assists_a = sample_player_assists(team_a_kills, kills_a)
        assists_b = sample_player_assists(team_b_kills, kills_b)

        # 7. CS (Creep Score): Normal(CSM * T / 60, sigma=18)
        def sample_player_cs(player_ids: list[str]) -> list[int]:
            cs_list = []
            for r, p_id in zip(ROLES, player_ids):
                prof = self._player_profiles.get(p_id) or self._player_profiles.get(p_id.lower())
                csm = prof.get("csm", self._role_csm.get(r, 8.0)) if prof else self._role_csm.get(r, 8.0)
                mean_cs = csm * duration_min
                sampled_cs = int(round(max(0.0, float(np_rng.normal(loc=mean_cs, scale=18.0)))))
                cs_list.append(sampled_cs)
            return cs_list

        cs_a = sample_player_cs(players_a)
        cs_b = sample_player_cs(players_b)

        # 8. Objectives (Dragons and Barons)
        d_rate = float(self._objective_rates.get("dragon_rate_per_minute", 0.145))
        d_win_share = float(self._objective_rates.get("dragon_winner_share", 0.68))
        tot_dragons = int(np_rng.poisson(d_rate * duration_min))
        win_dragons = int(np_rng.binomial(tot_dragons, d_win_share)) if tot_dragons > 0 else 0
        lose_dragons = tot_dragons - win_dragons

        b_rate = float(self._objective_rates.get("baron_rate_per_minute", 0.042))
        b_win_share = float(self._objective_rates.get("baron_winner_share", 0.82))
        baron_time = max(0.0, duration_min - 20.0)
        tot_barons = int(np_rng.poisson(b_rate * baron_time)) if baron_time > 0 else 0
        win_barons = int(np_rng.binomial(tot_barons, b_win_share)) if tot_barons > 0 else 0
        lose_barons = tot_barons - win_barons

        if winner == "A":
            dragons_a, dragons_b = win_dragons, lose_dragons
            barons_a, barons_b = win_barons, lose_barons
        else:
            dragons_a, dragons_b = lose_dragons, win_dragons
            barons_a, barons_b = lose_barons, win_barons

        # 9. Pentakills
        penta_rate = float(self._objective_rates.get("pentakill_rate_per_game", 0.016))
        penta_prob = float(np.clip(penta_rate * (total_kills / 26.5), 0.001, 0.25))
        has_pentakill = bool(np_rng.random() < penta_prob)
        pentakill_player: str | None = None

        if has_pentakill:
            # Find the player with the most kills across all 10 players
            all_kill_candidates: list[tuple[int, int, str]] = []  # (kills, priority, player_id)
            role_priority = {"ADC": 5, "MID": 4, "TOP": 3, "JUNGLE": 2, "SUPPORT": 1}
            for idx, r in enumerate(ROLES):
                all_kill_candidates.append((kills_a[idx], role_priority.get(r, 0), players_a[idx]))
                all_kill_candidates.append((kills_b[idx], role_priority.get(r, 0), players_b[idx]))
            all_kill_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            pentakill_player = all_kill_candidates[0][2]

        # 10. Assemble PlayerGameStats for all 10 players
        player_stats: dict[str, PlayerGameStats] = {}
        for idx, r in enumerate(ROLES):
            ka, da, aa, csa = kills_a[idx], deaths_a[idx], assists_a[idx], cs_a[idx]
            kda_a = (ka + aa) / max(1, da)
            key_a = f"A_{r}"
            player_stats[key_a] = PlayerGameStats(
                player_id=players_a[idx],
                role=r,
                champion_id=champs_a[idx],
                kills=ka,
                deaths=da,
                assists=aa,
                cs=csa,
                kda=float(kda_a),
            )

            kb, db, ab, csb = kills_b[idx], deaths_b[idx], assists_b[idx], cs_b[idx]
            kda_b = (kb + ab) / max(1, db)
            key_b = f"B_{r}"
            player_stats[key_b] = PlayerGameStats(
                player_id=players_b[idx],
                role=r,
                champion_id=champs_b[idx],
                kills=kb,
                deaths=db,
                assists=ab,
                cs=csb,
                kda=float(kda_b),
            )

        return GameStatsResult(
            duration_seconds=duration_sec,
            team_a_kills=team_a_kills,
            team_b_kills=team_b_kills,
            team_a_dragons=dragons_a,
            team_b_dragons=dragons_b,
            team_a_barons=barons_a,
            team_b_barons=barons_b,
            player_stats=player_stats,
            has_pentakill=has_pentakill,
            pentakill_player=pentakill_player,
        )


def train_player_statistics_model(
    as_of_date: str = "2026-01-01",
    output_dir: str = "data/06_models/player_statistics",
    min_games: int = 10,
    db: Any = None,
) -> dict[str, Any]:
    """Fit empirical prior distributions and serialize model artifacts.

    Parameters
    ----------
    as_of_date : str
        Temporal cutoff; only games strictly prior to this date are used.
    output_dir : str
        Directory where model_priors.joblib and model_metadata.json are stored.
    min_games : int
        Minimum games for per-player profiles.
    db : Any, optional
        Database session or connection.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary summarizing the trained model.
    """
    os.makedirs(output_dir, exist_ok=True)
    priors = extract_player_stat_priors(as_of_date=as_of_date, min_games=min_games, db=db)

    # Serialize priors with joblib
    priors_path = os.path.join(output_dir, "model_priors.joblib")
    joblib.dump(priors, priors_path)

    # Compute SHA-256
    with open(priors_path, "rb") as f:
        file_bytes = f.read()
    sha256_hex = hashlib.sha256(file_bytes).hexdigest()

    metadata = {
        "model_name": "player_statistics_hierarchical_model",
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of_date,
        "sample_games": priors.get("sample_games", 0),
        "sample_player_records": priors.get("sample_player_records", 0),
        "num_player_profiles": len(priors.get("player_profiles", {})),
        "role_kill_shares": priors.get("role_kill_shares", {}),
        "role_death_shares": priors.get("role_death_shares", {}),
        "role_assist_factors": priors.get("role_assist_factors", {}),
        "winner_kill_share": priors.get("winner_kill_share", {}),
        "objective_rates": priors.get("objective_rates", {}),
        "role_csm": priors.get("role_csm", {}),
        "duration_distribution": {
            "family": "lognormal",
            "mu": float(math.log(1920.0)),
            "sigma": 0.16,
            "bounds": [900, 3600],
        },
        "kills_distribution": {
            "family": "negative_binomial",
            "base_mu": 26.5,
            "dispersion_alpha": 0.18,
        },
        "sha256": sha256_hex,
        "artifacts": {
            "priors": "model_priors.joblib",
            "metadata": "model_metadata.json",
        },
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_player_statistics_model(
    model_dir: str = "data/06_models/player_statistics",
) -> PlayerStatsDistributionModel:
    """Load trained player statistics model from artifacts or fallback to defaults."""
    priors_path = os.path.join(model_dir, "model_priors.joblib")
    if os.path.exists(priors_path):
        priors = joblib.load(priors_path)
        return PlayerStatsDistributionModel(priors=priors)
    return PlayerStatsDistributionModel()
