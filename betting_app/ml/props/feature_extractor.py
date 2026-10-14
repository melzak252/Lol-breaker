"""Chronological, leakage-safe pre-match pace feature extractor for prop models."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any
import numpy as np

from betting_app.ml.props.schemas import (
    LeaguePaceContext,
    MatchPropContext,
    MatchupSpreadFeatures,
    PaceFeatures,
    RecentGameSummary,
    TeamRecentForm,
)
from src.ratings.elo import EloRating

# Regional pace baselines (mean kills, std kills, duration minutes, ckpm)
# Computed from 38,804 games (2022–2026) in reports/eda_prop_markets_idea018.md
LEAGUE_PACE_PRIORS: dict[str, dict[str, Any]] = {
    "LCK": {"mean_kills": 25.73, "mean_duration": 32.44, "ckpm": 0.804, "p25": 19.0, "median": 25.0, "p75": 32.0, "pace_category": "kontrolowana (wolne makro)"},
    "LCS": {"mean_kills": 26.46, "mean_duration": 32.76, "ckpm": 0.814, "p25": 20.0, "median": 26.0, "p75": 32.0, "pace_category": "kontrolowana (wolne makro)"},
    "LTA": {"mean_kills": 26.46, "mean_duration": 32.76, "ckpm": 0.814, "p25": 20.0, "median": 26.0, "p75": 32.0, "pace_category": "kontrolowana (wolne makro)"},
    "LEC": {"mean_kills": 27.44, "mean_duration": 32.90, "ckpm": 0.839, "p25": 21.0, "median": 27.0, "p75": 33.0, "pace_category": "zrównoważona"},
    "LPL": {"mean_kills": 28.21, "mean_duration": 31.87, "ckpm": 0.895, "p25": 22.0, "median": 28.0, "p75": 34.0, "pace_category": "dynamiczna (wysokie tempo)"},
    "LCK CL": {"mean_kills": 28.32, "mean_duration": 32.14, "ckpm": 0.889, "p25": 22.0, "median": 28.0, "p75": 34.0, "pace_category": "dynamiczna (wysokie tempo)"},
    "CBLOL": {"mean_kills": 28.23, "mean_duration": 32.92, "ckpm": 0.864, "p25": 22.0, "median": 28.0, "p75": 34.0, "pace_category": "zrównoważona"},
    "SUPERLIGA": {"mean_kills": 28.82, "mean_duration": 32.16, "ckpm": 0.906, "p25": 22.5, "median": 28.5, "p75": 35.0, "pace_category": "dynamiczna (wysokie tempo)"},
    "ULTRALIGA": {"mean_kills": 29.10, "mean_duration": 31.69, "ckpm": 0.926, "p25": 22.5, "median": 29.0, "p75": 35.0, "pace_category": "dynamiczna (wysokie tempo)"},
    "LFL": {"mean_kills": 29.99, "mean_duration": 32.72, "ckpm": 0.923, "p25": 23.0, "median": 29.0, "p75": 36.0, "pace_category": "dynamiczna (wysokie tempo)"},
    "VCS": {"mean_kills": 30.89, "mean_duration": 31.11, "ckpm": 1.004, "p25": 24.0, "median": 30.0, "p75": 37.0, "pace_category": "bardzo wysokie tempo (fiesta)"},
    "PRIME LEAGUE": {"mean_kills": 31.27, "mean_duration": 31.57, "ckpm": 0.999, "p25": 24.0, "median": 31.0, "p75": 37.5, "pace_category": "bardzo wysokie tempo (fiesta)"},
    "DEFAULT": {"mean_kills": 29.60, "mean_duration": 31.90, "ckpm": 0.930, "p25": 23.0, "median": 29.0, "p75": 36.0, "pace_category": "zrównoważona"},
}


def normalize_league_key(league_name: str | None) -> str:
    """Classify tournament/league string to canonical prior key."""
    if not league_name:
        return "DEFAULT"
    upper = str(league_name).strip().upper()
    if "LCK CL" in upper or "LCK CHALLENGERS" in upper:
        return "LCK CL"
    for key in ("LCK", "LPL", "LEC", "LCS", "LTA", "VCS", "CBLOL", "LFL", "PRIME LEAGUE", "SUPERLIGA", "ULTRALIGA"):
        if key in upper:
            return key
    return "DEFAULT"


class ChronologicalPaceTracker:
    """Stateful chronological pace tracker maintaining team history without lookahead bias."""

    def __init__(
        self,
        window_size: int = 20,
        prior_weight: float = 5.0,
        elo_system: EloRating | None = None,
    ) -> None:
        self.window_size = int(window_size)
        self.prior_weight = float(prior_weight)
        self.elo = elo_system if elo_system is not None else EloRating(initial_rating=1500.0, k_team=32.0)
        # team_name -> deque of recent games (dict: kills, deaths, duration_min)
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.window_size))
    def get_team_pace(self, team_name: str, league_name: str | None = None) -> PaceFeatures:
        """Extract pre-match pace features for a team using historical games up to now."""
        normalized_name = str(team_name).strip().lower()
        games = list(self._history.get(normalized_name, []))
        sample_size = len(games)

        league_key = normalize_league_key(league_name)
        prior = LEAGUE_PACE_PRIORS.get(league_key, LEAGUE_PACE_PRIORS["DEFAULT"])
        prior_kills = prior["mean_kills"] / 2.0  # single team offensive kills
        prior_deaths = prior["mean_kills"] / 2.0
        prior_dur = prior["mean_duration"]
        prior_ckpm = prior["ckpm"]

        if sample_size == 0:
            return PaceFeatures(
                team_name=team_name,
                sample_games=0,
                avg_kills=round(prior_kills, 2),
                avg_deaths=round(prior_deaths, 2),
                avg_duration_minutes=round(prior_dur, 2),
                ckpm=round(prior_ckpm, 4),
            )

        # Empirical sample means
        sample_kills = sum(g["kills"] for g in games) / sample_size
        sample_deaths = sum(g["deaths"] for g in games) / sample_size
        sample_dur = sum(g["duration_minutes"] for g in games) / sample_size
        sample_ckpm = (sample_kills + sample_deaths) / max(sample_dur, 10.0)

        # Bayesian shrinkage towards league prior for small samples
        w_sample = float(sample_size)
        w_prior = self.prior_weight
        total_w = w_sample + w_prior

        shrunk_kills = (w_sample * sample_kills + w_prior * prior_kills) / total_w
        shrunk_deaths = (w_sample * sample_deaths + w_prior * prior_deaths) / total_w
        shrunk_dur = (w_sample * sample_dur + w_prior * prior_dur) / total_w
        shrunk_ckpm = (w_sample * sample_ckpm + w_prior * prior_ckpm) / total_w

        return PaceFeatures(
            team_name=team_name,
            sample_games=sample_size,
            avg_kills=round(shrunk_kills, 3),
            avg_deaths=round(shrunk_deaths, 3),
            avg_duration_minutes=round(shrunk_dur, 2),
            ckpm=round(shrunk_ckpm, 4),
        )

    def get_matchup_expected_pace(
        self,
        team_a: str,
        team_b: str,
        league_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute symmetric expected combined kills and duration for a matchup."""
        pace_a = self.get_team_pace(team_a, league_name)
        pace_b = self.get_team_pace(team_b, league_name)

        # Team A offensive kills adjusted by Team B defensive deaths
        exp_kills_a = (pace_a.avg_kills + pace_b.avg_deaths) / 2.0
        exp_kills_b = (pace_b.avg_kills + pace_a.avg_deaths) / 2.0

        # Total combined expected kills is strictly symmetric
        expected_total_kills = exp_kills_a + exp_kills_b
        expected_duration = (pace_a.avg_duration_minutes + pace_b.avg_duration_minutes) / 2.0
        expected_ckpm = (pace_a.ckpm + pace_b.ckpm) / 2.0

        return {
            "expected_total_kills": round(expected_total_kills, 3),
            "expected_duration_minutes": round(expected_duration, 2),
            "expected_ckpm": round(expected_ckpm, 4),
            "team_a_pace": pace_a,
            "team_b_pace": pace_b,
            "league_key": normalize_league_key(league_name),
        }

    def get_matchup_spread_features(
        self,
        team_a: str,
        team_b: str,
        league_name: str | None = None,
    ) -> MatchupSpreadFeatures:
        """Extract pre-match rating spread, win probability, and pace metrics."""
        pace_matchup = self.get_matchup_expected_pace(team_a, team_b, league_name)
        pace_a = pace_matchup["team_a_pace"]
        pace_b = pace_matchup["team_b_pace"]

        exp_pace_a = (pace_a.avg_kills + pace_b.avg_deaths) / 2.0
        exp_pace_b = (pace_b.avg_kills + pace_a.avg_deaths) / 2.0

        r_a = self.elo.get_team_rating(team_a)
        r_b = self.elo.get_team_rating(team_b)
        elo_diff = r_a - r_b
        abs_diff = abs(elo_diff)

        p_a = self.elo._expected_score(r_a, r_b)
        p_b = 1.0 - p_a

        return MatchupSpreadFeatures(
            team_a=team_a,
            team_b=team_b,
            rating_a=round(r_a, 1),
            rating_b=round(r_b, 1),
            elo_diff=round(elo_diff, 1),
            abs_elo_diff=round(abs_diff, 1),
            prob_win_a=round(p_a, 4),
            prob_win_b=round(p_b, 4),
            exp_pace_kills_a=round(exp_pace_a, 3),
            exp_pace_kills_b=round(exp_pace_b, 3),
            expected_total_kills=pace_matchup["expected_total_kills"],
            expected_duration_minutes=pace_matchup["expected_duration_minutes"],
        )

    def update_game(
        self,
        team_1: str,
        team_2: str,
        kills_1: int | float,
        kills_2: int | float,
        duration_minutes: float,
        winner_team_1: bool,
        date: str | None = None,
    ) -> None:
        """Update both pace sliding windows and ratings simultaneously from identical pre-match state."""
        # Update pace
        self.update_post_game(
            team_name=team_1,
            kills=kills_1,
            deaths=kills_2,
            duration_minutes=duration_minutes,
            opponent=team_2,
            won=winner_team_1,
            date=date,
        )
        self.update_post_game(
            team_name=team_2,
            kills=kills_2,
            deaths=kills_1,
            duration_minutes=duration_minutes,
            opponent=team_1,
            won=not winner_team_1,
            date=date,
        )

        # Update Elo
        score_1 = 1 if winner_team_1 else 0
        score_2 = 0 if winner_team_1 else 1
        self.elo.update_team(team_1, team_2, score_1, score_2)

    def update_post_game(
        self,
        team_name: str,
        kills: int | float,
        deaths: int | float,
        duration_minutes: float,
        opponent: str | None = None,
        won: bool | None = None,
        date: str | None = None,
    ) -> None:
        """Update tracker with a completed game result (strictly post-game)."""
        normalized_name = str(team_name).strip().lower()
        self._history[normalized_name].append({
            "kills": float(kills),
            "deaths": float(deaths),
            "duration_minutes": float(duration_minutes),
            "opponent": str(opponent) if opponent else "",
            "won": bool(won) if won is not None else False,
            "date": str(date) if date else None,
            "total_kills": float(kills) + float(deaths),
        })

    def get_team_recent_form(self, team_name: str, n: int = 5) -> TeamRecentForm:
        """Extract last N games and rolling averages for a team."""
        normalized_name = str(team_name).strip().lower()
        games = list(self._history.get(normalized_name, []))
        if not games:
            return TeamRecentForm(
                team_name=team_name,
                sample_size=0,
                avg_kills=0.0,
                avg_deaths=0.0,
                avg_total_kills=0.0,
                avg_duration_minutes=0.0,
                recent_games=[],
            )

        last_n = games[-n:]
        recent_summaries = [
            RecentGameSummary(
                opponent=g.get("opponent", "Nieznany"),
                kills_for=int(round(g["kills"])),
                kills_against=int(round(g["deaths"])),
                total_kills=int(round(g.get("total_kills", g["kills"] + g["deaths"]))),
                duration_minutes=round(g["duration_minutes"], 1),
                won=bool(g.get("won", False)),
                date=g.get("date"),
            )
            for g in reversed(last_n)  # Most recent first
        ]

        kills_list = [int(round(g["kills"])) for g in last_n]
        avg_k = float(np.mean(kills_list))
        avg_d = float(np.mean([g["deaths"] for g in last_n]))
        avg_tot = float(np.mean([g.get("total_kills", g["kills"] + g["deaths"]) for g in last_n]))
        avg_dur = float(np.mean([g["duration_minutes"] for g in last_n]))

        min_k = int(min(kills_list))
        max_k = int(max(kills_list))
        std_k = float(np.std(kills_list)) if len(kills_list) > 1 else 0.0

        brackets = {
            "<10": sum(1 for k in kills_list if k < 10),
            "10-14": sum(1 for k in kills_list if 10 <= k <= 14),
            "15-19": sum(1 for k in kills_list if 15 <= k <= 19),
            "20-24": sum(1 for k in kills_list if 20 <= k <= 24),
            "25+": sum(1 for k in kills_list if k >= 25),
        }

        return TeamRecentForm(
            team_name=team_name,
            sample_size=len(games),
            avg_kills=round(avg_k, 1),
            avg_deaths=round(avg_d, 1),
            avg_total_kills=round(avg_tot, 1),
            avg_duration_minutes=round(avg_dur, 1),
            min_kills=min_k,
            max_kills=max_k,
            std_kills=round(std_k, 1),
            kill_brackets=brackets,
            recent_games=recent_summaries,
        )

    def get_league_context(self, league_name: str | None) -> LeaguePaceContext:
        """Retrieve league benchmark pace statistics and categorisation."""
        key = normalize_league_key(league_name)
        priors = LEAGUE_PACE_PRIORS.get(key, LEAGUE_PACE_PRIORS["DEFAULT"])
        return LeaguePaceContext(
            league_name=league_name,
            league_key=key,
            avg_kills=float(priors["mean_kills"]),
            avg_duration_minutes=float(priors["mean_duration"]),
            ckpm=float(priors["ckpm"]),
            pace_category=str(priors.get("pace_category", "zrównoważona")),
            p25_kills=float(priors.get("p25", 22.0)),
            median_kills=float(priors.get("median", 28.0)),
            p75_kills=float(priors.get("p75", 34.0)),
        )

    def get_matchup_prop_context(
        self,
        team_a: str,
        team_b: str,
        league_name: str | None,
        last_n: int = 5,
    ) -> MatchPropContext:
        """Build complete pre-match historical and league context."""
        form_a = self.get_team_recent_form(team_a, n=last_n)
        form_b = self.get_team_recent_form(team_b, n=last_n)
        league_ctx = self.get_league_context(league_name)

        # Expected pace
        pace_matchup = self.get_matchup_expected_pace(team_a, team_b, league_name)
        exp_pace = pace_matchup["expected_total_kills"]
        delta_vs_league = round(exp_pace - league_ctx.avg_kills, 1)

        # Build narrative
        delta_sign = f"+{delta_vs_league}" if delta_vs_league > 0 else f"{delta_vs_league}"
        pace_eval = "podwyższonym" if delta_vs_league > 1.5 else ("obniżonym" if delta_vs_league < -1.5 else "zbliżonym do średniej")

        narrative_parts = [
            f"Mecz o {pace_eval} tempie ({delta_sign} zabójstw względem średniej ligi {league_ctx.league_key}: {league_ctx.avg_kills:.1f}).",
        ]
        if form_a.recent_games:
            narrative_parts.append(f"{team_a} w ostatnich {len(form_a.recent_games)} grach notowało średnio {form_a.avg_total_kills:.1f} zabójstw na mapę.")
        if form_b.recent_games:
            narrative_parts.append(f"{team_b} notowało średnio {form_b.avg_total_kills:.1f} zabójstw na mapę.")

        return MatchPropContext(
            team_a_form=form_a,
            team_b_form=form_b,
            league_context=league_ctx,
            expected_pace_kills=round(exp_pace, 1),
            delta_vs_league_avg=delta_vs_league,
            summary_narrative=" ".join(narrative_parts),
        )
