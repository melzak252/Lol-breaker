"""Service for player comparisons, rating trajectories, and head-to-head analysis."""

from __future__ import annotations

import math
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from sqlalchemy.orm import Session

from betting_app.api.deps import query_df, query_one
from betting_app.api.schemas import (
    H2HGameItem,
    H2HSummary,
    ModelVerdict,
    PlayerComparisonResponse,
    PlayerProfileDetail,
    PlayerSearchItem,
    PlayerSystemRating,
    RatingTimelinePoint,
    SystemAdvantage,
    TopChampionItem,
)
from betting_app.services.rating_contract import OPERATIONAL_RATINGS_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DB_PATH = PROJECT_ROOT / "data" / "player_rating_history.sqlite"

SYSTEM_LABELS: dict[str, str] = {
    "elo": "Elo",
    "gl": "Glicko-2",
    "ts": "TrueSkill",
    "os": "OpenSkill",
    "pl": "Plackett–Luce",
    "tm": "Thurstone–Mosteller",
}


def search_players(
    db: Session,
    query: str,
    limit: int = 15,
) -> list[PlayerSearchItem]:
    """Search active or historical players by nickname or ID."""
    clean_query = query.strip().lower()
    if not clean_query:
        return []

    # Check operational version or fallback to latest-full
    active_version = _get_active_ratings_version(db)

    sql = """
        WITH player_matches AS (
            SELECT
                normalized_entity_name AS player_id,
                MAX(entity_name) AS player_name,
                MAX(team_name) AS team_name,
                MAX(role) AS role,
                MAX(games_played) AS games_played,
                MAX(last_match_at) AS last_match_at,
                MAX(CASE WHEN rating_system = 'elo' THEN rating_value END) AS current_elo,
                MAX(CASE WHEN rating_system = 'gl' THEN rating_value END) AS current_gl
            FROM entity_ratings
            WHERE entity_type = 'player'
              AND (ratings_version = :version OR :version IS NULL)
              AND rating_value IS NOT NULL
              AND (
                    LOWER(entity_name) LIKE :pattern
                 OR normalized_entity_name = :exact_query
              )
            GROUP BY normalized_entity_name
        )
        SELECT *
        FROM player_matches
        ORDER BY
            CASE WHEN LOWER(player_name) = :exact_query THEN 0 ELSE 1 END,
            CASE WHEN LOWER(player_name) LIKE :prefix_pattern THEN 0 ELSE 1 END,
            games_played DESC
        LIMIT :limit
    """
    rows = query_df(
        db,
        sql,
        {
            "version": active_version,
            "pattern": f"%{clean_query}%",
            "prefix_pattern": f"{clean_query}%",
            "exact_query": clean_query,
            "limit": limit,
        },
    )

    results: list[PlayerSearchItem] = []
    for r in rows:
        results.append(
            PlayerSearchItem(
                player_id=str(r["player_id"]),
                player_name=str(r["player_name"]),
                team_name=str(r["team_name"]) if r.get("team_name") else None,
                role=str(r["role"]) if r.get("role") else None,
                games_played=int(r.get("games_played") or 0),
                current_elo=round(float(r["current_elo"]), 1) if r.get("current_elo") is not None else None,
                current_gl=round(float(r["current_gl"]), 1) if r.get("current_gl") is not None else None,
                last_match_at=str(r["last_match_at"]) if r.get("last_match_at") else None,
            )
        )
    return results


def get_player_profile(db: Session, player_id: str) -> PlayerProfileDetail | None:
    """Fetch complete profile, current multi-system ratings, and career stats for a player."""
    active_version = _get_active_ratings_version(db)

    # 1. Load current ratings across all systems
    ratings_rows = query_df(
        db,
        """
        SELECT entity_name, team_name, role, rating_system, rating_value, rd, sigma, games_played, last_match_at
        FROM entity_ratings
        WHERE entity_type = 'player'
          AND normalized_entity_name = :player_id
          AND (ratings_version = :version OR :version IS NULL)
          AND rating_value IS NOT NULL
        ORDER BY rating_system
        """,
        {"player_id": player_id, "version": active_version},
    )

    if not ratings_rows:
        # Fallback to golgg_game_players to verify player existence
        first_row = query_one(
            db,
            "SELECT player_id, player_name, team_name, role FROM golgg_game_players WHERE player_id = :player_id LIMIT 1",
            {"player_id": player_id},
        )
        if not first_row:
            return None
        player_name = str(first_row.get("player_name") or player_id)
        team_name = str(first_row.get("team_name") or "") or None
        role = str(first_row.get("role") or "") or None
        games_played = 0
        last_match_at = None
        ratings_dict: dict[str, PlayerSystemRating] = {}
    else:
        sample = ratings_rows[0]
        player_name = str(sample.get("entity_name") or player_id)
        team_name = str(sample.get("team_name") or "") or None
        role = str(sample.get("role") or "") or None
        games_played = int(max(r.get("games_played") or 0 for r in ratings_rows))
        last_match_at = next((str(r["last_match_at"]) for r in ratings_rows if r.get("last_match_at")), None)

        ratings_dict = {}
        for r in ratings_rows:
            sys_code = str(r["rating_system"])
            val = float(r["rating_value"])
            rd_val = float(r["rd"]) if r.get("rd") is not None else None
            sigma_val = float(r["sigma"]) if r.get("sigma") is not None else None
            ratings_dict[sys_code] = PlayerSystemRating(
                system=sys_code,
                rating_value=round(val, 2),
                rd=round(rd_val, 1) if rd_val is not None else None,
                sigma=round(sigma_val, 2) if sigma_val is not None else None,
            )

    # 2. Career stats from golgg_game_players & golgg_games
    career_row = query_one(
        db,
        """
        SELECT
            COUNT(*) AS total_games,
            SUM(CASE WHEN (gp.side = 't1' AND g.team1_win = 1) OR (gp.side = 't2' AND g.team2_win = 1) THEN 1 ELSE 0 END) AS wins,
            MIN(g.date) AS first_date,
            MAX(g.date) AS last_date
        FROM golgg_game_players gp
        JOIN golgg_games g ON gp.game_id = g.game_id
        WHERE gp.player_id = :player_id
        """,
        {"player_id": player_id},
    ) or {}

    total_games = int(career_row.get("total_games") or games_played or 0)
    career_wins = int(career_row.get("wins") or 0)
    career_losses = max(0, total_games - career_wins)
    career_win_rate = round(career_wins / total_games, 3) if total_games > 0 else 0.0
    first_date = str(career_row.get("first_date"))[:10] if career_row.get("first_date") else None
    last_date = str(career_row.get("last_date"))[:10] if career_row.get("last_date") else last_match_at

    career_years = None
    if first_date and last_date:
        try:
            d1 = int(first_date[:4])
            d2 = int(last_date[:4])
            career_years = round(max(0.5, d2 - d1 + 1), 1)
        except Exception:
            career_years = None

    # 3. Teams played for
    team_rows = query_df(
        db,
        """
        SELECT gp.team_name, COUNT(*) AS games
        FROM golgg_game_players gp
        WHERE gp.player_id = :player_id AND gp.team_name IS NOT NULL
        GROUP BY gp.team_name
        ORDER BY games DESC
        LIMIT 6
        """,
        {"player_id": player_id},
    )
    teams = [str(r["team_name"]) for r in team_rows if r.get("team_name")]
    if not team_name and teams:
        team_name = teams[0]

    # 4. Top champions
    champ_rows = query_df(
        db,
        """
        SELECT
            gp.champion_name,
            COUNT(*) AS games,
            SUM(CASE WHEN (gp.side = 't1' AND g.team1_win = 1) OR (gp.side = 't2' AND g.team2_win = 1) THEN 1 ELSE 0 END) AS wins
        FROM golgg_game_players gp
        JOIN golgg_games g ON gp.game_id = g.game_id
        WHERE gp.player_id = :player_id
          AND gp.champion_name IS NOT NULL
          AND gp.champion_name != ''
        GROUP BY gp.champion_name
        ORDER BY games DESC
        LIMIT 5
        """,
        {"player_id": player_id},
    )
    top_champions: list[TopChampionItem] = []
    for cr in champ_rows:
        g_cnt = int(cr["games"])
        w_cnt = int(cr.get("wins") or 0)
        wr = round(w_cnt / g_cnt, 3) if g_cnt > 0 else 0.0
        top_champions.append(
            TopChampionItem(
                champion_name=str(cr["champion_name"]),
                games=g_cnt,
                wins=w_cnt,
                win_rate=wr,
            )
        )

    # 5. Peak ratings from history DB if available
    peak_elo, peak_elo_date, peak_gl, peak_gl_date = _get_player_peaks(player_id)

    return PlayerProfileDetail(
        player_id=player_id,
        player_name=player_name,
        team_name=team_name,
        role=role,
        games_played=total_games,
        career_wins=career_wins,
        career_losses=career_losses,
        career_win_rate=career_win_rate,
        career_first_date=first_date,
        career_last_date=last_date,
        career_years=career_years,
        teams=teams,
        top_champions=top_champions,
        ratings=ratings_dict,
        peak_elo=peak_elo,
        peak_elo_date=peak_elo_date,
        peak_gl=peak_gl,
        peak_gl_date=peak_gl_date,
    )


def get_head_to_head_summary(db: Session, player_a_id: str, player_b_id: str, limit: int = 50) -> H2HSummary:
    """Find all professional games where Player A and Player B faced each other."""
    sql = """
        SELECT
            g.game_id,
            g.match_id,
            g.date,
            g.tournament_name,
            gp1.team_name AS team_a,
            gp1.champion_name AS champ_a,
            gp2.team_name AS team_b,
            gp2.champion_name AS champ_b,
            CASE
                WHEN (gp1.side = 't1' AND g.team1_win = 1) OR (gp1.side = 't2' AND g.team2_win = 1) THEN 'a'
                ELSE 'b'
            END AS winner
        FROM golgg_game_players gp1
        JOIN golgg_game_players gp2
          ON gp1.game_id = gp2.game_id
         AND gp1.side != gp2.side
        JOIN golgg_games g
          ON gp1.game_id = g.game_id
        WHERE gp1.player_id = :player_a
          AND gp2.player_id = :player_b
        ORDER BY g.date DESC, g.game_id DESC
    """
    rows = query_df(db, sql, {"player_a": player_a_id, "player_b": player_b_id})

    total_games = len(rows)
    wins_a = sum(1 for r in rows if r.get("winner") == "a")
    wins_b = total_games - wins_a
    win_rate_a = round(wins_a / total_games, 3) if total_games > 0 else 0.0
    win_rate_b = round(wins_b / total_games, 3) if total_games > 0 else 0.0

    recent_games: list[H2HGameItem] = []
    for r in rows[:limit]:
        recent_games.append(
            H2HGameItem(
                game_id=str(r["game_id"]),
                match_id=str(r["match_id"]),
                date=str(r["date"])[:10] if r.get("date") else None,
                tournament_name=str(r["tournament_name"]) if r.get("tournament_name") else None,
                team_a=str(r["team_a"]) if r.get("team_a") else None,
                champ_a=str(r["champ_a"]) if r.get("champ_a") else None,
                team_b=str(r["team_b"]) if r.get("team_b") else None,
                champ_b=str(r["champ_b"]) if r.get("champ_b") else None,
                winner="a" if r.get("winner") == "a" else "b",
            )
        )

    return H2HSummary(
        total_games=total_games,
        wins_a=wins_a,
        wins_b=wins_b,
        win_rate_a=win_rate_a,
        win_rate_b=win_rate_b,
        recent_games=recent_games,
    )


def calculate_model_verdict(
    profile_a: PlayerProfileDetail,
    profile_b: PlayerProfileDetail,
    h2h: H2HSummary,
    selected_system: str = "unified",
) -> ModelVerdict:
    """Synthesize multi-system ratings, probability formulas, and historical context into a verdict."""
    system_advantages: list[SystemAdvantage] = []
    favor_a_count = 0
    favor_b_count = 0
    tied_count = 0

    # Common systems: elo, gl, ts, os, pl, tm
    all_systems = ("elo", "gl", "ts", "os", "pl", "tm")
    system_probs_a: list[float] = []

    for sys_code in all_systems:
        rat_a = profile_a.ratings.get(sys_code)
        rat_b = profile_b.ratings.get(sys_code)
        label = SYSTEM_LABELS.get(sys_code, sys_code.upper())

        if not rat_a or not rat_b:
            continue

        va = rat_a.rating_value
        vb = rat_b.rating_value
        diff = round(va - vb, 2)

        # Expected score / win prob calculation per system
        if sys_code == "elo":
            p_a = 1.0 / (1.0 + 10.0 ** ((vb - va) / 400.0))
        elif sys_code == "gl":
            rd_a = rat_a.rd or 65.0
            rd_b = rat_b.rd or 65.0
            q = math.log(10.0) / 400.0
            g_rd = 1.0 / math.sqrt(1.0 + 3.0 * (q ** 2) * (rd_a ** 2 + rd_b ** 2) / (math.pi ** 2))
            p_a = 1.0 / (1.0 + 10.0 ** (-g_rd * (va - vb) / 400.0))
        elif sys_code in ("ts", "os", "pl", "tm"):
            # Gaussian prob
            beta = 4.16 if sys_code == "ts" else 3.5
            sig_a = rat_a.sigma or 2.0
            sig_b = rat_b.sigma or 2.0
            denom = math.sqrt(sig_a ** 2 + sig_b ** 2 + 2 * (beta ** 2))
            z = (va - vb) / denom
            p_a = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        else:
            p_a = 0.5

        p_a = max(0.01, min(0.99, p_a))
        system_probs_a.append(p_a)

        if abs(diff) < 0.05:
            fav = "tied"
            tied_count += 1
        elif diff > 0:
            fav = "a"
            favor_a_count += 1
        else:
            fav = "b"
            favor_b_count += 1

        system_advantages.append(
            SystemAdvantage(
                system=sys_code,
                system_label=label,
                value_a=va,
                value_b=vb,
                difference=diff,
                favors=fav,
                win_prob_a=round(p_a, 3),
            )
        )

    # Average win probability across all available systems
    if system_probs_a:
        overall_prob_a = round(sum(system_probs_a) / len(system_probs_a), 3)
    else:
        overall_prob_a = 0.5
    overall_prob_b = round(1.0 - overall_prob_a, 3)

    # Determine overall better player
    total_sys = len(system_advantages)
    if favor_a_count > favor_b_count:
        better = "a"
        better_id = profile_a.player_id
        better_name = profile_a.player_name
    elif favor_b_count > favor_a_count:
        better = "b"
        better_id = profile_b.player_id
        better_name = profile_b.player_name
    else:
        # Tie break using win probability or Elo
        if overall_prob_a > 0.505:
            better = "a"
            better_id = profile_a.player_id
            better_name = profile_a.player_name
        elif overall_prob_a < 0.495:
            better = "b"
            better_id = profile_b.player_id
            better_name = profile_b.player_name
        else:
            better = "tied"
            better_id = None
            better_name = None

    # H2H winner
    if h2h.total_games > 0:
        if h2h.wins_a > h2h.wins_b:
            h2h_w = "a"
        elif h2h.wins_b > h2h.wins_a:
            h2h_w = "b"
        else:
            h2h_w = "tied"
    else:
        h2h_w = "tied"

    # Build qualitative Polish summary
    summary_pl = _generate_verdict_summary_pl(
        profile_a,
        profile_b,
        better,
        overall_prob_a,
        favor_a_count,
        favor_b_count,
        total_sys,
        h2h,
        system_advantages,
    )

    advantage_text = (
        f"{favor_a_count}/{total_sys} systemów wskazuje {profile_a.player_name}"
        if better == "a"
        else f"{favor_b_count}/{total_sys} systemów wskazuje {profile_b.player_name}"
        if better == "b"
        else f"Równowaga systemów ({favor_a_count} vs {favor_b_count})"
    )

    return ModelVerdict(
        better_player=better,
        better_player_id=better_id,
        better_player_name=better_name,
        win_probability_a=overall_prob_a,
        win_probability_b=overall_prob_b,
        systems_favor_a=favor_a_count,
        systems_favor_b=favor_b_count,
        systems_tied=tied_count,
        total_systems=total_sys,
        advantage_summary=advantage_text,
        system_advantages=system_advantages,
        h2h_winner=h2h_w,
        summary_pl=summary_pl,
    )


def get_player_rating_history(
    player_id: str,
    max_points: int = 250,
) -> list[RatingTimelinePoint]:
    """Retrieve smoothed or full historical rating progression for a player from SQLite cache."""
    if not HISTORY_DB_PATH.exists():
        return []

    try:
        conn = sqlite3.connect(f"file:{HISTORY_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT date, match_id, team_name, games_count, elo, gl, gl_rd, ts_mu, os_mu, pl_mu, tm_mu
            FROM player_rating_history
            WHERE player_id = ?
            ORDER BY date ASC, id ASC
            """,
            (player_id,),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []

    if not rows:
        return []

    # If count is within budget, return directly
    if len(rows) <= max_points:
        return [
            RatingTimelinePoint(
                date=str(r["date"]),
                match_id=str(r["match_id"]) if r["match_id"] else None,
                team_name=str(r["team_name"]) if r["team_name"] else None,
                games_count=int(r["games_count"]),
                elo=round(float(r["elo"]), 1),
                gl=round(float(r["gl"]), 1),
                gl_rd=round(float(r["gl_rd"]), 1) if r["gl_rd"] is not None else None,
                ts_mu=round(float(r["ts_mu"]), 1) if r["ts_mu"] is not None else None,
                os_mu=round(float(r["os_mu"]), 1) if r["os_mu"] is not None else None,
                pl_mu=round(float(r["pl_mu"]), 1) if r["pl_mu"] is not None else None,
                tm_mu=round(float(r["tm_mu"]), 1) if r["tm_mu"] is not None else None,
            )
            for r in rows
        ]

    # Downsample uniformly, always preserving first, last, and local peaks
    step = len(rows) / float(max_points)
    sampled_indices = {0, len(rows) - 1}
    for i in range(1, max_points - 1):
        sampled_indices.add(int(i * step))

    # Also include peak Elo and peak Gl points
    max_elo_idx = max(range(len(rows)), key=lambda i: float(rows[i]["elo"]))
    max_gl_idx = max(range(len(rows)), key=lambda i: float(rows[i]["gl"]))
    sampled_indices.add(max_elo_idx)
    sampled_indices.add(max_gl_idx)

    sorted_indices = sorted(sampled_indices)
    return [
        RatingTimelinePoint(
            date=str(rows[i]["date"]),
            match_id=str(rows[i]["match_id"]) if rows[i]["match_id"] else None,
            team_name=str(rows[i]["team_name"]) if rows[i]["team_name"] else None,
            games_count=int(rows[i]["games_count"]),
            elo=round(float(rows[i]["elo"]), 1),
            gl=round(float(rows[i]["gl"]), 1),
            gl_rd=round(float(rows[i]["gl_rd"]), 1) if rows[i]["gl_rd"] is not None else None,
            ts_mu=round(float(rows[i]["ts_mu"]), 1) if rows[i]["ts_mu"] is not None else None,
            os_mu=round(float(rows[i]["os_mu"]), 1) if rows[i]["os_mu"] is not None else None,
            pl_mu=round(float(rows[i]["pl_mu"]), 1) if rows[i]["pl_mu"] is not None else None,
            tm_mu=round(float(rows[i]["tm_mu"]), 1) if rows[i]["tm_mu"] is not None else None,
        )
        for i in sorted_indices
    ]


def compare_players(
    db: Session,
    player_a_id: str,
    player_b_id: str,
    selected_system: str = "unified",
) -> PlayerComparisonResponse | None:
    """Produce a full comparison between two players."""
    profile_a = get_player_profile(db, player_a_id)
    profile_b = get_player_profile(db, player_b_id)

    if not profile_a or not profile_b:
        return None

    h2h = get_head_to_head_summary(db, player_a_id, player_b_id)
    verdict = calculate_model_verdict(profile_a, profile_b, h2h, selected_system)
    timeline_a = get_player_rating_history(player_a_id)
    timeline_b = get_player_rating_history(player_b_id)

    available_systems = list(SYSTEM_LABELS.keys())

    return PlayerComparisonResponse(
        player_a=profile_a,
        player_b=profile_b,
        verdict=verdict,
        h2h=h2h,
        timeline_a=timeline_a,
        timeline_b=timeline_b,
        available_rating_systems=available_systems,
    )


# ── Internal Helpers ─────────────────────────────────────────────────────────


def _get_active_ratings_version(db: Session) -> str | None:
    run = query_one(
        db,
        """
        SELECT ratings_version
        FROM rating_runs
        WHERE status = 'completed'
          AND ratings_version = :operational_version
        LIMIT 1
        """,
        {"operational_version": OPERATIONAL_RATINGS_VERSION},
    )
    if run and run.get("ratings_version"):
        return str(run["ratings_version"])

    v_row = query_one(
        db,
        """
        SELECT ratings_version, COUNT(*) AS count
        FROM entity_ratings
        WHERE entity_type = 'player'
        GROUP BY ratings_version
        ORDER BY count DESC
        LIMIT 1
        """,
    )
    if v_row and v_row.get("ratings_version"):
        return str(v_row["ratings_version"])

    run_fallback = query_one(
        db,
        """
        SELECT ratings_version
        FROM rating_runs
        WHERE status = 'completed'
        ORDER BY games_processed DESC, id DESC
        LIMIT 1
        """,
    )
    return str(run_fallback["ratings_version"]) if run_fallback else None


def _get_player_peaks(player_id: str) -> tuple[float | None, str | None, float | None, str | None]:
    if not HISTORY_DB_PATH.exists():
        return None, None, None, None
    try:
        conn = sqlite3.connect(f"file:{HISTORY_DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT elo, date FROM player_rating_history
            WHERE player_id = ?
            ORDER BY elo DESC, date DESC LIMIT 1
            """,
            (player_id,),
        )
        elo_row = cur.fetchone()

        cur.execute(
            """
            SELECT gl, date FROM player_rating_history
            WHERE player_id = ?
            ORDER BY gl DESC, date DESC LIMIT 1
            """,
            (player_id,),
        )
        gl_row = cur.fetchone()
        conn.close()

        peak_elo = round(float(elo_row[0]), 1) if elo_row else None
        peak_elo_date = str(elo_row[1]) if elo_row else None
        peak_gl = round(float(gl_row[0]), 1) if gl_row else None
        peak_gl_date = str(gl_row[1]) if gl_row else None
        return peak_elo, peak_elo_date, peak_gl, peak_gl_date
    except Exception:
        return None, None, None, None


def _generate_verdict_summary_pl(
    profile_a: PlayerProfileDetail,
    profile_b: PlayerProfileDetail,
    better: str,
    overall_prob_a: float,
    favor_a: int,
    favor_b: int,
    total_sys: int,
    h2h: H2HSummary,
    advantages: Sequence[SystemAdvantage],
) -> str:
    name_a = profile_a.player_name
    name_b = profile_b.player_name

    parts: list[str] = []

    if better == "a":
        p_pct = f"{overall_prob_a * 100:.1f}%"
        parts.append(
            f"Na bazie zestawienia modeli ratingowych za lepszego gracza uznawany jest **{name_a}**, "
            f"uzyskując przewagę w **{favor_a} z {total_sys}** analizowanych systemów (ogólne szanse wygranej w starciu: {p_pct})."
        )
    elif better == "b":
        p_pct = f"{(1.0 - overall_prob_a) * 100:.1f}%"
        parts.append(
            f"Na bazie zestawienia modeli ratingowych za lepszego gracza uznawany jest **{name_b}**, "
            f"uzyskując przewagę w **{favor_b} z {total_sys}** analizowanych systemów (ogólne szanse wygranej w starciu: {p_pct})."
        )
    else:
        parts.append(
            f"Modele wskazują na **zbliżony poziom umiejętności** pomiędzy **{name_a}** i **{name_b}** "
            f"(podział systemów: {favor_a} vs {favor_b}, szanse w starciu ~50-50%)."
        )

    # Highlight specific systems
    elo_adv = next((adv for adv in advantages if adv.system == "elo"), None)
    gl_adv = next((adv for adv in advantages if adv.system == "gl"), None)
    if elo_adv and gl_adv:
        parts.append(
            f"W rankingu Elo bilans wynosi {elo_adv.value_a:.0f} vs {elo_adv.value_b:.0f} (różnica {abs(elo_adv.difference):.0f} pkt), "
            f"a w Glicko-2 {gl_adv.value_a:.0f} vs {gl_adv.value_b:.0f}."
        )

    # H2H note
    if h2h.total_games > 0:
        parts.append(
            f"W bezpośrednich pojedynkach spotkali się dotąd **{h2h.total_games} razy**, "
            f"gdzie bilans wynosi **{h2h.wins_a} zwycięstw {name_a}** do **{h2h.wins_b} zwycięstw {name_b}**."
        )
    else:
        parts.append("Gracze nie mierzyli się dotąd w bezpośrednim oficjalnym meczu na scenie profesjonalnej.")

    # Peak rating context
    if profile_a.peak_elo and profile_b.peak_elo:
        parts.append(
            f"Historyczny szczyt formy (Peak Elo): {name_a} osiągnął **{profile_a.peak_elo:.0f}** ({profile_a.peak_elo_date or '—'}), "
            f"podczas gdy {name_b} osiągnął **{profile_b.peak_elo:.0f}** ({profile_b.peak_elo_date or '—'})."
        )

    return " ".join(parts)
