"""Build upcoming features and predictions for the configured C0 model.

Legacy aggregate feature construction remains for frozen research models only;
the active native model consumes a separate, explicitly identified request.
"""

from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import logging
import math
import re
from urllib.parse import quote
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from openskill.models import PlackettLuce, ThurstoneMostellerFull
from trueskill import TrueSkill

from betting_app.core.db import get_session, query_df, transaction
from betting_app.core.ev import expected_value, fair_market_probabilities
from betting_app.core.matching import normalize_team_name
from betting_app.services.current_roster_service import (
    clean_player_name, load_current_roster, lock_current_roster_teams, upsert_current_roster,
)
from betting_app.core.staking import fractional_kelly_stake
from betting_app.scrapers.lineup_scraper import (
    ROLE_ORDER as LINEUP_ROLE_ORDER,
    check_confirmed_lineup,
    detect_lineup_substitution,
    normalize_player_id,
)
from betting_app.services.bet_qualification_service import (
    is_bet_eligible,
    DEFAULT_BASE_MIN_EV,
    DEFAULT_WASSERSTEIN_EPSILON,
    qualification_tier,
    model_requires_uncertainty,
    prediction_safety_diagnostics,
)
from betting_app.services.canonical_match_service import (
    align_snapshot_odds,
    canonical_team_key,
    parse_iso,
)
from betting_app.services.mapping_service import (
    golgg_name_from_id,
    native_golgg_team_id,
    suggest_mapping,
)
from betting_app.core.models import (
    HybridSpec,
    ModelSpec,
    PredictionEngine,
    UnifiedPredictionResult,
    get_active_hybrid,
    get_active_model,
    get_model,
)
from betting_app.core.models.engine import apply_temperature_scaling
from betting_app.services.prematch_quote_service import (
    load_prematch_quotes, serialize_market_context, prediction_time_is_actionable,
    stored_market_context_is_current, latest_predictions, parse_aware_utc,
    hybrid_prediction_contract_is_current,
)
from betting_app.core.models.registry import get_hybrid_spec
from src.models.competition_tiers import (
    CompetitionScope,
    CompetitionTier,
    classify_competition,
)
from src.ratings.competition_adjustment import (
    CompetitionAdjustment,
    NEUTRAL_COMPETITION_ADJUSTMENT,
    adjust_probability,
)


DEFAULT_FEATURE_VERSION = get_active_hybrid().base_model.feature_version
DEFAULT_RATINGS_VERSION = get_active_hybrid().base_model.ratings_version
DEFAULT_W20_VERSION = get_active_hybrid().base_model.w20_version
DEFAULT_MODEL_NAME = get_active_hybrid().base_model.name
DEFAULT_MODEL_VERSION = get_active_hybrid().base_model.version
DEFAULT_HYBRID_MODEL_NAME = get_active_hybrid().hybrid_model_name
DEFAULT_HYBRID_ALPHA = get_active_hybrid().alpha
DEFAULT_HYBRID_TEMPERATURE = get_active_hybrid().temperature
RATING_SYSTEMS = ("elo", "gl", "ts", "os", "pl", "tm")
W20_FIELDS = (
    "win_rate",
    "avg_kills",
    "avg_deaths",
    "avg_gd15",
    "avg_dpm",
    "avg_vspm",
    "avg_gold",
    "avg_towers",
    "avg_dragons",
    "avg_nashors",
    "avg_game_duration",
)
_TRUESKILL_PROBABILITY_MODEL = TrueSkill(
    draw_probability=0.0,
    beta=4.16,
    tau=0.25,
    mu=25.0,
    sigma=8.333,
)
_OPENSKILL_PROBABILITY_MODEL = PlackettLuce(
    mu=25.0,
    sigma=3.5,
    beta=25.0 / 6.0,
    tau=25.0 / 300.0,
    balance=False,
    limit_sigma=False,
)
_PLACKETT_LUCE_PROBABILITY_MODEL = PlackettLuce(
    mu=25.0, sigma=8.333, beta=18.75, tau=0.05
)
_THURSTONE_PROBABILITY_MODEL = ThurstoneMostellerFull(
    mu=25.0, sigma=8.333, beta=18.75, tau=0.05
)
logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def logit(probability: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, probability))
    return math.log(p / (1.0 - p))


def apply_temperature_probability(probability: float, temperature: float) -> float:
    """Apply binary temperature scaling to one probability."""

    return apply_temperature_scaling(probability, temperature)


def build_all_upcoming_features(
    *,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    ratings_version: str = DEFAULT_RATINGS_VERSION,
    w20_version: str = DEFAULT_W20_VERSION,
    min_mapping_confidence: float = 0.72,
    include_past: bool = False,
    limit: int | None = None,
    is_lineup_confirmed: bool = False,
) -> list[dict[str, Any]]:
    """Build and upsert feature vectors for canonical upcoming matches."""

    matches = load_canonical_matches(include_past=include_past, limit=limit)
    results: list[dict[str, Any]] = []
    for match in matches:
        result = build_features_for_match(
            dict(match),
            feature_version=feature_version,
            ratings_version=ratings_version,
            w20_version=w20_version,
            min_mapping_confidence=min_mapping_confidence,
            is_lineup_confirmed=is_lineup_confirmed,
        )
        results.append(result)
    return results


def _select_aligned_golgg_ids(
    canonical_match: dict[str, Any],
    offers: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    """Return internal GOL.GG team row IDs in canonical A/B order.

    ``upcoming_matches`` keeps the bookmaker's raw team order. Selecting the
    newest row directly therefore swaps rosters whenever a bookmaker lists
    the same event B-vs-A. Only use offers whose teams match the canonical
    identity, and swap their IDs when their raw order is reversed.
    """

    league = str(canonical_match.get("league") or "")
    match_date = canonical_match.get("start_time_normalized")
    canonical_a = canonical_team_key(
        str(canonical_match.get("team_a_name") or ""), league=league, match_date=match_date
    )
    canonical_b = canonical_team_key(
        str(canonical_match.get("team_b_name") or ""), league=league, match_date=match_date
    )
    for offer in offers:
        offer_league = str(offer.get("league") or league)
        raw_a = canonical_team_key(
            str(offer.get("raw_team_a") or ""), league=offer_league, match_date=match_date
        )
        raw_b = canonical_team_key(
            str(offer.get("raw_team_b") or ""), league=offer_league, match_date=match_date
        )
        golgg_a, golgg_b = offer.get("team_a_golgg_id"), offer.get("team_b_golgg_id")
        if golgg_a is None or golgg_b is None:
            continue
        if raw_a == canonical_a and raw_b == canonical_b:
            return int(golgg_a), int(golgg_b)
        if raw_a == canonical_b and raw_b == canonical_a:
            return int(golgg_b), int(golgg_a)
    return None, None




def resolve_exact_canonical_team_ids(match: dict[str, Any]) -> dict[str, Any]:
    """Attach canonical-side-aligned DB and provider IDs from exact sources."""
    resolved = dict(match)
    match_id = resolved.get("id") or resolved.get("canonical_match_id")
    if int(match_id or 0) != 0 and (
        resolved.get("team_a_golgg_id") is None or resolved.get("team_b_golgg_id") is None
    ):
        offers = query_df(
            """
            SELECT raw_team_a, raw_team_b, league, team_a_golgg_id, team_b_golgg_id
            FROM upcoming_matches
            WHERE canonical_match_id = ?
            ORDER BY last_seen_at DESC, id DESC
            """,
            (int(match_id),),
        ).to_dict("records")
        team_a_id, team_b_id = _select_aligned_golgg_ids(resolved, offers)
        if resolved.get("team_a_golgg_id") is None:
            resolved["team_a_golgg_id"] = team_a_id
        if resolved.get("team_b_golgg_id") is None:
            resolved["team_b_golgg_id"] = team_b_id
    if resolved.get("native_team_a_id") is None:
        resolved["native_team_a_id"] = native_golgg_team_id(
            resolved.get("team_a_golgg_id"), team_name=resolved.get("team_a_name"),
        )
    if resolved.get("native_team_b_id") is None:
        resolved["native_team_b_id"] = native_golgg_team_id(
            resolved.get("team_b_golgg_id"), team_name=resolved.get("team_b_name"),
        )
    return resolved
def load_canonical_matches(*, include_past: bool = False, limit: int | None = None):
    """Load canonical matches with internal team row IDs aligned to A/B sides."""

    where = "WHERE cm.status = 'upcoming'"
    params: list[Any] = []
    if not include_past:
        where += " AND (cm.start_time_normalized IS NULL OR cm.start_time_normalized >= ?)"
        params.append(datetime.now(UTC).replace(microsecond=0).isoformat())
    sql = f"""
        SELECT cm.*,
               COUNT(DISTINCT os.bookmaker_id) AS bookmaker_count,
               MAX(os.scraped_at) AS last_scraped_at
        FROM canonical_matches cm
        JOIN odds_snapshots os ON os.canonical_match_id = cm.id
        {where}
        GROUP BY cm.id
        ORDER BY cm.start_time_normalized ASC, cm.id ASC
    """
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    with transaction() as connection:
        matches = [dict(row) for row in connection.execute(sql, tuple(params)).fetchall()]
        if not matches:
            return []
        match_ids = [int(match["id"]) for match in matches]
        placeholders = ", ".join("?" for _ in match_ids)
        rows = connection.execute(
            f"""
            SELECT canonical_match_id, raw_team_a, raw_team_b, league,
                   team_a_golgg_id, team_b_golgg_id, last_seen_at
            FROM upcoming_matches
            WHERE canonical_match_id IN ({placeholders})
            ORDER BY canonical_match_id ASC, last_seen_at DESC, id DESC
            """,
            tuple(match_ids),
        ).fetchall()

    offers_by_match: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        offer = dict(row)
        offers_by_match.setdefault(int(offer["canonical_match_id"]), []).append(offer)
    for match in matches:
        team_a_id, team_b_id = _select_aligned_golgg_ids(
            match, offers_by_match.get(int(match["id"]), [])
        )
        match["team_a_golgg_id"] = team_a_id
        match["team_b_golgg_id"] = team_b_id
    return matches


def _normalize_roster_override(override: Any, default_team_name: str) -> dict[str, Any] | None:
    if not override or not isinstance(override, dict):
        return None
    if "players" in override and isinstance(override["players"], list):
        roster = deepcopy(override)
        roster["basis"] = "operator_override"
        return roster
    # Dict of {role: player_name}
    players = [{"role": str(r).upper(), "player_name": str(p)} for r, p in override.items()]
    return {"team_name": default_team_name, "players": players, "basis": "operator_override"}


def _aware_iso(value: Any, *, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} is required for native C0 inference")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _no_organization_target_id(team_name: Any) -> str:
    """Create a reversible, namespaced ID for a non-organization target."""
    return "no_organization:target:" + quote(str(team_name or "").strip(), safe="")


def _native_c0_request(
    match: dict[str, Any],
    roster_a: dict[str, Any] | None,
    roster_b: dict[str, Any] | None,
    *,
    mode: str = "full",
) -> tuple[dict[str, Any], list[str]]:
    """Build native C0 inputs without leaking local database team-row IDs."""
    missing: list[str] = []
    team_a_id = match.get("native_team_a_id")
    team_b_id = match.get("native_team_b_id")
    if team_a_id is None and match.get("team_a_golgg_id") is not None:
        team_a_id = native_golgg_team_id(
            match["team_a_golgg_id"], team_name=match.get("team_a_name"),
        )
    if team_b_id is None and match.get("team_b_golgg_id") is not None:
        team_b_id = native_golgg_team_id(
            match["team_b_golgg_id"], team_name=match.get("team_b_name"),
        )
    team_a_id = team_a_id if team_a_id is not None else (roster_a or {}).get("team_id")
    team_b_id = team_b_id if team_b_id is not None else (roster_b or {}).get("team_id")
    no_organization = (
        (team_a_id is None or team_b_id is None)
        and roster_a is not None and roster_b is not None
    )
    if no_organization:
        team_a_id = _no_organization_target_id(match.get("team_a_name"))
        team_b_id = _no_organization_target_id(match.get("team_b_name"))
        mode = "no_organization"
    if team_a_id is None:
        missing.append("c0_team_a_id")
    if team_b_id is None:
        missing.append("c0_team_b_id")

    def player_ids(roster: dict[str, Any] | None, side: str) -> list[str]:
        players = (roster or {}).get("players") or []
        ids = [
            str(player.get("player_id")).strip()
            for player in players
            if isinstance(player, dict) and player.get("player_id") not in (None, "")
        ]
        if len(ids) != 5 or len(set(ids)) != 5:
            missing.append(f"c0_{side}_roster_requires_five_exact_ids")
            return []
        return ids

    roster_a_ids = player_ids(roster_a, "team_a")
    roster_b_ids = player_ids(roster_b, "team_b")
    try:
        decision_at = _aware_iso(
            match.get("_c0_decision_at") or datetime.now(UTC),
            label="decision_at",
        )
    except (TypeError, ValueError) as error:
        missing.append(f"c0_time:{error}")
        decision_at = datetime.now(UTC).isoformat()
    try:
        start_at = _aware_iso(
            match.get("start_time_normalized") or match.get("start_at") or match.get("start_time"),
            label="start_at",
        )
        if datetime.fromisoformat(start_at) <= datetime.fromisoformat(decision_at):
            missing.append("c0_time:start_at_must_follow_decision_at")
    except (TypeError, ValueError) as error:
        missing.append(f"c0_time:{error}")
        start_at = None
    best_of = _normalized_best_of(match.get("best_of"))
    if best_of not in {1, 3, 5}:
        missing.append(f"c0_unsupported_best_of:{best_of}")
    context = match.get("league") or match.get("tournament_name")
    request = {
        "team1_id": str(team_a_id) if team_a_id is not None else "",
        "team2_id": str(team_b_id) if team_b_id is not None else "",
        "roster_a": roster_a_ids,
        "roster_b": roster_b_ids,
        "best_of": best_of,
        "decision_at": decision_at,
        "competition_context": str(context).strip() if context else None,
        "start_at": start_at,
        "mode": mode,
    }
    return request, missing


def _load_native_roster_overrides(
    match_id: int,
    *,
    session: Any = None,
    lock: bool = False,
) -> dict[str, dict[str, Any]]:
    rows = query_df(
        """
        SELECT team_side, roster_json, updated_at FROM match_roster_overrides
        WHERE canonical_match_id = ?
        ORDER BY team_side
        """ + (" FOR SHARE" if lock else ""),
        (match_id,),
        session=session,
    )
    overrides = {}
    for row in rows.to_dict("records"):
        roster = row["roster_json"]
        if isinstance(roster, str):
            roster = json.loads(roster)
        if not isinstance(roster, dict):
            raise ValueError("Stored operator roster must be a JSON object.")
        roster = deepcopy(roster)
        updated_at = row["updated_at"]
        roster.update(
            basis="operator_override",
            source=roster.get("source") or "manual",
            binding="match_override",
            updated_at=updated_at.isoformat() if isinstance(updated_at, datetime) else updated_at,
        )
        overrides[str(row["team_side"])] = roster
    return overrides


def _native_roster_provenance(
    roster: dict[str, Any] | None,
    tensor_order: list[str],
) -> dict[str, Any]:
    roster = roster or {}
    player_ids = [
        str(player["player_id"]).strip()
        for player in roster.get("players", [])
        if player.get("player_id") not in (None, "")
    ]
    return {
        "basis": roster.get("basis"),
        "binding": roster.get("binding"),
        "team_id": roster.get("team_id"),
        "team_name": roster.get("team_name"),
        "player_ids": player_ids,
        "tensor_player_order": list(tensor_order),
        "source": roster.get("source"),
        "source_match_id": roster.get("source_match_id"),
        "source_game_id": roster.get("source_game_id"),
        "source_match_date": roster.get("source_match_date"),
        "updated_at": roster.get("updated_at"),
        "source_players": deepcopy(roster.get("players", [])),
        "authenticated": False,
        "published_at": None,
        "observed_at": None,
    }


def validate_native_serving_inputs(
    features: dict[str, Any],
    *,
    session: Any = None,
    lock: bool = False,
) -> None:
    """Withhold changed current inputs; explicit hypothetical inference stays immutable.

    Persistence callers hold the canonical row and normalized team-state locks
    until their prediction commit. These locks cover absent current rows too;
    override mutations take the same canonical row lock.
    """
    if session is None:
        with get_session() as db:
            validate_native_serving_inputs(features, session=db, lock=lock)
        return
    request = features.get("c0_request")
    canonical = features.get("canonical") or {}
    match_id = features.get("canonical_match_id") or canonical.get("id")
    if not isinstance(request, dict) or not match_id:
        raise ValueError("Native serving requires a bound canonical request.")
    current = session.execute(
        text("SELECT * FROM canonical_matches WHERE id=:id" + (" FOR UPDATE" if lock else "")),
        {"id": match_id},
    ).mappings().first()
    if current is None or str(current["status"] or "").casefold() != "upcoming":
        raise ValueError("Native serving match changed or was retracted.")
    start_at = parse_aware_utc(current["start_time_normalized"])
    if start_at is None or start_at <= datetime.now(UTC):
        raise ValueError("Native serving match has no future aware start time.")
    if start_at != parse_aware_utc(request.get("start_at")):
        raise ValueError("Native serving start changed after feature construction.")
    if _normalized_best_of(current["best_of"]) != request.get("best_of"):
        raise ValueError("Native serving best_of changed after feature construction.")
    for side in ("a", "b"):
        field = f"team_{side}_name"
        if canonical.get(field) != current[field]:
            raise ValueError("Native serving team orientation changed after feature construction.")
    provenance = features.get("roster_provenance")
    if not isinstance(provenance, dict):
        # An older explicit request has no assertion that it used current rows.
        # Do not reinterpret it as a stored-roster snapshot.
        return
    context = str(current["league"]).strip() if current["league"] else None
    if context != request.get("competition_context"):
        raise ValueError("Native serving competition context changed after feature construction.")
    overrides = _load_native_roster_overrides(int(match_id), session=session, lock=lock)
    if lock:
        lock_current_roster_teams(session, [
            bound.get("team_name")
            for side in ("a", "b")
            if (bound := provenance.get(f"team_{side}") or {}).get("binding") != "match_override"
            and bound.get("basis") != "operator_override"
        ])
    for side in ("a", "b"):
        bound = provenance.get(f"team_{side}") or {}
        if bound.get("binding") == "match_override":
            roster = overrides.get(side)
        elif bound.get("basis") == "operator_override":
            # A request-local scenario is immutable, not a claim about current rows.
            if side in overrides:
                raise ValueError(f"Native serving operator roster {side} changed after feature construction.")
            continue
        else:
            if side in overrides:
                raise ValueError(f"Native serving roster {side} replaced by an operator override.")
            roster = load_current_roster(bound.get("team_name"), session=session, lock=lock)
            if bound.get("basis") == "historical_last_match":
                if roster is None:
                    roster = load_last_roster(bound.get("team_name"), session=session)
        if roster is None:
            raise ValueError(f"Native serving roster {side} changed or was retracted.")
        observed = _native_roster_provenance(roster, sorted(
            str(player["player_id"]).strip() for player in roster.get("players", [])
        ))
        if observed != bound or observed["tensor_player_order"] != sorted(request.get(f"roster_{side}") or []):
            raise ValueError(f"Native serving roster {side} changed after feature construction.")


def _native_c0_active(model_spec: ModelSpec | None = None) -> bool:
    spec = model_spec or get_active_model()
    return spec.family == "native_c0" or (
        spec.family == "bayesian_market_hybrid"
        and get_hybrid_spec(spec).base_model.family == "native_c0"
    )


def _prediction_cutoff(
    diagnostics: dict[str, Any],
    feature_cutoff: Any,
    *,
    native_c0: bool,
) -> Any:
    if native_c0:
        return diagnostics.get("data_cutoff_at")
    return diagnostics.get("market_cutoff_at") or feature_cutoff


def build_features_for_match(
    match: dict[str, Any],
    *,
    feature_version: str,
    ratings_version: str,
    w20_version: str,
    min_mapping_confidence: float = 0.50,
    team_a_roster_override: dict[str, Any] | None = None,
    team_b_roster_override: dict[str, Any] | None = None,
    persist: bool = True,
    is_lineup_confirmed: bool = False,
    native_c0: bool | None = None,
) -> dict[str, Any]:
    """Build one canonical match feature vector and upsert it."""

    canonical_match_id = int(match["id"])
    team_a_raw = str(match.get("team_a_name") or "")
    team_b_raw = str(match.get("team_b_name") or "")
    native_active = _native_c0_active() if native_c0 is None else native_c0

    # Resolve GOL.GG team names: prefer IDs from upcoming_matches (no fuzzy matching)
    golgg_a_id = match.get("team_a_golgg_id")
    golgg_b_id = match.get("team_b_golgg_id")
    team_a_golgg = golgg_name_from_id(golgg_a_id)
    team_b_golgg = golgg_name_from_id(golgg_b_id)
    mapping_from_ids = team_a_golgg is not None and team_b_golgg is not None

    if not mapping_from_ids:
        team_a_golgg, conf_a, source_a = suggest_mapping(
            team_a_raw,
            source_system="bookmaker",
            league=str(match.get("league") or ""),
            match_date=str(match.get("start_time_normalized") or ""),
        )
        team_b_golgg, conf_b, source_b = suggest_mapping(
            team_b_raw,
            source_system="bookmaker",
            league=str(match.get("league") or ""),
            match_date=str(match.get("start_time_normalized") or ""),
        )
    else:
        conf_a = conf_b = 1.0
        source_a = source_b = "golgg_id"

    missing: list[str] = []
    if not team_a_golgg or conf_a < min_mapping_confidence:
        missing.append(f"team_a_mapping:{team_a_raw}:{conf_a:.3f}")
    if not team_b_golgg or conf_b < min_mapping_confidence:
        missing.append(f"team_b_mapping:{team_b_raw}:{conf_b:.3f}")

    ratings_a = load_team_ratings(team_a_golgg, ratings_version) if team_a_golgg else {}
    ratings_b = load_team_ratings(team_b_golgg, ratings_version) if team_b_golgg else {}
    for system in RATING_SYSTEMS:
        if system not in ratings_a:
            missing.append(f"team_a_rating:{system}")
        if system not in ratings_b:
            missing.append(f"team_b_rating:{system}")

    w20_a = load_w20(team_a_golgg, w20_version) if team_a_golgg else None
    w20_b = load_w20(team_b_golgg, w20_version) if team_b_golgg else None
    if not w20_a:
        missing.append("team_a_w20")
    if not w20_b:
        missing.append("team_b_w20")

    try:
        rating_probs = rating_probabilities(ratings_a, ratings_b)
    except ValueError as error:
        missing.append(str(error))
        rating_probs = {}
    stored_overrides = _load_native_roster_overrides(canonical_match_id) if native_active else {}
    roster_a = _normalize_roster_override(
        team_a_roster_override or stored_overrides.get("a"), team_a_golgg or team_a_raw,
    )
    if "a" in stored_overrides and roster_a is not None and roster_a.get("players") == stored_overrides["a"].get("players"):
        roster_a = deepcopy(stored_overrides["a"])
    if not roster_a and (team_a_golgg or native_active):
        roster_a = load_last_roster(team_a_golgg or team_a_raw)
    roster_b = _normalize_roster_override(
        team_b_roster_override or stored_overrides.get("b"), team_b_golgg or team_b_raw,
    )
    if "b" in stored_overrides and roster_b is not None and roster_b.get("players") == stored_overrides["b"].get("players"):
        roster_b = deepcopy(stored_overrides["b"])
    if not roster_b and (team_b_golgg or native_active):
        roster_b = load_last_roster(team_b_golgg or team_b_raw)
    if not roster_a or len(roster_a.get("players", [])) < 5:
        missing.append("team_a_last_roster")
    if not roster_b or len(roster_b.get("players", [])) < 5:
        missing.append("team_b_last_roster")
    match_league = match.get("league") or match.get("tournament_name")
    match_comp = classify_competition(match_league) if match_league else None
    match_tier = match_comp.tier if match_comp else None
    player_ratings_a = (
        load_roster_player_ratings(roster_a, ratings_version, competition_tier=match_tier)
        if roster_a
        else {}
    )
    player_ratings_b = (
        load_roster_player_ratings(roster_b, ratings_version, competition_tier=match_tier)
        if roster_b
        else {}
    )
    for system in RATING_SYSTEMS:
        if system not in player_ratings_a:
            missing.append(f"team_a_player_rating:{system}")
        if system not in player_ratings_b:
            missing.append(f"team_b_player_rating:{system}")
    try:
        player_probs = player_rating_probabilities(player_ratings_a, player_ratings_b)
    except ValueError as error:
        missing.append(str(error))
        player_probs = {}
    c0_request, c0_missing = _native_c0_request(match, roster_a, roster_b)
    if native_active:
        # Aggregate ratings/W20 remain display/research diagnostics only.
        missing = list(c0_missing)
    lineup_confirmed_flag = is_lineup_confirmed or bool(match.get("is_lineup_confirmed", False))
    features = {
        "c0_request": c0_request,
        "canonical_match_id": canonical_match_id,
        "canonical": {
            "id": canonical_match_id,
            "team_a_name": team_a_raw,
            "team_b_name": team_b_raw,
            "start_time_normalized": match.get("start_time_normalized"),
            "best_of": match.get("best_of"),
            "league": match.get("league"),
            "bookmaker_count": match.get("bookmaker_count"),
            "is_lineup_confirmed": lineup_confirmed_flag,
        },
        "mapping": {
            "team_a_golgg_name": team_a_golgg,
            "team_b_golgg_name": team_b_golgg,
            "team_a_confidence": conf_a,
            "team_b_confidence": conf_b,
            "team_a_source": source_a,
            "team_b_source": source_b,
        },
        "ratings": {"team_a": ratings_a, "team_b": ratings_b, "probabilities": rating_probs},
        "player_ratings": {
            "team_a_roster": roster_a,
            "team_b_roster": roster_b,
            "team_a": player_ratings_a,
            "team_b": player_ratings_b,
            "probabilities": player_probs,
            "roster_source": "confirmed_lineup" if lineup_confirmed_flag else "current_team_roster",
            "is_lineup_confirmed": lineup_confirmed_flag,
        },
        "w20": {"team_a": w20_a, "team_b": w20_b},
        "diagnostics": {
            "missing": missing,
            "c0_missing": c0_missing,
            "missing_player_roster": not roster_a or not roster_b,
            "is_lineup_confirmed": lineup_confirmed_flag,
            "c0_input_mode": c0_request["mode"],
            "c0_organization_identity_missing": c0_request["mode"] == "no_organization",
            "note": "Native C0 consumes exact IDs and its bounded temporal history; aggregate ratings and W20 are not C0 inputs.",
        },
    }
    if native_active:
        features["roster_provenance"] = {
            "decision_at": c0_request["decision_at"],
            "start_at": c0_request["start_at"],
            "team_a": _native_roster_provenance(roster_a, sorted(c0_request["roster_a"])),
            "team_b": _native_roster_provenance(roster_b, sorted(c0_request["roster_b"])),
        }
    if not native_active:
        try:
            _exp078_snapshot_from_features(features)
        except ValueError as error:
            missing.append(str(error))
    status = "ready_player" if not missing else "partial"
    data_cutoff_at = None if native_active else latest_data_cutoff(ratings_version, w20_version)
    if persist:
        if not native_active:
            data_cutoff_at = latest_data_cutoff(ratings_version, w20_version)
        upsert_upcoming_features(
            canonical_match_id=canonical_match_id,
            feature_version=feature_version,
            ratings_version=ratings_version,
            data_cutoff_at=data_cutoff_at,
            team_a_golgg_name=team_a_golgg,
            team_b_golgg_name=team_b_golgg,
            feature_status=status,
            missing_reason=";".join(missing) if missing else None,
            features=features,
        )
    return {"canonical_match_id": canonical_match_id, "status": status, "missing": missing, "features": features}


def load_team_ratings(team_name: str | None, ratings_version: str) -> dict[str, dict[str, Any]]:
    if not team_name:
        return {}
    frame = query_df(
        """
        SELECT rating_system, rating_value, rd, sigma, games_played, last_match_at, state_json
        FROM entity_ratings
        WHERE ratings_version = ? AND entity_type = 'team' AND normalized_entity_name = ?
        """,
        (ratings_version, normalize_team_name(team_name)),
    )
    result: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        system = str(row["rating_system"])
        result[system] = {
            "rating_value": none_or_float(row.get("rating_value")),
            "rd": none_or_float(row.get("rd")),
            "sigma": none_or_float(row.get("sigma")),
            "games_played": int(row.get("games_played") or 0),
            "last_match_at": row.get("last_match_at"),
        }
    return result


def load_player_ratings(player_names: list[str], ratings_version: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Load ratings for a list of players.

    Returns: { player_name: { system_name: { rating_value, rd, sigma, ... } } }
    """
    if not player_names:
        return {}

    cleaned_names = [clean_player_name(p) for p in player_names if clean_player_name(p)]
    if not cleaned_names:
        return {}

    norm_names = list(dict.fromkeys(normalize_team_name(p) for p in cleaned_names))
    lower_names = list(dict.fromkeys(p.lower() for p in cleaned_names))
    squashed_names = list(dict.fromkeys(re.sub(r"[^a-zA-Z0-9]", "", p).lower() for p in cleaned_names if p))

    norm_ph = ", ".join(["?"] * len(norm_names))
    lower_ph = ", ".join(["?"] * len(lower_names))
    squashed_clause = ""
    extra_params: list[str] = []
    if squashed_names:
        squashed_ph = ", ".join(["?"] * len(squashed_names))
        squashed_clause = f"OR REPLACE(LOWER(entity_name), ' ', '') IN ({squashed_ph})"
        extra_params = squashed_names

    frame = query_df(
        f"""
        SELECT normalized_entity_name, entity_name, rating_system, rating_value, rd, sigma, games_played, last_match_at
        FROM entity_ratings
        WHERE ratings_version = ? AND entity_type = 'player'
          AND (
              normalized_entity_name IN ({norm_ph})
              OR LOWER(entity_name) IN ({lower_ph})
              {squashed_clause}
          )
        """,
        (ratings_version, *norm_names, *lower_names, *extra_params),
    )

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        norm_name = str(row["normalized_entity_name"])
        ent_name = str(row.get("entity_name") or "")
        clean_ent = clean_player_name(ent_name)
        squashed_ent = re.sub(r"[^a-zA-Z0-9]", "", clean_ent).lower()
        system = str(row["rating_system"])
        data = {
            "rating_value": none_or_float(row.get("rating_value")),
            "rd": none_or_float(row.get("rd")),
            "sigma": none_or_float(row.get("sigma")),
            "games_played": int(row.get("games_played") or 0),
            "last_match_at": row.get("last_match_at"),
            "entity_name": ent_name,
            "normalized_entity_name": norm_name,
        }
        for k in (norm_name, ent_name, clean_ent, clean_ent.lower(), squashed_ent):
            if k:
                if k not in result:
                    result[k] = {}
                result[k][system] = data
    return result


def load_w20(team_name: str | None, feature_version: str, window_size: int = 20) -> dict[str, Any] | None:
    if not team_name:
        return None
    frame = query_df(
        """
        SELECT *
        FROM team_rolling_features
        WHERE feature_version = ? AND normalized_team_name = ? AND window_size = ?
        LIMIT 1
        """,
        (feature_version, normalize_team_name(team_name), window_size),
    )
    if frame.empty:
        return None
    row = frame.iloc[0].to_dict()
    result = {
        "team_name": row.get("team_name"),
        "matches_count": int(row.get("matches_count") or 0),
        "games_count": int(row.get("games_count") or 0),
        "data_cutoff_at": row.get("data_cutoff_at"),
    }
    for field in W20_FIELDS:
        result[field] = none_or_float(row.get(field))
    try:
        extra = json.loads(row.get("features_json") or "{}")
        result["last_match_at"] = extra.get("last_match_at")
        result["team_id"] = extra.get("team_id")
    except json.JSONDecodeError:
        pass
    return result


def load_last_roster(team_name: str | None, *, session: Any = None) -> dict[str, Any] | None:
    """Return the best known current roster, then GOL.GG last-match fallback.

    A durable roster is populated after completed GOL.GG games and may be
    manually confirmed before an announced lineup first appears in GOL.GG.
    It discloses this as stored-current or historical-last-match input, never an
    authenticated prior-available announcement.
    """

    if not team_name:
        return None
    current = load_current_roster(team_name, session=session)
    if current is not None:
        return current

    matches = query_df(
        """
        SELECT gm.match_id, gm.date, gm.tournament_name
        FROM golgg_matches gm
        JOIN golgg_game_players gp ON gp.match_id = gm.match_id
        WHERE gp.team_name = ?
        ORDER BY gm.date DESC, gm.match_id DESC
        LIMIT 1
        """,
        (team_name,),
        session=session,
    )
    if matches.empty:
        # Fallback for exact-name drift: scan candidate team names and compare with Python normalization.
        candidates = query_df(
            """
            SELECT gp.team_name, gm.match_id, gm.date, gm.tournament_name
            FROM golgg_game_players gp
            JOIN golgg_matches gm ON gm.match_id = gp.match_id
            WHERE gp.team_name IS NOT NULL
            ORDER BY gm.date DESC, CAST(gm.match_id AS INTEGER) DESC
            LIMIT 20000
            """,
            session=session,
        )
        wanted = normalize_team_name(team_name)
        candidates = candidates[candidates["team_name"].map(lambda value: normalize_team_name(str(value)) == wanted)]
        if candidates.empty:
            return None
        matches = candidates.head(1)
    match = matches.iloc[0].to_dict()
    first_game = query_df(
        """
        SELECT game_id
        FROM golgg_game_players
        WHERE match_id = ? AND team_name = ?
        GROUP BY game_id
        ORDER BY CAST(game_id AS INTEGER) ASC, game_id ASC
        LIMIT 1
        """,
        (str(match["match_id"]), team_name),
        session=session,
    )
    if first_game.empty:
        # If fallback matched a historical spelling, reuse that exact spelling.
        exact_team = str(candidates.iloc[0]["team_name"]) if "candidates" in locals() and not candidates.empty else team_name
        first_game = query_df(
            """
            SELECT game_id
            FROM golgg_game_players
            WHERE match_id = ? AND team_name = ?
            GROUP BY game_id
            ORDER BY CAST(game_id AS INTEGER) ASC, game_id ASC
            LIMIT 1
            """,
            (str(match["match_id"]), exact_team),
            session=session,
        )
        team_name = exact_team
    if first_game.empty:
        return None
    players = query_df(
        """
        SELECT team_id, player_id, player_name, role, team_name
        FROM golgg_game_players
        WHERE match_id = ? AND game_id = ? AND team_name = ?
        ORDER BY CASE role
            WHEN 'TOP' THEN 1 WHEN 'JUNGLE' THEN 2 WHEN 'MID' THEN 3
            WHEN 'ADC' THEN 4 WHEN 'SUPPORT' THEN 5 ELSE 9 END, role
        """,
        (str(match["match_id"]), str(first_game.iloc[0]["game_id"]), team_name),
        session=session,
    )
    source_rows = players.to_dict("records")
    source_team_ids = {
        str(row.get("team_id")).strip()
        for row in source_rows if row.get("team_id") not in (None, "")
    }
    roster_players = [
        {
            "player_id": str(row.get("player_id") or ""),
            "player_name": row.get("player_name"),
            "role": row.get("role"),
        }
        for row in source_rows
        if row.get("player_id")
    ]
    return {
        "basis": "historical_last_match",
        "source": "golgg_last_match",
        "updated_at": None,
        "team_name": team_name,
        "team_id": next(iter(source_team_ids)) if len(source_team_ids) == 1 else None,
        "source_match_id": str(match["match_id"]),
        "source_match_date": match.get("date"),
        "source_tournament": match.get("tournament_name"),
        "source_game_id": str(first_game.iloc[0]["game_id"]),
        "players": roster_players,
    }


_REGIONAL_ENGINE_CACHE: dict[str, Any] = {}


def resolve_team_affiliation(
    rating: dict[str, Any] | None,
    tournament_name: str | None = None,
) -> tuple[str, str] | None:
    """Resolve (family, tier) affiliation from state_json or tournament name."""
    if isinstance(rating, dict):
        state_json = rating.get("state_json")
        if state_json:
            try:
                state = json.loads(str(state_json))
                if isinstance(state, dict):
                    family = state.get("family")
                    tier = state.get("tier")
                    if isinstance(family, str) and isinstance(tier, str):
                        return family, tier
            except Exception:
                pass

    if tournament_name:
        identity = classify_competition(tournament_name)
        if (
            identity.scope != CompetitionScope.CROSS_LEAGUE
            and identity.tier != CompetitionTier.UNKNOWN
        ):
            return identity.family, identity.tier.value

    return None


def load_regional_adjustment(
    ratings_a: dict[str, Any],
    ratings_b: dict[str, Any],
    ratings_version: str = DEFAULT_RATINGS_VERSION,
    tournament_a: str | None = None,
    tournament_b: str | None = None,
) -> CompetitionAdjustment:
    """Load the one persisted regional posterior for a matchup."""
    left = ratings_a.get("gl") if isinstance(ratings_a, dict) else None
    right = ratings_b.get("gl") if isinstance(ratings_b, dict) else None
    affiliation_a = resolve_team_affiliation(left, tournament_a)
    affiliation_b = resolve_team_affiliation(right, tournament_b)
    if affiliation_a is None or affiliation_b is None:
        return NEUTRAL_COMPETITION_ADJUSTMENT

    engine = _REGIONAL_ENGINE_CACHE.get(ratings_version)
    if engine is None:
        runs = query_df(
            """
            SELECT systems_json
            FROM rating_runs
            WHERE ratings_version = ? AND status = 'completed'
            ORDER BY finished_at DESC, id DESC
            LIMIT 1
            """,
            (ratings_version,),
        )
        if runs.empty:
            return NEUTRAL_COMPETITION_ADJUSTMENT
        try:
            payload = json.loads(str(runs.iloc[0]["systems_json"]))
            regional = payload["gl"]
            from src.ratings.family_calibrated_glicko2 import FamilyCalibratedGlicko2

            engine = FamilyCalibratedGlicko2.from_state(regional["state"])
            _REGIONAL_ENGINE_CACHE[ratings_version] = engine
        except Exception:
            return NEUTRAL_COMPETITION_ADJUSTMENT

    location = engine.get_location_difference(
        affiliation_a[0],
        affiliation_a[1],
        affiliation_b[0],
        affiliation_b[1],
    )
    return CompetitionAdjustment(
        mean=float(location.mean), variance=float(location.variance)
    )


def regional_adjustment_state(adjustment: CompetitionAdjustment) -> dict[str, Any]:
    """Return serializable representation of a regional adjustment."""
    return {
        "engine": "family-calibrated-glicko2-v1",
        "mean": adjustment.mean,
        "variance": adjustment.variance,
        "applies_to": list(RATING_SYSTEMS),
    }


def get_player_major_game_counts(player_ids: Sequence[str]) -> dict[str, int]:
    """Return count of major tournament games played by each player."""
    if not player_ids:
        return {}
    clean_ids = [str(p) for p in player_ids if p]
    if not clean_ids:
        return {}
    placeholders = ",".join("?" for _ in clean_ids)
    try:
        df = query_df(
            f"""
            SELECT gpm.player_id, count(distinct gpm.game_id) as major_games
            FROM golgg_game_players gpm
            JOIN golgg_matches gm ON gpm.match_id = gm.match_id
            WHERE gpm.player_id IN ({placeholders})
              AND (
                  gm.tournament_name LIKE '%LEC%'
                  OR gm.tournament_name LIKE '%LCK%'
                  OR gm.tournament_name LIKE '%LPL%'
                  OR gm.tournament_name LIKE '%LCS%'
                  OR gm.tournament_name LIKE '%Worlds%'
                  OR gm.tournament_name LIKE '%Mid-Season Invitational%'
                  OR gm.tournament_name LIKE '%MSI%'
              )
            GROUP BY gpm.player_id
            """,
            tuple(clean_ids),
        )
        result = {pid: 0 for pid in clean_ids}
        if not df.empty and "player_id" in df.columns:
            for _, row in df.iterrows():
                result[str(row["player_id"])] = int(row["major_games"])
        return result
    except Exception:
        return {pid: 0 for pid in clean_ids}


def load_roster_player_ratings(
    roster: dict[str, Any] | None,
    ratings_version: str,
    competition_tier: CompetitionTier | None = None,
) -> dict[str, dict[str, Any]]:
    """Load aggregate player ratings for a roster by rating system, applying
    rookie tier calibration if the upcoming match is in a major competition."""
    if not roster:
        return {}
    roster_players = {
        str(player["player_id"]): player
        for player in roster.get("players", [])
        if player.get("player_id")
    }
    if not roster_players:
        return {}
    roster_players_by_identity: dict[str, dict[str, Any]] = {}
    for player in roster_players.values():
        for identity in (player.get("player_id"), player.get("player_name")):
            if identity:
                cleaned_id = clean_player_name(identity)
                roster_players_by_identity.setdefault(
                    normalize_team_name(str(identity)),
                    player,
                )
                roster_players_by_identity.setdefault(
                    cleaned_id.lower(),
                    player,
                )
                squashed = re.sub(r"[^a-zA-Z0-9]", "", cleaned_id).lower()
                if squashed:
                    roster_players_by_identity.setdefault(squashed, player)

    player_ids = list(roster_players)
    player_names = list(
        dict.fromkeys(
            str(identity).casefold()
            for player in roster_players.values()
            for identity in (player.get("player_id"), player.get("player_name"))
            if identity
        )
    )
    squashed_names = list(
        dict.fromkeys(
            re.sub(r"[^a-zA-Z0-9]", "", str(identity)).casefold()
            for player in roster_players.values()
            for identity in (player.get("player_id"), player.get("player_name"))
            if identity and re.sub(r"[^a-zA-Z0-9]", "", str(identity))
        )
    )
    id_placeholders = ",".join("?" for _ in player_ids)
    name_placeholders = ",".join("?" for _ in player_names)
    squashed_clause = ""
    extra_params: list[str] = []
    if squashed_names:
        squashed_ph = ",".join("?" for _ in squashed_names)
        squashed_clause = f"OR REPLACE(LOWER(entity_name), ' ', '') IN ({squashed_ph})"
        extra_params = squashed_names

    frame = query_df(
        f"""
        SELECT rating_system, entity_name, normalized_entity_name, rating_value, rd, sigma, games_played, last_match_at
        FROM entity_ratings
        WHERE ratings_version = ? AND entity_type = 'player'
          AND (
              normalized_entity_name IN ({id_placeholders})
              OR LOWER(entity_name) IN ({name_placeholders})
              {squashed_clause}
          )
        """,
        (ratings_version, *player_ids, *player_names, *extra_params),
    )
    if frame.empty or "rating_system" not in frame.columns:
        return {}
    is_major = competition_tier in (
        CompetitionTier.MAJOR,
        CompetitionTier.INTERNATIONAL,
    )
    major_counts = get_player_major_game_counts(player_ids) if is_major else {}

    result: dict[str, dict[str, Any]] = {}
    for system, group in frame.groupby("rating_system"):
        matched_players: dict[str, dict[str, Any]] = {}
        for player in group.to_dict("records"):
            norm_entity = str(player.get("normalized_entity_name") or "")
            ent_name = str(player.get("entity_name") or "")
            clean_ent = clean_player_name(ent_name)
            squashed_ent = re.sub(r"[^a-zA-Z0-9]", "", clean_ent).lower()

            roster_player = roster_players.get(norm_entity)
            if roster_player is None:
                roster_player = roster_players_by_identity.get(normalize_team_name(norm_entity))
            if roster_player is None:
                roster_player = roster_players_by_identity.get(normalize_team_name(ent_name))
            if roster_player is None:
                roster_player = roster_players_by_identity.get(clean_ent.lower())
            if roster_player is None and squashed_ent:
                roster_player = roster_players_by_identity.get(squashed_ent)
            if roster_player is None:
                continue
            player_id = str(roster_player["player_id"])
            existing = matched_players.get(player_id)
            if existing is None or int(player.get("games_played") or 0) > int(
                existing.get("games_played") or 0
            ):
                rec = {
                    **player,
                    "player_id": player_id,
                    "player_name": clean_player_name(roster_player.get("player_name") or player.get("entity_name")),
                    "role": roster_player.get("role"),
                }
                if is_major and major_counts.get(player_id, 0) < 20:
                    if system == "elo":
                        val = none_or_float(rec.get("rating_value"))
                        if val is not None and val > 1750.0:
                            rec["rating_value"] = 1750.0
                    elif system in ("ts", "os"):
                        val = none_or_float(rec.get("rating_value"))
                        if val is not None and val > 28.0:
                            rec["rating_value"] = 28.0
                        sig = none_or_float(rec.get("sigma"))
                        if sig is None or sig < 3.83:
                            rec["sigma"] = 3.83
                    elif system == "pl":
                        val = none_or_float(rec.get("rating_value"))
                        if val is not None and val > 33.0:
                            rec["rating_value"] = 33.0
                        sig = none_or_float(rec.get("sigma"))
                        if sig is None or sig < 5.0:
                            rec["sigma"] = 5.0
                    elif system == "tm":
                        val = none_or_float(rec.get("rating_value"))
                        if val is not None and val > 31.0:
                            rec["rating_value"] = 31.0
                        sig = none_or_float(rec.get("sigma"))
                        if sig is None or sig < 3.95:
                            rec["sigma"] = 3.95
                    elif system == "gl":
                        rd = none_or_float(rec.get("rd"))
                        if rd is None or rd < 150.0:
                            rec["rd"] = 150.0
                matched_players[player_id] = rec
        player_records = list(matched_players.values())
        ratings = [none_or_float(player.get("rating_value")) for player in player_records]
        ratings = [value for value in ratings if value is not None]
        if not ratings:
            continue
        result[str(system)] = {
            "avg_rating_value": sum(ratings) / len(ratings),
            "min_rating_value": min(ratings),
            "max_rating_value": max(ratings),
            "players_with_rating": len(ratings),
            "expected_players": len(player_ids),
            "players": player_records,
        }
    return result

def _state_probability(
    system: str,
    left_states: list[dict[str, Any]],
    right_states: list[dict[str, Any]],
) -> float:
    left_ratings = [
        _required_float(state.get("rating_value"), f"{system} rating")
        for state in left_states
    ]
    right_ratings = [
        _required_float(state.get("rating_value"), f"{system} rating")
        for state in right_states
    ]
    if system == "elo":
        left = sum(left_ratings) / len(left_ratings)
        right = sum(right_ratings) / len(right_ratings)
        return 1.0 / (1.0 + 10 ** ((right - left) / 400.0))
    if system == "gl":
        left_rd = math.sqrt(
            sum(
                _required_float(state.get("rd"), "Glicko RD") ** 2
                for state in left_states
            )
            / len(left_states)
        )
        right_rd = math.sqrt(
            sum(
                _required_float(state.get("rd"), "Glicko RD") ** 2
                for state in right_states
            )
            / len(right_states)
        )
        left = sum(left_ratings) / len(left_ratings)
        right = sum(right_ratings) / len(right_ratings)
        if len(left_states) > 1 or len(right_states) > 1:
            left, right, left_rd, right_rd = map(int, (left, right, left_rd, right_rd))
        combined_rd = math.sqrt(left_rd**2 + right_rd**2)
        q = math.log(10.0) / 400.0
        g_factor = 1.0 / math.sqrt(
            1.0 + 3.0 * q**2 * combined_rd**2 / math.pi**2
        )
        return 1.0 / (1.0 + 10 ** (-g_factor * (left - right) / 400.0))
    if system == "ts":
        delta_mu = sum(left_ratings) - sum(right_ratings)
        variance = sum(
            _required_float(state.get("sigma"), "TrueSkill sigma") ** 2
            for state in (*left_states, *right_states)
        )
        denominator = math.sqrt(
            variance + (len(left_states) + len(right_states)) * 4.16**2
        )
        return float(_TRUESKILL_PROBABILITY_MODEL.cdf(delta_mu / denominator))

    model = {
        "os": _OPENSKILL_PROBABILITY_MODEL,
        "pl": _PLACKETT_LUCE_PROBABILITY_MODEL,
        "tm": _THURSTONE_PROBABILITY_MODEL,
    }[system]
    left = [
        model.rating(
            mu=rating,
            sigma=_required_float(state.get("sigma"), f"{system} sigma"),
        )
        for rating, state in zip(left_ratings, left_states, strict=True)
    ]
    right = [
        model.rating(
            mu=rating,
            sigma=_required_float(state.get("sigma"), f"{system} sigma"),
        )
        for rating, state in zip(right_ratings, right_states, strict=True)
    ]
    return float(model.predict_win([left, right])[0])


def _rating_probability(system: str, left: float, right: float) -> float:
    diff = left - right
    if system in {"elo", "gl"}:
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))
    if system == "os":
        return sigmoid(diff / 5.0)
    return sigmoid(diff / 8.333)


def _with_regional_adjustment(
    system: str,
    probability: float,
    adjustment: CompetitionAdjustment,
) -> float:
    if system == "gl":
        return probability
    return adjust_probability(probability, adjustment)


def rating_probabilities(
    ratings_a: dict[str, Any],
    ratings_b: dict[str, Any],
    adjustment: CompetitionAdjustment = NEUTRAL_COMPETITION_ADJUSTMENT,
) -> dict[str, float]:
    probs: dict[str, float] = {}
    for system in RATING_SYSTEMS:
        left = ratings_a.get(system)
        right = ratings_b.get(system)
        if left is None or right is None:
            continue
        val_left = left.get("rating_value") if isinstance(left, dict) else left
        val_right = right.get("rating_value") if isinstance(right, dict) else right
        if val_left is None or val_right is None:
            continue
        prob = _rating_probability(system, float(val_left), float(val_right))
        probs[system] = _with_regional_adjustment(system, prob, adjustment)
    if probs:
        probs["consensus"] = sum(probs.values()) / len(probs)
    return probs


def player_rating_probabilities(
    ratings_a: dict[str, Any],
    ratings_b: dict[str, Any],
    adjustment: CompetitionAdjustment = NEUTRAL_COMPETITION_ADJUSTMENT,
) -> dict[str, float]:
    probs: dict[str, float] = {}
    for system in RATING_SYSTEMS:
        left = ratings_a.get(system, {})
        right = ratings_b.get(system, {})
        left_players = left.get("players", []) if isinstance(left, dict) else []
        right_players = right.get("players", []) if isinstance(right, dict) else []
        if (
            left_players
            and right_players
            and all("rating_value" in p for p in (*left_players, *right_players))
            and (system != "gl" or all("rd" in p for p in (*left_players, *right_players)))
            and (system in {"elo", "gl"} or all("sigma" in p for p in (*left_players, *right_players)))
        ):
            raw_prob = _state_probability(system, left_players, right_players)
            probs[system] = _with_regional_adjustment(system, raw_prob, adjustment)
            continue

        val_left = (
            left.get("avg_rating_value")
            if isinstance(left, dict)
            else None
        )
        val_right = (
            right.get("avg_rating_value")
            if isinstance(right, dict)
            else None
        )
        if val_left is None or val_right is None:
            continue
        prob = _rating_probability(system, float(val_left), float(val_right))
        probs[system] = _with_regional_adjustment(system, prob, adjustment)
    if probs:
        probs["consensus"] = sum(probs.values()) / len(probs)
    return probs


def upsert_upcoming_features(*, session: Any = None, **kwargs: Any) -> None:
    if session is None:
        with get_session() as db:
            upsert_upcoming_features(session=db, **kwargs)
            db.commit()
        return
    session.execute(
        text(
            """
            INSERT INTO upcoming_match_features(
                canonical_match_id, feature_version, ratings_version, data_cutoff_at,
                team_a_golgg_name, team_b_golgg_name, feature_status, missing_reason,
                features_json
            ) VALUES (
                :canonical_match_id, :feature_version, :ratings_version, :data_cutoff_at,
                :team_a_golgg_name, :team_b_golgg_name, :feature_status, :missing_reason,
                :features_json
            )
            ON CONFLICT(canonical_match_id, feature_version, ratings_version) DO UPDATE SET
                data_cutoff_at = excluded.data_cutoff_at,
                team_a_golgg_name = excluded.team_a_golgg_name,
                team_b_golgg_name = excluded.team_b_golgg_name,
                feature_status = excluded.feature_status,
                missing_reason = excluded.missing_reason,
                features_json = excluded.features_json
            """
        ),
        {**kwargs, "features_json": json.dumps(kwargs["features"], ensure_ascii=False, sort_keys=True)},
    )


def latest_data_cutoff(ratings_version: str, w20_version: str) -> str | None:
    rating = query_df("SELECT data_cutoff_at FROM rating_runs WHERE ratings_version = ?", (ratings_version,))
    if not rating.empty and rating.iloc[0].get("data_cutoff_at"):
        return str(rating.iloc[0]["data_cutoff_at"])
    w20 = query_df(
        "SELECT MAX(data_cutoff_at) AS data_cutoff_at FROM team_rolling_features WHERE feature_version = ?",
        (w20_version,),
    )
    if not w20.empty and w20.iloc[0].get("data_cutoff_at"):
        return str(w20.iloc[0]["data_cutoff_at"])
    return None


def register_operational_model(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    ratings_version: str = DEFAULT_RATINGS_VERSION,
    runtime_identity: dict[str, Any] | None = None,
) -> int:
    """Register a pure-model contract with its true serving artifact identity."""
    spec = get_model(model_name)
    native_c0 = spec is not None and spec.family == "native_c0"
    if (
        spec is None or spec.artifact_path is None
        or spec.metadata.get("operational") is False
        or (not native_c0 and not spec.artifact_path.is_file())
    ):
        raise ValueError("operational model contract requires an available pure-model artifact")
    expected = (spec.name, spec.version, spec.feature_version, spec.ratings_version)
    actual = (model_name, model_version, feature_version, ratings_version)
    if actual != expected:
        raise ValueError(
            "operational model contract must be "
            f"{expected!r}, got {actual!r}"
        )

    if spec.family == "linear_symmetric":
        from src.models.symmetric_series import SymmetricSeriesModel
        SymmetricSeriesModel.load_default()

    artifact: dict[str, Any] = {}
    artifact_sha256 = None
    if native_c0:
        if runtime_identity is None:
            from betting_app.services.c0_inference import get_c0_artifact_identity

            runtime_identity = get_c0_artifact_identity()
        feature_schema = {
            "feature_version": spec.feature_version,
            "input_contract": "native-c0-request-v1",
            "request_fields": [
                "team1_id", "team2_id", "roster_a", "roster_b", "best_of",
                "decision_at", "competition_context", "start_at", "mode",
            ],
            "market_features_used": False,
            "has_uncertainty": False,
            "side_symmetric": True,
            "artifact_release": spec.metadata.get("artifact_release"),
            "artifact_kind": spec.metadata.get("artifact_kind"),
        }
        arch = {}
        params = {
            "artifact_path": str(spec.artifact_path),
            "artifact_release": spec.metadata.get("artifact_release"),
            "artifact_kind": spec.metadata.get("artifact_kind"),
            "model_identity": runtime_identity,
        }
    else:
        with spec.artifact_path.open("rb") as handle:
            artifact_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            handle.seek(0)
            artifact = json.load(handle) if spec.artifact_path.suffix == ".json" else {}
        arch = artifact.get("architecture", artifact.get("estimator", {}))
        scaler = artifact.get("scaler", artifact.get("estimator", {}))
        feature_schema = {
            "feature_version": spec.feature_version,
            "ratings_version": ratings_version,
            "feature_names": scaler.get("feature_names"),
            "ratings": list(RATING_SYSTEMS),
            "player_ratings": list(RATING_SYSTEMS),
            "regional_projection": {
                "engine": "regional_offset_projection",
                "excluded_system": "gl",
                "applies_to": [s for s in RATING_SYSTEMS if s != "gl"],
            },
            "roster_source": "current team roster, refreshed from GOL.GG or manual confirmation",
            "w20_fields": list(W20_FIELDS),
            "market_features_used": False,
            "side_symmetric": True,
            "limitations": [
                "last-match roster fallback",
                "not confirmed upcoming rosters",
                "epistemic uncertainty gating under turnover tax",
            ],
        }
        params = {
            key: arch[key] for key in (
                "family", "d_in", "d_h1", "d_h2", "loss", "risk_kappa", "n_members", "C"
            ) if key in arch
        }
        params.update(
            artifact_path=str(spec.artifact_path.relative_to(spec.artifact_path.parents[2])),
            artifact_sha256=artifact_sha256,
            artifact_model_name=artifact.get("model_name"),
            artifact_model_version=artifact.get("model_version"),
        )
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO model_artifacts(
                model_name, model_version, feature_schema_json, model_params_json, status
            ) VALUES (?, ?, ?, ?, 'active')
            ON CONFLICT(model_name, model_version) DO UPDATE SET
                feature_schema_json = excluded.feature_schema_json,
                model_params_json = excluded.model_params_json,
                status = 'active'
            """,
            (
                spec.name,
                spec.version,
                json.dumps(feature_schema, ensure_ascii=False, sort_keys=True),
                json.dumps(params, ensure_ascii=False, sort_keys=True),
            ),
        )
        row = connection.execute(
            "SELECT id FROM model_artifacts WHERE model_name = ? AND model_version = ?",
            (spec.name, spec.version),
        ).fetchone()
        return int(row["id"])


def predict_all_upcoming(
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    feature_version: str | None = None,
    ratings_version: str = DEFAULT_RATINGS_VERSION,
    include_partial: bool = False,
) -> list[dict[str, Any]]:
    """Publish each fixture atomically, releasing its locks before the next one."""
    active = get_active_model()
    m_name = model_name or active.name
    m_version = model_version or active.version
    f_version = feature_version or active.feature_version

    if (m_name, m_version, f_version, ratings_version) != (
        active.name, active.version, active.feature_version, active.ratings_version
    ):
        raise ValueError("Prediction identity must match the configured model and feature contract.")
    if active.family == "bayesian_market_hybrid":
        hybrid = get_active_hybrid()
        model_artifact_id = register_hybrid_model(
            spec=hybrid, base_version=hybrid.base_model.version, version=m_version
        )
    else:
        model_artifact_id = register_operational_model(
            model_name=m_name, model_version=m_version, feature_version=f_version,
            ratings_version=ratings_version,
        )
    params: list[Any] = [f_version, ratings_version]
    status_filter = "AND feature_status = 'ready_player'"
    if include_partial:
        status_filter = "AND feature_status IN ('ready_player', 'ready_team', 'partial')"
    frame = query_df(
        f"""
        SELECT umf.*, cm.team_a_name, cm.team_b_name, cm.start_time_normalized, cm.league,
               cm.status
        FROM upcoming_match_features umf
        JOIN canonical_matches cm ON cm.id = umf.canonical_match_id
        WHERE feature_version = ? AND ratings_version = ? {status_filter}
        ORDER BY cm.start_time_normalized ASC
        """,
        tuple(params),
    )
    decision_clock = datetime.now(UTC)
    results: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        match_id = int(row["canonical_match_id"])
        status = str(row.get("status") or "").strip().casefold()
        start = row.get("start_time_normalized")
        try:
            start_at = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            future = start_at.tzinfo is not None and start_at > decision_clock
        except (TypeError, ValueError):
            future = False
        if status != "upcoming" or not future:
            results.append({
                "prediction_id": None,
                "canonical_match_id": match_id,
                "match": f"{row.get('team_a_name')} vs {row.get('team_b_name')}",
                "prediction_status": "skipped",
                "error": "Match is finished or has no future aware start time.",
            })
        else:
            eligible_rows.append(row)
    for row in eligible_rows:
        # The batch owns this transaction: every skip, success or failure exits it
        # before another canonical fixture can be locked.
        with transaction() as connection:
            raw_features = row.get("features_json") or "{}"
            features = deepcopy(raw_features) if isinstance(raw_features, dict) else json.loads(raw_features)
            try:
                if _native_c0_active(active):
                    validate_native_serving_inputs(features, session=connection._session)
                    features = deepcopy(features)
                    features["c0_request"]["decision_at"] = decision_clock.isoformat()
                    if isinstance(features.get("roster_provenance"), dict):
                        features["roster_provenance"]["decision_at"] = decision_clock.isoformat()
                prob_a, diagnostics = predict_probability_from_features(features)
            except ValueError as error:
                results.append(
                    {
                        "prediction_id": None,
                        "canonical_match_id": int(row["canonical_match_id"]),
                        "match": f"{row.get('team_a_name')} vs {row.get('team_b_name')}",
                        "prediction_status": "skipped",
                        "error": str(error),
                    }
                )
                continue
            match_id = int(row["canonical_match_id"])
            persist_at = datetime.now(UTC)
            start_at = parse_aware_utc(row.get("start_time_normalized"))
            if start_at is None or persist_at >= start_at:
                results.append({"canonical_match_id":match_id,"prediction_id":None,"prediction_status":"skipped","error":"Match started before persistence."})
                continue
            if _native_c0_active(active):
                try:
                    validate_native_serving_inputs(features, session=connection._session, lock=True)
                except ValueError as error:
                    results.append({
                        "canonical_match_id": match_id, "prediction_id": None,
                        "prediction_status": "skipped", "error": str(error),
                    })
                    continue
            saved_market = diagnostics.get("market_context")
            if saved_market and saved_market.get("sources"):
                evaluation_at = parse_aware_utc(saved_market.get("decision_at"))
                if evaluation_at is None:
                    continue
                at_evaluation = load_prematch_quotes([match_id],decision_at=evaluation_at,session=connection._session)[match_id]
                current = load_prematch_quotes([match_id],decision_at=persist_at,session=connection._session)[match_id]
                if not stored_market_context_is_current(saved_market,evaluation_context=at_evaluation,current_context=current)[0]:
                    results.append({"canonical_match_id":match_id,"prediction_id":None,"prediction_status":"skipped","error":"Market context stale before persistence."})
                    continue
            connection.execute(
                """
                UPDATE canonical_predictions
                SET prediction_status = 'stale'
                WHERE canonical_match_id = ? AND prediction_status = 'active'
                  AND model_name = ? AND model_version = ?
                """,
                (match_id, m_name, m_version),
            )
            cursor = connection.execute(
                """
                INSERT INTO canonical_predictions(
                    canonical_match_id, model_artifact_id, model_name, model_version, predicted_at,
                    prob_a, prob_b, features_version, ratings_version, data_cutoff_at, diagnostics_json,
                    prediction_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    match_id,
                    model_artifact_id,
                    m_name,
                    m_version,
                    utc_now_iso(),
                    prob_a,
                    1.0 - prob_a,
                    f_version,
                    ratings_version,
                    _prediction_cutoff(
                        diagnostics, row.get("data_cutoff_at"),
                        native_c0=_native_c0_active(active),
                    ),
                    json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                ),
            )
            results.append(
                {
                    "prediction_id": int(cursor.lastrowid),
                    "canonical_match_id": match_id,
                    "match": f"{row.get('team_a_name')} vs {row.get('team_b_name')}",
                    "prob_a": prob_a,
                    "prob_b": 1.0 - prob_a,
                    "diagnostics": diagnostics,
                }
            )
    return results

def _normalized_best_of(best_of: Any) -> int:
    """Validate supported series lengths; an absent format is a single map."""
    try:
        value = 1 if best_of is None else int(best_of)
    except (TypeError, ValueError):
        raise ValueError(f"best_of must be one of 1, 3, 5, 7; got {best_of!r}")
    if value not in {1, 3, 5, 7}:
        raise ValueError(f"best_of must be one of 1, 3, 5, 7; got {best_of!r}")
    return value


def w20_probability(w20_a: dict[str, Any], w20_b: dict[str, Any]) -> float:
    win = (w20_a.get("win_rate") or 0.5) - (w20_b.get("win_rate") or 0.5)
    kills = (w20_a.get("avg_kills") or 12.0) - (w20_b.get("avg_kills") or 12.0)
    deaths = (w20_a.get("avg_deaths") or 12.0) - (w20_b.get("avg_deaths") or 12.0)
    gd15 = (w20_a.get("avg_gd15") or 0.0) - (w20_b.get("avg_gd15") or 0.0)
    dpm = (w20_a.get("avg_dpm") or 1800.0) - (w20_b.get("avg_dpm") or 1800.0)
    towers = (w20_a.get("avg_towers") or 5.0) - (w20_b.get("avg_towers") or 5.0)
    score = 1.25 * win + 0.035 * kills - 0.03 * deaths + 0.00018 * gd15 + 0.00008 * dpm + 0.07 * towers
    return sigmoid(score)

def series_probability(map_probability: float, best_of: Any) -> float:
    """Convert a per-map probability to a BoN series probability using binomial expansion."""
    maps = _normalized_best_of(best_of)
    probability = max(1e-6, min(1.0 - 1e-6, float(map_probability)))
    wins_required = maps // 2 + 1
    return sum(
        math.comb(maps, wins)
        * probability**wins
        * (1.0 - probability) ** (maps - wins)
        for wins in range(wins_required, maps + 1)
    )


def predict_operational_match(
    match: dict[str, Any],
    *,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    ratings_version: str = DEFAULT_RATINGS_VERSION,
    w20_version: str = DEFAULT_W20_VERSION,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    include_partial: bool = False,
    team_a_roster_override: dict[str, Any] | None = None,
    team_b_roster_override: dict[str, Any] | None = None,
    persist: bool = True,
    is_lineup_confirmed: bool = False,
    validate_current_inputs: bool = False,
) -> dict[str, Any]:
    """Build one match feature vector and predict with the exact registered model."""
    model_spec = get_model(model_name)
    if model_spec is None or model_spec.version != model_version:
        raise ValueError("Prediction identity must identify an exact registered model version.")
    if _native_c0_active(model_spec):
        match = resolve_exact_canonical_team_ids(match)
    feature_result = build_features_for_match(
        match,
        feature_version=feature_version,
        ratings_version=ratings_version,
        w20_version=w20_version,
        min_mapping_confidence=0.72,
        team_a_roster_override=team_a_roster_override,
        team_b_roster_override=team_b_roster_override,
        persist=False,
        is_lineup_confirmed=is_lineup_confirmed,
        native_c0=_native_c0_active(model_spec),
    )
    if feature_result["status"] != "ready_player" and not include_partial:
        raise ValueError(
            "operational prediction requires ready player features: "
            + "; ".join(feature_result["missing"])
        )
    features = feature_result["features"]
    if (persist or validate_current_inputs) and _native_c0_active(model_spec):
        validate_native_serving_inputs(features)
    prob_a, diagnostics = predict_probability_from_features(features, model_spec=model_spec)
    if persist:
        with get_session() as db:
            if _native_c0_active(model_spec):
                validate_native_serving_inputs(features, session=db, lock=True)
            upsert_upcoming_features(
                session=db, canonical_match_id=int(match["id"]),
                feature_version=feature_version, ratings_version=ratings_version,
                data_cutoff_at=feature_result.get("data_cutoff_at"),
                team_a_golgg_name=features.get("mapping", {}).get("team_a_golgg_name"),
                team_b_golgg_name=features.get("mapping", {}).get("team_b_golgg_name"),
                feature_status=feature_result["status"],
                missing_reason=";".join(feature_result.get("missing", [])) or None,
                features=features,
            )
            db.commit()
    return {
        "canonical_match_id": match.get("id"),
        "prob_a": prob_a,
        "prob_b": 1.0 - prob_a,
        "model_name": model_spec.name,
        "model_version": model_spec.version,
        "data_cutoff_at": _prediction_cutoff(
            diagnostics, feature_result.get("data_cutoff_at"),
            native_c0=_native_c0_active(model_spec),
        ),
        "diagnostics": diagnostics,
        "features": features,
    }

def _format_roster_five(roster: Sequence[Any], team_name: str) -> list[dict[str, Any]]:
    """Format a 5-player roster sequence into dictionaries with canonical roles."""
    roles = list(LINEUP_ROLE_ORDER)
    formatted: list[dict[str, Any]] = []
    for idx, item in enumerate(roster[:5]):
        if isinstance(item, dict):
            pid = normalize_player_id(item)
            pname = str(item.get("player_name") or item.get("name") or pid)
            role = str(item.get("role") or (roles[idx] if idx < len(roles) else "FLEX")).upper()
        else:
            pid = normalize_player_id(item)
            pname = str(item).strip()
            role = roles[idx] if idx < len(roles) else "FLEX"
        formatted.append({
            "player_id": pid,
            "player_name": pname,
            "role": role,
            "team_name": team_name,
        })
    return formatted


def process_confirmed_lineup_update(
    match: dict[str, Any],
    *,
    confirmed_roster_a: Sequence[Any] | None = None,
    confirmed_roster_b: Sequence[Any] | None = None,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    ratings_version: str = DEFAULT_RATINGS_VERSION,
    w20_version: str = DEFAULT_W20_VERSION,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    current_time: Any = None,
    force_re_infer: bool = False,
    session: Any = None,
) -> dict[str, Any]:
    """Check confirmed lineup, detect substitutions, update starting 5, and force re-inference.

    If substitution is detected against the baseline/previous starting five:
    1. Updates the starting five in `team_current_roster_players` and feature mapping.
    2. Rebuilds match features with `is_lineup_confirmed=True`.
    3. Forces re-inference before bet qualification (marking prior predictions stale).

    If no substitution is detected (same starters):
    1. Updates match features with `is_lineup_confirmed=True`.
    2. Does not force re-inference (re_inference_triggered=False).
    """
    match_id = match.get("id") or match.get("canonical_match_id")
    start_at = match.get("start_time_normalized") or match.get("start_at") or match.get("start_time")

    lineup_check = check_confirmed_lineup(
        match_id=match_id,
        match_start_at=start_at,
        canonical_match=match,
        current_time=current_time,
        session=session,
    )

    conf_a = confirmed_roster_a or lineup_check.get("team_a_roster")
    conf_b = confirmed_roster_b or lineup_check.get("team_b_roster")

    if not lineup_check.get("window_valid", True) and not force_re_infer:
        return {
            "status": lineup_check.get("status", "outside_window"),
            "canonical_match_id": match_id,
            "window_valid": False,
            "is_lineup_confirmed": False,
            "substitution_detected": False,
            "re_inference_triggered": False,
            "substituted_players": {"team_a": [], "team_b": []},
            "substituted_player_ids": [],
            "reason": lineup_check.get("reason", "Outside confirmed lineup window"),
        }

    if not conf_a and not conf_b and not force_re_infer:
        return {
            "status": "unconfirmed",
            "canonical_match_id": match_id,
            "window_valid": True,
            "is_lineup_confirmed": False,
            "substitution_detected": False,
            "re_inference_triggered": False,
            "substituted_players": {"team_a": [], "team_b": []},
            "substituted_player_ids": [],
            "reason": "No confirmed rosters available",
        }

    team_a_name = str(match.get("team_a_name") or match.get("team_a_golgg_name") or "")
    team_b_name = str(match.get("team_b_name") or match.get("team_b_golgg_name") or "")

    prev_a = match.get("previous_roster_a")
    if prev_a is None and team_a_name:
        prev_a = load_last_roster(team_a_name)
    prev_b = match.get("previous_roster_b")
    if prev_b is None and team_b_name:
        prev_b = load_last_roster(team_b_name)

    prev_a_ids = [
        normalize_player_id(p)
        for p in (prev_a.get("players", []) if isinstance(prev_a, dict) else (prev_a or []))
        if normalize_player_id(p)
    ]
    prev_b_ids = [
        normalize_player_id(p)
        for p in (prev_b.get("players", []) if isinstance(prev_b, dict) else (prev_b or []))
        if normalize_player_id(p)
    ]

    has_sub_a, subs_a = detect_lineup_substitution(prev_a_ids, conf_a) if conf_a else (False, [])
    has_sub_b, subs_b = detect_lineup_substitution(prev_b_ids, conf_b) if conf_b else (False, [])
    substitution_detected = has_sub_a or has_sub_b

    model_spec = get_model(model_name)
    native_c0 = model_spec is not None and _native_c0_active(model_spec)
    if native_c0:
        match = resolve_exact_canonical_team_ids(match)
        for side, roster in (("team_a", conf_a), ("team_b", conf_b)):
            if roster is None:
                continue
            if (
                len(roster) != 5
                or any(not isinstance(player, dict) or not player.get("player_id") for player in roster)
                or len({str(player["player_id"]) for player in roster if isinstance(player, dict) and player.get("player_id")}) != 5
            ):
                raise ValueError(f"Native C0 confirmed {side} lineup requires five unique exact player IDs.")
    formatted_a = _format_roster_five(conf_a, team_a_name) if conf_a else None
    formatted_b = _format_roster_five(conf_b, team_b_name) if conf_b else None

    roster_updates = [
        (team_name, formatted_roster)
        for team_name, formatted_roster in ((team_a_name, formatted_a), (team_b_name, formatted_b))
        if team_name and formatted_roster and len(formatted_roster) == 5
    ]
    if roster_updates:
        # One writer/session only; the former raw fallback used a second transaction
        # and could race publication (or block on its own retained roster locks).
        with (get_session() if session is None else nullcontext(session)) as roster_db:
            lock_current_roster_teams(roster_db, [team_name for team_name, _ in roster_updates])
            for team_name, formatted_roster in roster_updates:
                if not upsert_current_roster(
                    roster_db, team_name=team_name, players=formatted_roster,
                    source="confirmed_lineup", force=True,
                ):
                    raise ValueError(f"Confirmed lineup for {team_name} is not a complete role roster.")
            if session is None:
                roster_db.commit()

    # Feature mapping overrides
    team_a_override = {"team_name": team_a_name, "players": formatted_a} if formatted_a else None
    team_b_override = {"team_name": team_b_name, "players": formatted_b} if formatted_b else None

    # Rebuild features with confirmed lineup flag
    feature_result = build_features_for_match(
        match,
        feature_version=feature_version,
        ratings_version=ratings_version,
        w20_version=w20_version,
        team_a_roster_override=team_a_override,
        team_b_roster_override=team_b_override,
        is_lineup_confirmed=True,
        native_c0=native_c0,
        persist=not (substitution_detected or force_re_infer),
    )

    canonical_match_id = int(match_id) if match_id is not None else 0

    if substitution_detected or force_re_infer:
        model_spec = get_model(model_name)
        if model_spec is None or model_spec.version != model_version:
            raise ValueError("Lineup re-inference requires an exact registered model version.")

        features = feature_result["features"]
        if native_c0:
            validate_native_serving_inputs(features)
        prob_a, diagnostics = predict_probability_from_features(features, model_spec=model_spec)
        diagnostics["lineup_substitution_detected"] = True
        diagnostics["substituted_players"] = {"team_a": subs_a, "team_b": subs_b}

        if model_spec.family == "bayesian_market_hybrid":
            hybrid = get_hybrid_spec(model_spec)
            model_artifact_id = register_hybrid_model(
                spec=hybrid, base_version=hybrid.base_model.version, version=model_version
            )
        else:
            model_artifact_id = register_operational_model(
                model_name=model_name, model_version=model_version,
                feature_version=feature_version, ratings_version=ratings_version,
                runtime_identity=diagnostics.get("model_identity"),
            )
        with transaction() as conn:
            if native_c0:
                try:
                    validate_native_serving_inputs(features, session=conn._session, lock=True)
                except ValueError as error:
                    return {
                        "status": "skipped", "prediction_status": "skipped",
                        "canonical_match_id": canonical_match_id, "prediction_id": None,
                        "re_inference_triggered": False, "error": str(error), "reason": str(error),
                    }
            upsert_upcoming_features(
                session=conn._session, canonical_match_id=canonical_match_id,
                feature_version=feature_version, ratings_version=ratings_version,
                data_cutoff_at=feature_result.get("data_cutoff_at"),
                team_a_golgg_name=features.get("mapping", {}).get("team_a_golgg_name"),
                team_b_golgg_name=features.get("mapping", {}).get("team_b_golgg_name"),
                feature_status="partial" if feature_result.get("missing") else "ready_player",
                missing_reason=";".join(feature_result.get("missing", [])) or None,
                features=features,
            )
            conn.execute(
                """
                UPDATE canonical_predictions SET prediction_status = 'stale'
                WHERE canonical_match_id = ? AND model_name = ? AND model_version = ?
                """,
                (canonical_match_id, model_name, model_version),
            )
            cursor = conn.execute(
                """
                INSERT INTO canonical_predictions(
                    canonical_match_id, model_artifact_id, model_name, model_version, predicted_at,
                    prob_a, prob_b, features_version, ratings_version, data_cutoff_at, diagnostics_json,
                    prediction_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active') RETURNING id
                """,
                (
                    canonical_match_id, model_artifact_id, model_name, model_version,
                    utc_now_iso(), prob_a, 1.0 - prob_a, feature_version, ratings_version,
                    _prediction_cutoff(
                        diagnostics, feature_result.get("data_cutoff_at"),
                        native_c0=native_c0,
                    ),
                    json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                ),
            )
            pred_id = int(cursor.fetchone()["id"])

        return {
            "status": "re_inferred",
            "canonical_match_id": canonical_match_id,
            "window_valid": True,
            "is_lineup_confirmed": True,
            "substitution_detected": True,
            "re_inference_triggered": True,
            "substituted_players": {"team_a": subs_a, "team_b": subs_b},
            "substituted_player_ids": subs_a + subs_b,
            "prob_a": prob_a,
            "prob_b": 1.0 - prob_a,
            "prediction_id": pred_id,
            "features": features,
            "diagnostics": diagnostics,
        }
    else:
        # Same starters -> no re-inference
        return {
            "status": "confirmed_no_change",
            "canonical_match_id": canonical_match_id,
            "window_valid": True,
            "is_lineup_confirmed": True,
            "substitution_detected": False,
            "re_inference_triggered": False,
            "substituted_players": {"team_a": [], "team_b": []},
            "substituted_player_ids": [],
            "features": feature_result["features"],
        }


def _required_float(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"EXP-078 requires {label}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"EXP-078 requires finite {label}")
    return number


def _player_metric(group: dict[str, Any], metric: str, label: str) -> float:
    values = [
        _required_float(player.get(metric), label)
        for player in group.get("players", [])
        if player.get(metric) is not None
    ]
    if not values:
        raise ValueError(f"EXP-078 requires {label}")
    return sum(values) / len(values)


def _days_since_last(features: dict[str, Any], side: str) -> float:
    start = parse_iso(features.get("canonical", {}).get("start_time_normalized"))
    if start is None:
        start = datetime.now(UTC)
    dates = [
        parse_iso(str(state.get("last_match_at")))
        for state in features.get("ratings", {}).get(side, {}).values()
        if state.get("last_match_at")
    ]
    valid_dates = [value for value in dates if value is not None]
    if not valid_dates:
        return 7.0
    return max(0.0, (start - max(valid_dates)).total_seconds() / 86_400.0)


def _exp078_snapshot_from_features(features: dict[str, Any]) -> tuple[dict[str, float], int]:
    canonical = features.get("canonical", {})
    best_of = _normalized_best_of(canonical.get("best_of") or features.get("best_of"))
    ratings = features.get("ratings", {})
    team_a = ratings.get("team_a", {})
    team_b = ratings.get("team_b", {})
    team_probabilities = ratings.get("probabilities", {})
    player_ratings = features.get("player_ratings", {})
    player_a = player_ratings.get("team_a", {})
    player_b = player_ratings.get("team_b", {})
    player_probabilities = player_ratings.get("probabilities", {})
    w20 = features.get("w20", {})
    w20_a = w20.get("team_a") or {}
    w20_b = w20.get("team_b") or {}

    snapshot: dict[str, float] = {}
    for system in RATING_SYSTEMS:
        snapshot[f"team_{system}"] = _required_float(
            team_probabilities.get(system), f"team {system} probability"
        )
        snapshot[f"player_{system}"] = _required_float(
            player_probabilities.get(system), f"player {system} probability"
        )
    snapshot.update(
        {
            "team_elo_r1": _required_float(
                team_a.get("elo", {}).get("rating_value"), "team_a Elo rating"
            ),
            "team_elo_r2": _required_float(
                team_b.get("elo", {}).get("rating_value"), "team_b Elo rating"
            ),
            "team_gl_r1": _required_float(
                team_a.get("gl", {}).get("rating_value"), "team_a Glicko rating"
            ),
            "team_gl_r2": _required_float(
                team_b.get("gl", {}).get("rating_value"), "team_b Glicko rating"
            ),
            "team_gl_rd1": _required_float(
                team_a.get("gl", {}).get("rd"), "team_a Glicko RD"
            ),
            "team_gl_rd2": _required_float(
                team_b.get("gl", {}).get("rd"), "team_b Glicko RD"
            ),
            "player_elo_min1": _required_float(
                player_a.get("elo", {}).get("min_rating_value"),
                "team_a minimum player Elo",
            ),
            "player_elo_min2": _required_float(
                player_b.get("elo", {}).get("min_rating_value"),
                "team_b minimum player Elo",
            ),
            "player_gl_max1": _required_float(
                player_a.get("gl", {}).get("max_rating_value"),
                "team_a maximum player Glicko",
            ),
            "player_gl_max2": _required_float(
                player_b.get("gl", {}).get("max_rating_value"),
                "team_b maximum player Glicko",
            ),
            "player_gl_rd_avg1": _player_metric(
                player_a.get("gl", {}), "rd", "team_a player Glicko RD"
            ),
            "player_gl_rd_avg2": _player_metric(
                player_b.get("gl", {}), "rd", "team_b player Glicko RD"
            ),
            "days_since_last_1": _days_since_last(features, "team_a"),
            "days_since_last_2": _days_since_last(features, "team_b"),
        }
    )
    for system in ("ts", "os", "pl", "tm"):
        snapshot[f"team_{system}_mu1"] = _required_float(
            team_a.get(system, {}).get("rating_value"), f"team_a {system} rating"
        )
        snapshot[f"team_{system}_mu2"] = _required_float(
            team_b.get(system, {}).get("rating_value"), f"team_b {system} rating"
        )
        snapshot[f"team_{system}_sigma1"] = _required_float(
            team_a.get(system, {}).get("sigma"), f"team_a {system} sigma"
        )
        snapshot[f"team_{system}_sigma2"] = _required_float(
            team_b.get(system, {}).get("sigma"), f"team_b {system} sigma"
        )
        snapshot[f"player_{system}_sigma_avg1"] = _player_metric(
            player_a.get(system, {}), "sigma", f"team_a player {system} sigma"
        )
        snapshot[f"player_{system}_sigma_avg2"] = _player_metric(
            player_b.get(system, {}), "sigma", f"team_b player {system} sigma"
        )

    w20_names = {
        "win_rate": "win_rate",
        "kills": "avg_kills",
        "deaths": "avg_deaths",
        "gd15": "avg_gd15",
        "dpm": "avg_dpm",
        "vspm": "avg_vspm",
        "towers": "avg_towers",
        "nashors": "avg_nashors",
        "gold": "avg_gold",
        "duration": "avg_game_duration",
    }
    for feature_name, stored_name in w20_names.items():
        snapshot[f"t1_rolling_{feature_name}"] = _required_float(
            w20_a.get(stored_name), f"team_a W20 {stored_name}"
        )
        snapshot[f"t2_rolling_{feature_name}"] = _required_float(
            w20_b.get(stored_name), f"team_b W20 {stored_name}"
        )
    return snapshot, best_of


def fetch_latest_pre_match_market_prob(
    canonical_match_id: int | None,
    normalized_team_a: str | None = None,
    normalized_team_b: str | None = None,
    *,
    as_of: datetime | None = None,
) -> float | None:
    """Return the shared admitted same-book proportional consensus."""
    if canonical_match_id is None:
        return None
    return load_prematch_quotes(
        [canonical_match_id], decision_at=as_of or datetime.now(UTC)
    )[canonical_match_id].prob_a


def evaluate_bayesian_market_hybrid(
    features: dict[str, Any],
    spec: Any,
) -> UnifiedPredictionResult:
    """Blend the declared sports artifact once using the shared fixed-weight policy.

    Alpha is the sports-model weight, as in PredictionEngine.blend_with_market.
    A missing market leaves the sports prediction unchanged; a failed sports
    artifact or invalid input is an error, never an unnamed rating fallback.
    """
    if "market_novig_prob_a" in features or "market_prob_a" in features:
        raise ValueError("Bare market probability overrides are not admitted; timestamped fixture quotes are required.")
    market_cutoff = datetime.now(UTC)
    hybrid = get_hybrid_spec(spec)
    base_spec = hybrid.base_model
    PredictionEngine.blend_with_market(0.5, 0.5, hybrid)
    sports = PredictionEngine.predict_from_features(features, base_spec)
    canonical = features.get("canonical", {})
    canonical_match_id = features.get("canonical_match_id") or canonical.get("id")
    context = (
        load_prematch_quotes([canonical_match_id], decision_at=market_cutoff)[canonical_match_id]
        if canonical_match_id is not None else None
    )
    market = context.prob_a if context is not None else None
    if market is not None:
        sports_cutoff = parse_aware_utc(sports.diagnostics.get("data_cutoff_at"))
        if sports_cutoff is None or sports_cutoff > market_cutoff:
            raise ValueError("Sports input cutoff must be aware and available by market evaluation.")
        market = float(market)
        if not math.isfinite(market) or not 0.0 < market < 1.0:
            raise ValueError("Market probability must be finite and strictly between zero and one.")
        probability = PredictionEngine.blend_with_market(sports.prob_a, market, hybrid)
        low_a = (
            PredictionEngine.blend_with_market(sports.p_low_a, market, hybrid)
            if sports.p_low_a is not None else None
        )
        low_b = (
            PredictionEngine.blend_with_market(sports.p_low_b, 1.0 - market, hybrid)
            if sports.p_low_b is not None else None
        )
    else:
        probability = sports.prob_a
        low_a, low_b = sports.p_low_a, sports.p_low_b
    # Match-winner odds describe a series, not its individual maps.
    base_data_cutoff = sports.diagnostics.get("data_cutoff_at")
    market_cutoff_at = market_cutoff.isoformat() if market_cutoff is not None and market is not None else None
    effective_cutoff = base_data_cutoff
    if market_cutoff_at:
        cutoffs = [value for value in (base_data_cutoff, market_cutoff_at) if value]
        try:
            effective_cutoff = max(
                cutoffs,
                key=lambda value: datetime.fromisoformat(str(value).replace("Z", "+00:00")),
            )
        except (TypeError, ValueError):
            effective_cutoff = market_cutoff_at
    diagnostics = {
        **sports.diagnostics,
        "base_diagnostics": sports.diagnostics,
        "base_model_identity": sports.diagnostics.get("model_identity"),
        "model_identity": sports.diagnostics.get("model_identity"),
        "base_data_cutoff_at": base_data_cutoff,
        "data_cutoff_at": effective_cutoff,
        "feature_provenance": sports.diagnostics.get("feature_provenance"),
        "family": spec.family,
        "base_model_name": sports.model_name,
        "base_model_version": sports.model_version,
        "p_sports": sports.prob_a,
        "p_market_novig": market,
        "alpha": hybrid.alpha,
        "alpha_semantics": "sports_model_weight",
        "temperature": hybrid.temperature,
        "blending_mode": hybrid.blending_mode,
        "market_features_used": market is not None,
        "market_cutoff_at": market_cutoff_at,
        "market_context": serialize_market_context(
            context, model_name=spec.name, model_version=spec.version,
            base_model_name=base_spec.name, base_model_version=base_spec.version,
        ) if context is not None else None,
        "market_unavailable_reason": "market_missing" if market is None else None,
        "side_symmetric": sports.diagnostics.get("side_symmetric", False),
        "uncertainty_required": base_spec.has_uncertainty,
        "epistemic_sigma_z": sports.epistemic_sigma_z,
        "p_low_a": low_a,
        "p_low_b": low_b,
        "map_probability_source": "series_hybrid_bo1" if sports.best_of == 1 else "sports_model",
    }
    map_probability = probability if sports.best_of == 1 else sports.map_prob_a
    return UnifiedPredictionResult(
        canonical_match_id=canonical_match_id,
        prob_a=probability,
        prob_b=1.0 - probability,
        map_prob_a=map_probability,
        map_prob_b=1.0 - map_probability if map_probability is not None else None,
        p_low_a=low_a,
        p_low_b=low_b,
        epistemic_sigma_z=sports.epistemic_sigma_z,
        model_name=spec.name,
        model_version=spec.version,
        feature_version=spec.feature_version,
        best_of=sports.best_of,
        diagnostics=diagnostics,
    )


def predict_probability_from_features(
    features: dict[str, Any],
    *,
    model_spec: ModelSpec | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate model prediction via the unified PredictionEngine."""
    spec = model_spec or get_active_model()
    result = PredictionEngine.predict_from_features(features, spec)
    diag = {
        "model_name": result.model_name,
        "model_version": result.model_version,
        "feature_version": result.feature_version,
        "best_of": result.best_of,
        "market_features_used": False,
        "side_symmetric": True,
        "epistemic_sigma_z": result.epistemic_sigma_z,
        "p_low_a": result.p_low_a,
        "p_low_b": result.p_low_b,
    }
    ratings = {} if spec.family == "native_c0" else features.get("ratings", {}).get("probabilities", {})
    player_ratings = {} if spec.family == "native_c0" else features.get("player_ratings", {}).get("probabilities", {})
    diag.update(result.diagnostics)
    c0_request = features.get("c0_request")
    if isinstance(c0_request, dict):
        diag["c0_input_mode"] = c0_request.get("mode")
        diag["c0_organization_identity_missing"] = c0_request.get("mode") == "no_organization"
    if _native_c0_active(spec) and isinstance(features.get("roster_provenance"), dict):
        provenance = deepcopy(features["roster_provenance"])
        bound_request = diag.get("c0_request") or c0_request
        provenance["decision_at"] = bound_request.get("decision_at")
        provenance["start_at"] = bound_request.get("start_at")
        for side in ("a", "b"):
            provenance[f"team_{side}"]["tensor_player_order"] = list(bound_request[f"roster_{side}"])
        diag["roster_provenance"] = provenance
    if ratings and player_ratings:
        try:
            team_probs = [float(ratings[system]) for system in RATING_SYSTEMS]
            player_probs = [float(player_ratings[system]) for system in RATING_SYSTEMS]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "all six team/player rating probabilities are required"
            ) from exc
        if any(not 0.0 <= value <= 1.0 for value in (*team_probs, *player_probs)):
            raise ValueError("rating probabilities must be finite and in [0, 1]")
        diag["rating_disagreement"] = abs(sum(player_probs) - sum(team_probs)) / len(
            RATING_SYSTEMS
        )
    diag = prediction_safety_diagnostics(
        diag,
        result.prob_a,
        result.prob_b,
        uncertainty_required=model_requires_uncertainty(result.model_name),
    )
    return result.prob_a, diag


def generate_hybrid_predictions(
    *,
    base_model_name: str | None = None,
    base_model_version: str | None = None,
    alpha: float | None = None,
    temperature: float | None = None,
    hybrid_model_name: str | None = None,
    hybrid_model_version: str | None = None,
    blending_mode: str | None = None,
    canonical_match_id: int | None = None,
) -> list[dict[str, Any]]:
    """Blend validated base means and sidewise safety scores with the latest market."""
    active_hybrid = get_active_hybrid()
    base_model_name = base_model_name or active_hybrid.base_model.name
    base_spec = get_model(base_model_name)
    if base_spec is None:
        raise ValueError(f"unknown hybrid base model: {base_model_name}")
    if base_spec.family == "bayesian_market_hybrid" or base_spec.metadata.get("operational") is False:
        raise ValueError("Hybrid generation requires an available pure sports model.")
    base_model_version = base_model_version or base_spec.version
    alpha = active_hybrid.alpha if alpha is None else alpha
    temperature = active_hybrid.temperature if temperature is None else temperature
    hybrid_model_name = hybrid_model_name or active_hybrid.hybrid_model_name
    blending_mode = blending_mode or active_hybrid.blending_mode

    hybrid_spec = HybridSpec(
        base_model=base_spec,
        alpha=alpha,
        hybrid_model_name=hybrid_model_name,
        temperature=temperature,
        blending_mode=blending_mode,
    )
    # Validate configuration before persistence, even when there are no input rows.
    PredictionEngine.blend_with_market(0.5, 0.5, hybrid_spec)
    if hybrid_model_version is None:
        hybrid_model_version = (
            active_hybrid.hybrid_model_version
            if (
                base_model_version == active_hybrid.base_model.version
                and base_spec == active_hybrid.base_model
                and (alpha, temperature, hybrid_model_name, blending_mode) == (
                    active_hybrid.alpha, active_hybrid.temperature,
                    active_hybrid.hybrid_model_name, active_hybrid.blending_mode,
                )
            )
            else f"{base_model_version}-a{alpha:g}-t{temperature:g}-{blending_mode}-v2"
        )
    model_artifact_id = register_hybrid_model(
        spec=hybrid_spec,
        base_version=base_model_version,
        version=hybrid_model_version,
    )
    market_cutoff = datetime.now(UTC)
    rows = query_df(
        """SELECT p.*, p.id AS base_prediction_id, p.prob_a AS model_prob_a
        FROM canonical_predictions p WHERE model_name=? AND model_version=?
        AND prediction_status='active' AND (CAST(? AS INTEGER) IS NULL OR canonical_match_id=?)""",
        (base_model_name,base_model_version,canonical_match_id,canonical_match_id),
    )
    predictions = latest_predictions(rows.to_dict("records"))
    contexts = load_prematch_quotes(
        list(dict.fromkeys(int(r["canonical_match_id"]) for r in predictions)), decision_at=market_cutoff,
    )
    results: list[dict[str, Any]] = []
    with transaction() as connection:
        for first in predictions:
            canonical_match_id = int(first["canonical_match_id"])
            context = contexts[canonical_match_id]
            if context.prob_a is None or not prediction_time_is_actionable(
                first,decision_at=market_cutoff,start_at=context.start_at,
                model_name=base_model_name,model_version=base_model_version,
            )[0]:
                continue
            market_probs = [(1/q.odds_a)/(1/q.odds_a+1/q.odds_b) for q in context.quotes]
            try:
                model_prob = float(first["model_prob_a"])
                base_diagnostics = prediction_safety_diagnostics(
                    first.get("diagnostics_json"),
                    model_prob,
                    1.0 - model_prob,
                    uncertainty_required=model_requires_uncertainty(base_model_name),
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping hybrid prediction %s: %s", canonical_match_id, exc
                )
                continue
            base_raw = first.get("diagnostics_json")
            if isinstance(base_raw, str):
                try:
                    base_raw = json.loads(base_raw)
                except (TypeError, ValueError):
                    base_raw = {}
            if not isinstance(base_raw, dict):
                base_raw = {}
            base_cutoff = base_raw.get("data_cutoff_at") or first.get("data_cutoff_at")
            market_cutoff_at = market_cutoff.isoformat()
            effective_cutoff = market_cutoff_at
            if base_cutoff:
                try:
                    if datetime.fromisoformat(str(base_cutoff).replace("Z", "+00:00")) > market_cutoff:
                        effective_cutoff = str(base_cutoff)
                except (TypeError, ValueError):
                    effective_cutoff = market_cutoff_at
            model_t = apply_temperature_probability(model_prob, temperature)
            market_prob = sum(market_probs) / len(market_probs)
            hybrid_prob = PredictionEngine.blend_with_market(
                model_prob, market_prob, hybrid_spec
            )
            formula_name = f"{blending_mode}: blend(temp(model), average_no_vig_market)"
            diagnostics = {
                "base_model_name": base_model_name,
                "base_model_version": base_model_version,
                "base_prediction_id": int(first["base_prediction_id"]),
                "base_model_identity": base_raw.get("model_identity"),
                "model_identity": base_raw.get("model_identity"),
                "base_data_cutoff_at": base_cutoff,
                "market_context": serialize_market_context(
                    context, model_name=hybrid_model_name, model_version=hybrid_model_version,
                    base_model_name=base_model_name, base_model_version=base_model_version,
                ),
                "data_cutoff_at": effective_cutoff,
                "base_diagnostics": base_raw,
                "feature_provenance": base_raw.get("feature_provenance"),
                "c0_request": base_raw.get("c0_request"),
                "roster_provenance": base_raw.get("roster_provenance"),
                "uncertainty_available": base_raw.get("uncertainty_available"),
                "alpha": alpha,
                "temperature": temperature,
                "blending_mode": blending_mode,
                "market_cutoff_at": market_cutoff_at,
                "alpha_semantics": "sports_model_weight",
                "model_prob_a": model_prob,
                "model_prob_a_temperature": model_t,
                "market_prob_a_avg_no_vig": market_prob,
                "bookmakers_used": len(market_probs),
                "formula": formula_name,
                "uncertainty_required": base_diagnostics["uncertainty_required"],
                "uncertainty_basis": base_diagnostics["uncertainty_basis"],
                "uncertainty_reason": base_diagnostics["uncertainty_reason"],
                "rating_disagreement": base_diagnostics.get("rating_disagreement"),
            }
            if base_diagnostics["uncertainty_required"] and all(
                base_diagnostics.get(key) is not None for key in ("p_low_a", "p_low_b")
            ):
                diagnostics.update(
                    p_low_a=PredictionEngine.blend_with_market(
                        base_diagnostics["p_low_a"],
                        market_prob,
                        hybrid_spec,
                    ),
                    p_low_b=min(
                        1.0 - hybrid_prob,
                        PredictionEngine.blend_with_market(
                            base_diagnostics["p_low_b"],
                            1.0 - market_prob,
                            hybrid_spec,
                        ),
                    ),
                    epistemic_sigma_z=base_diagnostics["epistemic_sigma_z"],
                    uncertainty_semantics="sidewise transformed conservative scores; no coverage guarantee",
                )
            persist_at = datetime.now(UTC)
            current = load_prematch_quotes([canonical_match_id],decision_at=persist_at,session=connection._session)[canonical_match_id]
            if not stored_market_context_is_current(
                diagnostics["market_context"],evaluation_context=context,current_context=current,
            )[0]:
                continue
            connection.execute(
                """
                UPDATE canonical_predictions
                SET prediction_status = 'stale'
                WHERE canonical_match_id = ? AND prediction_status = 'active'
                  AND model_name = ? AND model_version = ?
                """,
                (int(canonical_match_id), hybrid_model_name, hybrid_model_version),
            )
            cursor = connection.execute(
                """
                INSERT INTO canonical_predictions(
                    canonical_match_id, model_artifact_id, model_name, model_version, predicted_at,
                    prob_a, prob_b, features_version, ratings_version, data_cutoff_at, diagnostics_json,
                    prediction_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active') RETURNING id
                """,
                (
                    int(canonical_match_id),
                    model_artifact_id,
                    hybrid_model_name,
                    hybrid_model_version,
                    utc_now_iso(),
                    hybrid_prob,
                    1.0 - hybrid_prob,
                    first.get("features_version"),
                    first.get("ratings_version"),
                    effective_cutoff,
                    json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                ),
            )
            results.append(
                {
                    "prediction_id": int(cursor.fetchone()["id"]),
                    "canonical_match_id": int(canonical_match_id),
                    "prob_a": hybrid_prob,
                    "prob_b": 1.0 - hybrid_prob,
                    "diagnostics": diagnostics,
                }
            )
    return results


def register_hybrid_model(*, spec: HybridSpec, base_version: str, version: str) -> int:
    feature_schema = {
        "base_model": f"{spec.base_model.name}/{base_version}",
        "market_signal": "average no-vig probability from latest bookmaker odds",
        "blending_mode": spec.blending_mode,
        "uncertainty": "sidewise transformed scores; no statistical coverage guarantee",
    }
    params = {
        "base_model_name": spec.base_model.name,
        "base_model_version": base_version,
        "alpha": spec.alpha,
        "temperature": spec.temperature,
        "blending_mode": spec.blending_mode,
    }
    with transaction() as connection:
        existing = connection.execute(
            "SELECT * FROM model_artifacts WHERE model_name=? AND model_version=?",
            (spec.hybrid_model_name,version),
        ).fetchone()
        if existing:
            old = json.loads(existing["model_params_json"] or "{}")
            schema = json.loads(existing["feature_schema_json"] or "{}")
            declared = schema.get("base_model")
            if not old.get("base_model_name") and declared == f"{spec.base_model.name}/{base_version}":
                old.update(base_model_name=spec.base_model.name,base_model_version=base_version)
            if any(old.get(k) != v for k,v in params.items()):
                raise ValueError("Conflicting parameters for existing exact hybrid identity")
            connection.execute(
                "UPDATE model_artifacts SET model_params_json=? WHERE id=?",
                (json.dumps(params,ensure_ascii=False,sort_keys=True),int(existing["id"])),
            )
        connection.execute(
            """
            INSERT INTO model_artifacts(
                model_name, model_version, feature_schema_json, model_params_json, status
            ) VALUES (?, ?, ?, ?, 'active')
            ON CONFLICT(model_name, model_version) DO NOTHING
            """,
            (
                spec.hybrid_model_name,
                version,
                json.dumps(feature_schema, ensure_ascii=False, sort_keys=True),
                json.dumps(params, ensure_ascii=False, sort_keys=True),
            ),
        )
        row = connection.execute(
            "SELECT id FROM model_artifacts WHERE model_name = ? AND model_version = ?",
            (spec.hybrid_model_name, version),
        ).fetchone()
        return int(row["id"])


def generate_model_ev_signals(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    tax_rate: float = 0.12,
    min_ev: float = DEFAULT_BASE_MIN_EV,
    wasserstein_epsilon: float | None = DEFAULT_WASSERSTEIN_EPSILON,
    bankroll: float = 100.0,
    reserved_bankroll: float | None = None,
    check_confirmed_lineups: bool = False,
    lineup_current_time: Any = None,
    return_stats: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate EV rows for latest predictions and latest odds per bookmaker.

    Calculates multi-bookmaker consensus fair market probabilities, identifies
    bookmaker market edges/outliers, and sizes stakes taking into account currently
    open bets and portfolio risk limits.
    """
    if check_confirmed_lineups:
        upcoming_matches = load_canonical_matches(include_past=False)
        for u_match in upcoming_matches:
            process_confirmed_lineup_update(
                dict(u_match),
                current_time=lineup_current_time,
                model_name=model_name,
                model_version=model_version,
            )
    if not math.isfinite(tax_rate) or not 0.0 <= tax_rate < 1.0:
        raise ValueError("tax_rate must be finite and in [0, 1)")
    if not math.isfinite(min_ev) or min_ev < 0.0:
        raise ValueError("min_ev must be finite and nonnegative")
    if not math.isfinite(bankroll) or bankroll < 0.0:
        raise ValueError("bankroll must be finite and nonnegative")
    open_bets_df = query_df(
        "SELECT COALESCE(SUM(stake), 0.0) AS open_stake FROM bets WHERE status = 'open'"
    )
    existing_reserved = (
        float(open_bets_df.iloc[0]["open_stake"]) if not open_bets_df.empty else 0.0
    )

    effective_reserved = (
        existing_reserved if reserved_bankroll is None else float(reserved_bankroll)
    )
    if not math.isfinite(effective_reserved) or effective_reserved < 0.0:
        raise ValueError("reserved_bankroll must be finite and nonnegative")

    evaluated_at = datetime.now(UTC)
    rows = query_df(
        """SELECT p.*, p.id AS prediction_id, cm.team_a_name,cm.team_b_name,
        cm.league,cm.start_time_normalized,cm.best_of
        FROM canonical_predictions p JOIN canonical_matches cm ON cm.id=p.canonical_match_id
        WHERE p.prediction_status='active' AND p.model_name=? AND p.model_version=?""",
        (model_name,model_version),
    )
    predictions = latest_predictions(rows.to_dict("records"))
    cm_ids = list(dict.fromkeys(int(r["canonical_match_id"]) for r in predictions))
    contexts = load_prematch_quotes(cm_ids,decision_at=evaluated_at)
    generated: list[dict[str, Any]] = []
    parsed_rows = []
    consensus_market_p_a = {}
    for prediction in predictions:
        mid = int(prediction["canonical_match_id"])
        context = contexts[mid]
        if context.prob_a is None or not prediction_time_is_actionable(
            prediction,decision_at=evaluated_at,start_at=context.start_at,
            model_name=model_name,model_version=model_version,
        )[0]:
            continue
        diagnostic = prediction.get("diagnostics_json")
        try:
            diagnostic = json.loads(diagnostic) if isinstance(diagnostic,str) else diagnostic or {}
        except (TypeError,ValueError):
            continue
        if not isinstance(diagnostic,dict):
            continue
        forecast_context = diagnostic.get("market_context")
        if diagnostic.get("base_model_name") or diagnostic.get("market_features_used") or str(model_name).startswith("Hybrid"):
            if not hybrid_prediction_contract_is_current(prediction)[0]:
                continue
            if not forecast_context:
                continue
            forecast_at = parse_aware_utc(forecast_context.get("decision_at"))
            if forecast_at is None:
                continue
            at_forecast = load_prematch_quotes([mid],decision_at=forecast_at)[mid]
            if not stored_market_context_is_current(forecast_context,evaluation_context=at_forecast,current_context=context)[0]:
                continue
        consensus_market_p_a[mid] = context.prob_a
        for quote in context.quotes:
            row = dict(prediction, odds_snapshot_id=quote.odds_snapshot_id,bookmaker_id=quote.bookmaker_id,
                bookmaker=quote.bookmaker,offer_url=quote.offer_url,scraped_at=quote.scraped_at)
            ma,mb = fair_market_probabilities(quote.odds_a,quote.odds_b)
            parsed_rows.append(dict(row=row,odds_a=quote.odds_a,odds_b=quote.odds_b,single_market_a=ma,single_market_b=mb))
    opening_market_probs: dict[int, list[float]] = {}
    if cm_ids:
        placeholders = ",".join("?" for _ in cm_ids)
        historical = query_df(
            f"SELECT * FROM odds_snapshots WHERE market_type='match_winner' AND canonical_match_id IN ({placeholders})",
            tuple(cm_ids),
        ).to_dict("records")
        earliest = {}
        blocked = set()
        for observed in historical:
            key = (int(observed["canonical_match_id"]),int(observed["bookmaker_id"]))
            clock = parse_aware_utc(observed.get("scraped_at"))
            if clock is None:
                blocked.add(key)
                continue
            if clock > evaluated_at:
                continue
            rank = (clock,int(observed["id"]))
            if key not in earliest or rank < earliest[key][0]:
                earliest[key] = (rank,observed)
        for key,(rank,observed) in earliest.items():
            context = contexts[key[0]]
            if key in blocked or not context.start_at or rank[0] >= context.start_at or observed.get("is_live") != 0:
                continue
            aligned = align_snapshot_odds(
                context.team_a_name,context.team_b_name,
                str(observed.get("raw_team_a") or ""),str(observed.get("raw_team_b") or ""),
                observed.get("odds_a"),observed.get("odds_b"),
                league=context.league,source_system="bookmaker",match_date=context.start_at.date(),
            )
            if aligned is not None:
                opening_market_probs.setdefault(key[0],[]).append(fair_market_probabilities(*aligned)[0])

    consensus_open_market_p_a: dict[int, float] = {
        cm_id: sum(probs) / len(probs)
        for cm_id, probs in opening_market_probs.items()
        if probs
    }
    thesis_prob_a_by_match: dict[int, float] = {}
    if cm_ids:
        try:
            placeholders = ",".join("?" for _ in cm_ids)
            thesis_rows = query_df(
                f"""SELECT canonical_match_id, prob_a, predicted_at, id
                FROM canonical_predictions
                WHERE prediction_status='active'
                  AND model_name='Sym-Cal LR-ElasticNet-W20-Binomial'
                  AND canonical_match_id IN ({placeholders})""",
                tuple(cm_ids),
            ).to_dict("records")
            for t_row in latest_predictions(thesis_rows):
                t_mid = int(t_row["canonical_match_id"])
                t_pa = none_or_float(t_row.get("prob_a"))
                if t_pa is not None and 0.0 < t_pa < 1.0:
                    thesis_prob_a_by_match[t_mid] = t_pa
        except Exception:
            thesis_prob_a_by_match = {}

    stats: dict[str, Any] = {
        "total_fixtures_evaluated": len(cm_ids),
        "predictions_evaluated": len(rows),
        "candidate_sides_evaluated": 0,
        "value_bets_qualified": 0,
        "total_quarantined": 0,
        "quarantined_breakdown": Counter(),
        "disqualified_breakdown": Counter(),
    }

    # 2. Collect candidate signals across both sides
    candidate_signals: list[dict[str, Any]] = []
    for item in parsed_rows:
        row = item["row"]
        cm_id = int(row["canonical_match_id"])
        cons_a = consensus_market_p_a.get(cm_id, item["single_market_a"])
        cons_b = 1.0 - cons_a
        open_a = consensus_open_market_p_a.get(cm_id, cons_a)
        open_b = 1.0 - open_a

        # Extract tier, best_of, and rating disagreement for safety gating
        tier_str = qualification_tier(
            row.get("league"), row.get("start_time_normalized")
        )
        effective_best_of = row.get("best_of")

        try:
            diag_data = prediction_safety_diagnostics(
                row.get("diagnostics_json"),
                float(row["prob_a"]),
                float(row["prob_b"]),
                uncertainty_required=model_requires_uncertainty(model_name),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping EV prediction %s: %s", row["prediction_id"], exc)
            continue
        disag = diag_data.get("rating_disagreement")
        if disag is None and cm_id in thesis_prob_a_by_match:
            base_pa = none_or_float(diag_data.get("base_prob_a"))
            if base_pa is None or not 0.0 < base_pa < 1.0:
                base_pa = float(row["prob_a"])
            disag = abs(base_pa - thesis_prob_a_by_match[cm_id])
        if effective_best_of is None:
            effective_best_of = diag_data.get("best_of")

        candidates = [
            ("a", float(row["prob_a"]), item["odds_a"], cons_a, open_a),
            ("b", float(row["prob_b"]), item["odds_b"], cons_b, open_b),
        ]
        for side, prob, odds, cons_prob, open_prob in candidates:
            stats["candidate_sides_evaluated"] += 1
            conservative_prob = diag_data.get(f"p_low_{side}")
            eligible, reason, diag_res = is_bet_eligible(
                prob_model=prob,
                odds=odds,
                prob_market_novig=open_prob,
                tax_rate=tax_rate,
                min_ev_net=min_ev,
                rating_disagreement=disag,
                competition_tier=tier_str,
                prob_conservative=conservative_prob,
                uncertainty_required=diag_data["uncertainty_required"],
                uncertainty_basis=diag_data["uncertainty_basis"],
                uncertainty_reason=diag_data["uncertainty_reason"],
                best_of=int(effective_best_of) if effective_best_of else None,
                prob_market_close_novig=cons_prob,
                max_ev_net=0.25,
                max_rating_disagreement=0.10,
                universal_rating_guard=True,
                max_odds=4.00,
                wasserstein_epsilon=wasserstein_epsilon,
            )
            if not eligible:
                stats["disqualified_breakdown"][reason] += 1
                if diag_res.get("quarantine"):
                    q_reason = diag_res.get("quarantine_reason", reason)
                    stats["quarantined_breakdown"][q_reason] += 1
                    stats["total_quarantined"] += 1
                continue
            stats["value_bets_qualified"] += 1
            ev = expected_value(diag_res["prob_for_ev"], odds, tax_rate)
            market_edge = (odds * cons_prob) - 1.0
            candidate_signals.append(
                {
                    "row": row,
                    "side": side,
                    "odds": odds,
                    "prob": prob,
                    "prob_conservative": diag_res["prob_conservative"],
                    "prob_for_ev": diag_res["prob_for_ev"],
                    "uncertainty_basis": diag_res["uncertainty_basis"],
                    "uncertainty_reason": diag_res["uncertainty_reason"],
                    "competition_tier": tier_str,
                    "market_prob": cons_prob,
                    "ev": ev,
                    "market_edge": market_edge,
                    "outlier_ratio": odds * cons_prob,
                    "logit_disagreement": diag_res.get("logit_disagreement"),
                }
            )

    # Sort candidates by EV descending so highest-value bets receive allocation priority
    candidate_signals.sort(key=lambda x: x["ev"], reverse=True)

    with transaction() as connection:
        connection.execute(
            """
            UPDATE model_ev_signals
            SET status = 'stale'
            WHERE status = 'new' AND canonical_prediction_id IN (
                SELECT id FROM canonical_predictions WHERE model_name = ? AND model_version = ?
            )
            """,
            (model_name, model_version),
        )

        running_reserved = effective_reserved
        allocated_matches: set[tuple[int, str]] = set()

        for sig in candidate_signals:
            row = sig["row"]
            cm_id = int(row["canonical_match_id"])
            context = contexts[cm_id]
            signal_context = serialize_market_context(context,model_name=model_name,model_version=model_version)
            signal_context["logit_disagreement"] = sig.get("logit_disagreement")
            current = load_prematch_quotes([cm_id],decision_at=datetime.now(UTC),session=connection._session)[cm_id]
            if not stored_market_context_is_current(
                signal_context,evaluation_context=context,current_context=current,
            )[0]:
                continue
            side = sig["side"]
            odds = sig["odds"]
            prob = sig["prob"]
            market_prob = sig["market_prob"]
            ev = sig["ev"]

            stake = fractional_kelly_stake(
                bankroll,
                sig["prob_for_ev"],
                odds,
                fraction=0.25,
                tax_rate=tax_rate,
                reserved_bankroll=running_reserved,
            )
            if (cm_id, side) not in allocated_matches and stake > 0:
                allocated_matches.add((cm_id, side))
                running_reserved += stake

            cursor = connection.execute(
                """
                INSERT INTO model_ev_signals(
                    canonical_match_id, canonical_prediction_id, odds_snapshot_id, bookmaker_id,
                    side, odds, model_prob, market_prob, ev, tax_rate, stake_suggestion, status,
                    evaluated_at, market_context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?) RETURNING id
                """,
                (
                    cm_id,
                    int(row["prediction_id"]),
                    int(row["odds_snapshot_id"]),
                    int(row["bookmaker_id"]),
                    side,
                    odds,
                    prob,
                    market_prob,
                    ev,
                    tax_rate,
                    stake,
                    evaluated_at,
                    json.dumps(signal_context,ensure_ascii=False,sort_keys=True),
                ),
            )
            generated.append(
                {
                    "signal_id": int(cursor.fetchone()["id"]),
                    "canonical_match_id": cm_id,
                    "odds_snapshot_id": int(row["odds_snapshot_id"]),
                    "quote_scraped_at": row["scraped_at"].isoformat(),
                    "evaluated_at": evaluated_at.isoformat(),
                    "match": f"{row.get('team_a_name')} vs {row.get('team_b_name')}",
                    "bookmaker": row.get("bookmaker"),
                    "side": side,
                    "odds": odds,
                    "model_prob": prob,
                    "prob_conservative": sig["prob_conservative"],
                    "prob_for_ev": sig["prob_for_ev"],
                    "uncertainty_basis": sig["uncertainty_basis"],
                    "uncertainty_reason": sig["uncertainty_reason"],
                    "competition_tier": sig["competition_tier"],
                    "market_prob": market_prob,
                    "market_edge": round(sig["market_edge"], 4),
                    "outlier_ratio": round(sig["outlier_ratio"], 4),
                    "ev": ev,
                    "stake_suggestion": stake,
                    "logit_disagreement": sig.get("logit_disagreement"),
                    "offer_url": row.get("offer_url"),
                }
            )
    res = sorted(generated, key=lambda item: item["ev"], reverse=True)
    setattr(generate_model_ev_signals, "last_stats", stats)
    if return_stats:
        return res, stats
    return res


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)
