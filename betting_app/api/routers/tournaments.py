"""FastAPI router for tournament brackets and Monte Carlo simulations."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from betting_app.services.liquipedia_bracket_service import LiquipediaBracketService
from betting_app.services.enc_simulation_service import EncSimulator, build_enc_configuration
from betting_app.services.tournament_cache_service import (
    current_input_fingerprint,
    get_cache_file,
    get_cached_simulation,
    recalculate_and_cache,
)
from betting_app.services.tournament_service import (
    SUPPORTED_BRACKETS,
    TournamentSimulator,
    WorldsSimulator,
    WorldsTeam,
)
router = APIRouter(prefix="/tournaments", tags=["tournaments"])


class SimulateTournamentRequest(BaseModel):
    simulations: int = 10000
    manual_overrides: dict[str, str] | None = None  # match_id -> winner team name
    engine: str = "advanced"  # "advanced" | "classic"
    use_fearless: bool = True
    use_bayesian_momentum: bool = True
    use_hybrid_knockout: bool = True
    seed: int = 42

class SyncBracketRequest(BaseModel):
    source: str = "auto"  # "auto" | "fandom" | "liquipedia"
    force: bool = False
    raw_content: str | None = None


class WorldsTeamInput(BaseModel):
    team: str
    region: str
    pool: int | None = None


class SimulateWorldsRequest(BaseModel):
    simulations: int = 5000
    direct_teams: list[WorldsTeamInput] | None = None
    play_in_teams: list[WorldsTeamInput] | None = None
    play_in_winner_pool: int = 4
    engine: str = "advanced"  # "advanced" | "classic"
    use_fearless: bool = True
    use_bayesian_momentum: bool = True
    use_hybrid_knockout: bool = True
    use_riot_lottery: bool = True
    seed: int = 42

class SimulateWorldsCrystalBallRequest(BaseModel):
    n_simulations: int = Field(default=1000, ge=10, le=10000)
    seed: int = 42
    direct_teams: list[WorldsTeamInput] | None = None
    play_in_teams: list[WorldsTeamInput] | None = None

class SimulateEncRequest(BaseModel):
    simulations: int = 5000

class SimulateFearlessTournamentRequest(BaseModel):
    team_a_name: str
    team_b_name: str
    team_a_roster: list[str] | None = None
    team_b_roster: list[str] | None = None
    best_of: int = 5
    n_simulations: int = 5000
    seed: int = 42
    base_map_prob: float | None = None

def _validate_bracket_for_route(
    bracket: Any,
    manual_overrides: dict[str, str] | None = None,
) -> None:
    """Reject malformed bracket graphs before resolving native C0 inputs."""
    TournamentSimulator()._validate_bracket(bracket, manual_overrides or {})


def _invalid_bracket(error: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=f"Invalid tournament bracket: {error}")


def _native_c0_error(error: Exception) -> HTTPException:
    if isinstance(error, ValueError):
        return HTTPException(status_code=422, detail=f"Native C0 unavailable: {error}")
    return HTTPException(status_code=500, detail="Native C0 service unavailable.")


@router.get("")
def list_tournaments() -> list[dict[str, Any]]:
    """Return available tournaments supported for bracket simulation."""
    result = []
    for fn in SUPPORTED_BRACKETS.values():
        b = fn()
        result.append(
            {
                "id": b.id,
                "name": b.name,
                "region": b.region,
                "format": b.format,
                "teams": b.teams,
            }
        )
    return result



@router.post("/worlds/simulate")
def simulate_worlds(body: SimulateWorldsRequest = SimulateWorldsRequest()) -> dict[str, Any]:
    """Simulate Worlds 2026 Play-In, Swiss Stage, and Knockouts with advanced dynamic engine or classic."""
    if body.engine == "advanced":
        from src.models.advanced_tournament_simulator import AdvancedTournamentSimulator
        sim = AdvancedTournamentSimulator(
            use_fearless=body.use_fearless,
            use_bayesian_momentum=body.use_bayesian_momentum,
            use_hybrid_knockout=body.use_hybrid_knockout,
            use_riot_lottery=body.use_riot_lottery,
            shrinkage_lambda=0.25,
            temperature=1.08,
        )
        teams = [t.team for t in body.direct_teams] if body.direct_teams else None
        regions = {t.team: t.region for t in body.direct_teams} if body.direct_teams else None
        n_sims = min(max(body.simulations, 100), 20000)
        res = sim.simulate_worlds(
            teams=teams,
            team_regions=regions,
            n_simulations=n_sims,
            seed=body.seed,
        )
        return res.to_dict() if hasattr(res, "to_dict") else dict(res)

    # Classic fallback
    if not body.direct_teams or not body.play_in_teams:
        raise HTTPException(status_code=422, detail="Classic engine requires explicit direct_teams and play_in_teams.")
    simulator = WorldsSimulator(competition_context="worlds_2026")
    direct_teams = [
        WorldsTeam(name=team.team, region=team.region, pool=team.pool)
        for team in body.direct_teams
    ]
    play_in_teams = [
        WorldsTeam(name=team.team, region=team.region)
        for team in body.play_in_teams
    ]
    n_simulations = min(max(body.simulations, 100), 20000)
    try:
        return simulator.simulate_worlds(
            direct_teams=direct_teams,
            play_in_teams=play_in_teams,
            play_in_winner_pool=body.play_in_winner_pool,
            n_simulations=n_simulations,
        )
    except Exception as error:
        raise _native_c0_error(error) from error

@router.post("/worlds/crystal-ball")
def simulate_worlds_crystal_ball_endpoint(
    body: SimulateWorldsCrystalBallRequest = SimulateWorldsCrystalBallRequest(),
) -> dict[str, Any]:
    """Simulate Worlds 2026 Crystal Ball categories under Hard Fearless rules."""
    from src.models.crystal_ball_simulator import simulate_worlds_crystal_ball

    direct_input = (
        [{"team": t.team, "region": t.region, "pool": t.pool} for t in body.direct_teams]
        if body.direct_teams
        else None
    )
    play_in_input = (
        [{"team": t.team, "region": t.region} for t in body.play_in_teams]
        if body.play_in_teams
        else None
    )

    try:
        results = simulate_worlds_crystal_ball(
            direct_teams=direct_input,
            play_in_teams=play_in_input,
            n_simulations=body.n_simulations,
            seed=body.seed,
        )
        return results.to_dict()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Crystal Ball simulation failed: {error}") from error

@router.get("/enc")
def get_enc_configuration() -> dict[str, Any]:
    """Return published ENC roster and ratings as metadata for native C0."""
    try:
        return build_enc_configuration()
    except Exception as error:
        raise _native_c0_error(error) from error


@router.post("/enc/simulate")
def simulate_enc(body: SimulateEncRequest) -> dict[str, Any]:
    """Simulate the published ENC 2027 format when every roster is verifiable."""
    n_simulations = min(max(body.simulations, 100), 50000)
    try:
        configuration = build_enc_configuration()
        return EncSimulator.from_configuration(configuration).simulate(n_simulations)
    except Exception as error:
        raise _native_c0_error(error) from error


@router.get("/{tournament_id}")
def get_tournament_bracket(
    tournament_id: str,
    simulations: int = Query(10000, description="Monte Carlo simulation path depth (5000, 10000, 20000, 100000)"),
) -> dict[str, Any]:
    """Return a fresh C0 simulation when its current input fingerprint matches."""
    builder = SUPPORTED_BRACKETS.get(tournament_id)
    if not builder:
        raise HTTPException(status_code=404, detail=f"Tournament {tournament_id} not found")

    bracket = builder()
    try:
        _validate_bracket_for_route(bracket)
    except Exception as error:
        raise _invalid_bracket(error) from error

    try:
        expected_fingerprint = current_input_fingerprint(tournament_id, bracket.teams)
        cached = get_cached_simulation(
            tournament_id,
            n_simulations=simulations,
            expected_input_fingerprint=expected_fingerprint,
        )
        if cached is not None:
            return cached
        return recalculate_and_cache(tournament_id, depths=(simulations,))
    except Exception as error:
        raise _native_c0_error(error) from error

@router.post("/{tournament_id}/sync")
def sync_tournament_bracket(
    tournament_id: str,
    body: SyncBracketRequest = SyncBracketRequest(),
) -> dict[str, Any]:
    """Sync the tournament bracket from LoL Fandom or Liquipedia, then simulate."""
    builder = SUPPORTED_BRACKETS.get(tournament_id)
    if not builder:
        raise HTTPException(status_code=404, detail=f"Tournament {tournament_id} not found")

    bracket = builder()
    try:
        _validate_bracket_for_route(bracket)
    except Exception as error:
        raise _invalid_bracket(error) from error

    service = LiquipediaBracketService()
    sync_res = service.sync_bracket(
        tournament_id,
        source=body.source,
        raw_content=body.raw_content,
        force=body.force,
    )
    if sync_res.get("ok"):
        get_cache_file(tournament_id).unlink(missing_ok=True)
    bracket = sync_res.get("bracket") or bracket
    try:
        _validate_bracket_for_route(bracket)
    except Exception as error:
        raise _invalid_bracket(error) from error
    simulator = TournamentSimulator()
    simulator.competition_context = tournament_id
    try:
        sim_res = simulator.simulate(bracket, n_simulations=5000)
    except Exception as error:
        raise _native_c0_error(error) from error
    sim_res["source"] = sync_res.get("source", "curated")
    sim_res["status"] = sync_res.get("status", "ready")
    sim_res["synced_at"] = sync_res.get("synced_at")
    sim_res["sync_message"] = sync_res.get("message")
    sim_res["updated_matches"] = sync_res.get("updated_matches", 0)
    return sim_res



@router.post("/fearless/simulate")
def simulate_fearless_tournament(body: SimulateFearlessTournamentRequest) -> dict[str, Any]:
    """Simulate a Fearless Draft series between two tournament teams."""
    from src.models.fearless_simulator import simulate_fearless_series
    from src.features.champion_pool_entropy import ROLES

    roster_a = body.team_a_roster or [f"{body.team_a_name}_{r}" for r in ROLES]
    roster_b = body.team_b_roster or [f"{body.team_b_name}_{r}" for r in ROLES]

    result = simulate_fearless_series(
        team_a=roster_a,
        team_b=roster_b,
        best_of=body.best_of,
        n_simulations=body.n_simulations,
        seed=body.seed,
        base_map_prob=body.base_map_prob,
    )
    return {
        "team_a_name": body.team_a_name,
        "team_b_name": body.team_b_name,
        **result.to_dict(),
    }
@router.post("/{tournament_id}/simulate")
def simulate_tournament(
    tournament_id: str,
    body: SimulateTournamentRequest = SimulateTournamentRequest(),
) -> dict[str, Any]:
    """Run Monte Carlo simulation with optional what-if manual match winners."""
    builder = SUPPORTED_BRACKETS.get(tournament_id)
    if not builder:
        raise HTTPException(status_code=404, detail=f"Tournament {tournament_id} not found")

    bracket = builder()
    try:
        _validate_bracket_for_route(bracket, body.manual_overrides)
    except Exception as error:
        raise _invalid_bracket(error) from error

    service = LiquipediaBracketService()
    sync_res = service.load_bracket(tournament_id)
    bracket = sync_res.get("bracket") or bracket
    try:
        _validate_bracket_for_route(bracket, body.manual_overrides)
    except Exception as error:
        raise _invalid_bracket(error) from error

    n_sims = min(max(body.simulations, 100), 50000)

    if body.engine == "advanced":
        from src.models.advanced_tournament_simulator import AdvancedTournamentSimulator
        sim = AdvancedTournamentSimulator(
            use_fearless=body.use_fearless,
            use_bayesian_momentum=body.use_bayesian_momentum,
            use_hybrid_knockout=body.use_hybrid_knockout,
            shrinkage_lambda=0.25,
            temperature=1.08,
        )
        sim_res = sim.simulate_bracket(
            bracket,
            n_simulations=n_sims,
            manual_overrides=body.manual_overrides,
            seed=body.seed,
        )
        if hasattr(sim_res, "to_dict"):
            sim_dict = sim_res.to_dict()
        elif isinstance(sim_res, dict):
            sim_dict = sim_res
        else:
            sim_dict = dict(sim_res)
        sim_dict["source"] = sync_res.get("source", "curated")
        sim_dict["status"] = sync_res.get("status", "ready")
        sim_dict["synced_at"] = sync_res.get("synced_at")
        sim_dict["sync_message"] = sync_res.get("message")
        sim_dict["updated_matches"] = sync_res.get("updated_matches", 0)
        return sim_dict

    # Classic fallback
    simulator = TournamentSimulator()
    simulator.competition_context = tournament_id
    try:
        sim_res = simulator.simulate(bracket, n_simulations=n_sims, manual_overrides=body.manual_overrides)
    except Exception as error:
        raise _native_c0_error(error) from error
    sim_res["source"] = sync_res.get("source", "curated")
    sim_res["status"] = sync_res.get("status", "ready")
    sim_res["synced_at"] = sync_res.get("synced_at")
    sim_res["sync_message"] = sync_res.get("message")
    sim_res["updated_matches"] = sync_res.get("updated_matches", 0)
    return sim_res

@router.post("/{tournament_id}/recalculate")
def recalculate_tournament_bracket(
    tournament_id: str,
    body: SimulateTournamentRequest = SimulateTournamentRequest(),
) -> dict[str, Any]:
    """Recalculate native C0 tournament distributions and refresh the cache."""
    builder = SUPPORTED_BRACKETS.get(tournament_id)
    if not builder:
        raise HTTPException(status_code=404, detail=f"Tournament {tournament_id} not found")


    try:
        return recalculate_and_cache(
            tournament_id,
            depths=(body.simulations,),
            manual_overrides=body.manual_overrides,
        )
    except Exception as error:
        raise _native_c0_error(error) from error

