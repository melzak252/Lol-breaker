from betting_app.scrapers.golgg import GolggScraper

import asyncio
from pathlib import Path

import httpx
from parsel import Selector
import pytest


def _game_html(blue_id, red_id, blue_win):
    blocks = []
    tables = []
    for side, team_id, won in (("blue", blue_id, blue_win), ("red", red_id, not blue_win)):
        blocks.append(
            f'<div><div><div class="{side}-line-header"><a href="../teams/team-stats/{team_id}/">Team {team_id}</a> - {"WIN" if won else "LOSS"}</div></div>'
            '<div>' + ''.join(f'<div><span>{value}</span></div>' for value in ("18", "7", "2", "1", "56.8k", "")) + '</div><div></div></div>'
        )
        tables.append('<table class="playersInfosLine">' + ''.join(
            f'<tr><td><a href="../players/player-stats/{team_id}{index}/">Player {team_id}{index}</a></td></tr>'
            for index in range(5)
        ) + '</table>')
    return '<html><body><div class="col-cadre"><div><h1>29:58</h1></div><div>' + ''.join(blocks) + '</div></div>' + ''.join(tables) + '</body></html>'


def _summary_html():
    # Summary order and labels intentionally conflict with each actual map.
    return '<div id="gameMenuToggler"><ul><li><a href="../game/stats/101/page-game/">Game 1</a></li><li><a href="../game/stats/102/page-game/">Game 2</a></li></ul></div><div class="col-cadre"><div><a href="../teams/team-stats/99/">Wrong summary identity</a><span> BO3 </span><a href="../teams/team-stats/98/">Wrong opponent</a></div><div><span class="text_victory">WIN</span></div><div><span class="text_defeat">LOSS</span></div></div>'


def test_side_swapped_series_uses_actual_map_identity_winner_and_stats():
    async def run():
        def respond(request):
            if "page-summary" in request.url.path:
                html = _summary_html()
            elif "page-fullstats" in request.url.path:
                html = '<table class="completestats"><tr><td>Kills</td>' + ''.join(f'<td>{value}</td>' for value in range(10)) + '</tr></table>'
            else:
                html = _game_html("1", "2", False) if "/101/" in request.url.path else _game_html("2", "1", True)
            return httpx.Response(200, text=html)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            scraper = GolggScraper()
            scraper.client = client
            games = await scraper.get_games_in_match("101")
        first, second = games
        assert (first["t1_id"], first["t2_id"], first["t1_win"], first["t2_win"]) == ("1", "2", False, True)
        assert (second["t1_id"], second["t2_id"], second["t1_win"], second["t2_win"]) == ("2", "1", True, False)
        assert first["t2_players"]["TOP"]["player_id"] == second["t1_players"]["TOP"]["player_id"] == "20"
        assert first["t2_players"]["TOP"]["stats"]["kills"] == 5
        assert second["t1_players"]["TOP"]["stats"]["kills"] == 0
        assert all((game["t1_side"], game["t2_side"]) == ("blue", "red") for game in games)
        assert all(game["source_game_ids"] == ["101", "102"] for game in games)
        assert all(game["source_best_of"] == 3 for game in games)
    asyncio.run(run())


def _evidenced_correction():
    return {"games": {"19645": {
        "expected": {
            "blue_id": "809", "red_id": "809",
            "blue_name": "Inside Games", "red_name": "Inside Games",
            "blue_win": True, "red_win": False,
            "blue_player_ids": ["1869", "3143", "2466", "2469", "1873"],
            "red_player_ids": ["2457", "1870", "2455", "2534", "2485"],
        },
        "corrected": {"red_id": "815", "red_name": "iNvolute"},
        "provenance": {"source_url": "https://gol.gg/game/stats/19646/page-game/", "evidence": "Map 19646 independently identifies each five-player roster on opposite sides."},
    }}}


def test_verified_identity_correction_preserves_both_opposing_rosters():
    source = Selector(text=(Path(__file__).parent / "fixtures/golgg/duplicate_team_game.html").read_text())
    scraper = GolggScraper(source_corrections=_evidenced_correction())
    players = asyncio.run(scraper.get_players_in_game(source, game_id="19645"))
    assert {p["player_id"] for p in players["809"].values()} == {"1869", "3143", "2466", "2469", "1873"}
    assert {p["player_id"] for p in players["815"].values()} == {"2457", "1870", "2455", "2534", "2485"}
    stats = asyncio.run(scraper.get_team_stats(source, game_id="19645"))
    assert (stats["blue_id"], stats["red_id"], stats["blue"]["kills"], stats["red"]["kills"]) == ("809", "815", 18, 4)


def test_evidenced_correction_rejects_changed_source_roster():
    source = Selector(text=(Path(__file__).parent / "fixtures/golgg/duplicate_team_game.html").read_text().replace("player-stats/1869/", "player-stats/9999/"))
    scraper = GolggScraper(source_corrections=_evidenced_correction())
    with pytest.raises(ValueError, match="expectation mismatch"):
        asyncio.run(scraper.get_players_in_game(source, game_id="19645"))


def test_incomplete_series_raises_instead_of_returning_partial_maps():
    async def run():
        html = _summary_html().replace('<li><a href="../game/stats/102/page-game/">Game 2</a></li>', '')
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))) as client:
            scraper = GolggScraper()
            scraper.client = client
            with pytest.raises(ValueError, match="map rows"):
                await scraper.get_games_in_match("101")
    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["LOSS", "UNKNOWN"])
def test_missing_or_conflicting_map_winners_are_rejected(outcome):
    html = _game_html("1", "2", True).replace(" - WIN", f" - {outcome}")
    with pytest.raises(ValueError, match="winner flags"):
        asyncio.run(GolggScraper().get_players_in_game(Selector(text=html)))


def test_preview_without_result_is_unplayed_but_broken_completed_result_is_not():
    async def run():
        html = '<table class="table_list" data-sort="0"><tbody>'
        for mid, page in (("1", "page-preview"), ("2", "page-summary")):
            html += f'<tr><td><a href="../game/stats/{mid}/{page}/">A vs B</a></td><td>A</td><td> - </td><td>B</td></tr>'
        html += '</tbody></table>'
        scraper = GolggScraper()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))) as client:
            scraper.client = client
            assert await scraper.get_matches_in_tournament("Fixture") == []
        assert {row["match_id"] for row in scraper.unplayed_rows} == {"1"}
        assert {row["match_id"] for row in scraper.failed_rows} == {"2"}
    asyncio.run(run())


def test_missing_tournament_table_is_not_a_successful_empty_history():
    async def run():
        scraper = GolggScraper()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>Unavailable</html>"))) as client:
            scraper.client = client
            with pytest.raises(ValueError, match="games table"):
                await scraper.get_matches_in_tournament("Fixture")
    asyncio.run(run())
