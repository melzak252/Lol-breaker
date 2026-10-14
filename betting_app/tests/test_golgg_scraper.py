import asyncio
from pathlib import Path

from parsel import Selector
import pytest

import os

from betting_app.core.db import dispose_engine, init_db
from betting_app.scrapers.golgg import score_for_link_order
from betting_app.services.mapping_service import upsert_alias


def _init_tmp_db(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'golgg_aliases.sqlite3'}"
    dispose_engine()
    init_db()


def test_score_for_link_order_handles_golgg_abbrev_vs_full_name_reversed_result_cells(tmp_path) -> None:
    _init_tmp_db(tmp_path)
    try:
        upsert_alias(
            "BLG",
            "Bilibili Gaming",
            source="golgg-short",
            source_system="golgg",
            league_pattern="MSI",
        )
        team_a_score, team_b_score = score_for_link_order(
            team_a="BLG",
            team_b="T1",
            result_left_team="T1",
            result_right_team="Bilibili Gaming",
            score_left=2,
            score_right=3,
            won="Bilibili Gaming",
            tournament_name="MSI 2026",
        )
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()

    assert (team_a_score, team_b_score) == (3, 2)


def test_score_for_link_order_keeps_matching_result_cell_order() -> None:
    team_a_score, team_b_score = score_for_link_order(
        team_a="T1",
        team_b="Team Liquid",
        result_left_team="T1",
        result_right_team="Team Liquid",
        score_left=1,
        score_right=0,
        won="T1",
    )

    assert (team_a_score, team_b_score) == (1, 0)


def test_duplicate_team_ids_cannot_silently_overwrite_opponent_roster() -> None:
    from betting_app.scrapers.golgg import GolggScraper

    source = Selector(
        text=(Path(__file__).parent / "fixtures/golgg/duplicate_team_game.html").read_text()
    )
    with pytest.raises(ValueError, match="(?i)(duplicate|distinct|same team)"):
        asyncio.run(GolggScraper().get_players_in_game(source))


@pytest.fixture
def isolated_score_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'score_aliases.sqlite3'}")
    dispose_engine()
    init_db()
    yield
    dispose_engine()


def test_winner_only_evidence_assigns_larger_score_not_left_score(isolated_score_aliases):
    assert score_for_link_order(
        team_a="T1", team_b="Top Esports",
        result_left_team=None, result_right_team=None,
        score_left=0, score_right=3, won="T1",
    ) == (3, 0)


@pytest.mark.parametrize(
    ("left", "right", "won"),
    [
        ("T1", "Top Esports", "Top Esports"),
        ("T1", "T1", "T1"),
        (None, None, None),
    ],
)
def test_conflicting_or_ungrounded_score_order_is_rejected(
    isolated_score_aliases, left, right, won
):
    with pytest.raises(ValueError):
        score_for_link_order(
            team_a="T1", team_b="Top Esports",
            result_left_team=left, result_right_team=right,
            score_left=3, score_right=0, won=won,
        )


def test_draw_without_side_evidence_has_no_directional_ambiguity(isolated_score_aliases):
    assert score_for_link_order(
        team_a="T1", team_b="Top Esports",
        result_left_team=None, result_right_team=None,
        score_left=1, score_right=1,
    ) == (1, 1)
