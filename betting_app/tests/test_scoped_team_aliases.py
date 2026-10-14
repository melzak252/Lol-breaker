from __future__ import annotations

import os

from betting_app.core.db import dispose_engine, init_db, transaction
from betting_app.services.canonical_match_service import (
    canonical_match_score,
    canonical_team_key,
)
from betting_app.services.mapping_service import suggest_mapping, upsert_alias
from betting_app.services.team_alias_service import AliasContext, resolve_scoped_alias


def _init_tmp_db(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'aliases.sqlite3'}"
    dispose_engine()
    init_db()


def test_short_alias_requires_scope(tmp_path) -> None:
    _init_tmp_db(tmp_path)
    try:
        upsert_alias(
            "USE",
            "Unicorns Of Love Sexy Edition",
            source="golgg-short",
            source_system="golgg",
            league_pattern="Prime League",
        )

        assert resolve_scoped_alias("USE", context=AliasContext()).target_name is None
        assert resolve_scoped_alias("USE", context=AliasContext(source_system="golgg", league="NACL")).target_name is None

        resolved = resolve_scoped_alias("USE", context=AliasContext(source_system="golgg", league="Prime League"))
        assert resolved.target_name == "Unicorns Of Love Sexy Edition"
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()


def test_equally_scoped_conflicting_aliases_are_ambiguous(tmp_path) -> None:
    _init_tmp_db(tmp_path)
    try:
        upsert_alias(
            "Keyd Stars",
            "Keyd Stars",
            source="manual-a",
            source_system="bookmaker",
            league_pattern="CBLOL",
        )
        upsert_alias(
            "Keyd Stars",
            "Vivo Keyd Stars",
            source="manual-b",
            source_system="bookmaker",
            league_pattern="CBLOL",
        )

        resolution = resolve_scoped_alias(
            "Keyd Stars",
            context=AliasContext(source_system="bookmaker", league="CBLOL"),
        )

        assert resolution.target_name is None
        assert resolution.source == "ambiguous"
        assert resolution.confidence == 0.0
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()


def test_short_bookmaker_alias_is_limited_to_its_competition(tmp_path) -> None:
    _init_tmp_db(tmp_path)
    try:
        upsert_alias(
            "NiP",
            "Ninjas in Pyjamas",
            source="manual",
            source_system="bookmaker",
            league_pattern="LPL",
        )

        assert canonical_team_key(
            "NiP",
            league="TJ Sports LoL / LPL",
            source_system="bookmaker",
        ) == "ninjas in pyjamas"
        assert canonical_team_key(
            "NiP",
            league="NLC",
            source_system="bookmaker",
        ) == "nip"
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()


def test_scoped_nip_alias_deduplicates_reversed_lpl_fixture(tmp_path) -> None:
    _init_tmp_db(tmp_path)
    try:
        upsert_alias(
            "NiP",
            "Ninjas in Pyjamas",
            source="manual",
            source_system="bookmaker",
            league_pattern="LPL",
        )
        incoming_a = canonical_team_key(
            "JD",
            league="LPL",
            source_system="bookmaker",
        )
        incoming_b = canonical_team_key(
            "NiP",
            league="LPL",
            source_system="bookmaker",
        )

        score = canonical_match_score(
            incoming_a,
            incoming_b,
            "2026-09-05T09:00:00+00:00",
            "LPL",
            {
                "normalized_team_a": "ninjas in pyjamas",
                "normalized_team_b": "jd",
                "start_time_normalized": "2026-09-05T09:00:00+00:00",
                "league": "TJ Sports LoL / LPL",
                "status": "upcoming",
            },
        )

        assert score >= 0.85
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()


def test_alias_api_requires_scope_for_short_names(client) -> None:
    unscoped = client.post(
        "/matches/alias",
        json={"raw_name": "NiP", "golgg_team_name": "Ninjas in Pyjamas"},
    )
    scoped = client.post(
        "/matches/alias",
        json={
            "raw_name": "NiP",
            "golgg_team_name": "Ninjas in Pyjamas",
            "source_system": "bookmaker",
            "league_pattern": "LPL",
        },
    )

    assert unscoped.status_code == 400
    assert scoped.status_code == 200
    assert scoped.json()["source"] == "manual-scoped"


def test_canonical_team_key_uses_db_scoped_alias(tmp_path) -> None:
    _init_tmp_db(tmp_path)
    try:
        upsert_alias(
            "KHK",
            "Kaufland Hangry Knights",
            source="golgg-short",
            source_system=None,
            league_pattern="Prime League",
        )

        assert canonical_team_key("KHK") == "khk"
        assert canonical_team_key("KHK", league="Prime League") == "hangryknights"
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()


def test_suggest_mapping_accepts_scoped_alias_context(tmp_path) -> None:
    _init_tmp_db(tmp_path)
    try:
        with transaction() as connection:
            connection.execute(
                "INSERT INTO golgg_teams(team_name, normalized_name) VALUES (?, ?)",
                ("Bilibili Gaming", "bilibili gaming"),
            )
        upsert_alias(
            "BLG",
            "Bilibili Gaming",
            source="golgg-short",
            source_system="golgg",
            league_pattern="MSI",
        )

        assert suggest_mapping("BLG") == (None, 0.0, None)
        assert suggest_mapping("BLG", source_system="golgg", league="MSI 2026") == (
            "Bilibili Gaming",
            1.0,
            "golgg-short:golgg:MSI:*:*:*",
        )
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()
