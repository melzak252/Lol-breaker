from betting_app.services.golgg_import_service import match_row


def test_match_row_preserves_games_played_when_match_metadata_has_no_nested_games() -> None:
    row = match_row(
        {
            "match_id": "79618",
            "date": "2026-07-04",
            "tournament_name": "MSI 2026",
            "sname_t1": "BLG",
            "sname_t2": "T1",
            "t1_score": 3,
            "t2_score": 2,
            "games_played": 5,
            "best_of": 5,
            "won": "Bilibili Gaming",
            "lost": "T1",
        },
        "79618",
    )

    # match_row tuple column order: ..., draw, games_played, best_of, winner_name, ...
    assert row[13] == 5
    assert row[14] == 5
