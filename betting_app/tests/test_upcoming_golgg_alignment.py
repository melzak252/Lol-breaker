from betting_app.services.upcoming_inference_service import _select_aligned_golgg_ids


def test_latest_reversed_bookmaker_offer_is_swapped_to_canonical_order() -> None:
    canonical = {
        "team_a_name": "JD Gaming",
        "team_b_name": "LGD Gaming",
        "league": "LPL",
        "start_time_normalized": "2026-08-05T09:00:00+00:00",
    }
    newest_reversed_offer = {
        "raw_team_a": "LGD Gaming",
        "raw_team_b": "JD Gaming",
        "league": "LPL",
        "team_a_golgg_id": 14080,
        "team_b_golgg_id": 991,
    }

    assert _select_aligned_golgg_ids(canonical, [newest_reversed_offer]) == (991, 14080)


def test_wrong_team_offer_is_not_used_as_a_golgg_mapping() -> None:
    canonical = {
        "team_a_name": "NS Challengers",
        "team_b_name": "KT Rolster Challengers",
        "league": "LCK Challengers",
        "start_time_normalized": "2026-08-04T10:00:00+00:00",
    }
    stale_wrong_offer = {
        "raw_team_a": "T1 Challengers",
        "raw_team_b": "KT Rolster Challengers",
        "league": "LCK Challengers",
        "team_a_golgg_id": 1742,
        "team_b_golgg_id": 1049,
    }

    assert _select_aligned_golgg_ids(canonical, [stale_wrong_offer]) == (None, None)
