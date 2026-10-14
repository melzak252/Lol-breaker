from __future__ import annotations
import pytest


@pytest.fixture(autouse=True)
def identity_review_token(monkeypatch):
    monkeypatch.setenv("IDENTITY_REVIEW_TOKEN", "test-review-token")


from betting_app.core.db import transaction


def _seed_review_mapping(*, with_second_mapping: bool = False) -> None:
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO canonical_matches(
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b,
                start_time_normalized, league, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (101, "review-101", "Alpha", "Beta", "alpha", "beta", "2026-01-02T12:00:00+00:00", "LPL", "finished"),
        )
        connection.execute(
            """
            INSERT INTO golgg_matches(match_id, date, tournament_name, team1_name, team2_name)
            VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
            """,
            ("501", "2026-01-01", "LCK", "Wrong A", "Wrong B", "502", "2026-01-02", "LPL", "Alpha", "Beta"),
        )
        connection.execute(
            """
            INSERT INTO golgg_match_mappings(
                canonical_match_id, golgg_match_id, confidence, mapped_by
            ) VALUES (?, ?, ?, ?)
            """,
            (101, "501", 0.7, "auto-fuzzy"),
        )
        if with_second_mapping:
            connection.execute(
                """
                INSERT INTO canonical_matches(
                    id, canonical_key, team_a_name, team_b_name,
                    normalized_team_a, normalized_team_b,
                    start_time_normalized, league, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (102, "review-102", "Gamma", "Delta", "gamma", "delta", "2026-01-02T12:00:00+00:00", "LPL", "finished"),
            )
            connection.execute(
                """
                INSERT INTO golgg_match_mappings(
                    canonical_match_id, golgg_match_id, confidence, mapped_by
                ) VALUES (?, ?, ?, ?)
                """,
                (102, "502", 1.0, "manual"),
            )


def test_review_queue_exposes_evidence_and_dependency_counts(client) -> None:
    _seed_review_mapping()

    response = client.get("/matches/mapping-review")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    item = payload["items"][0]
    assert item["canonical_match_id"] == 101
    assert set(item["reasons"]) == {
        "confidence_below_0_95",
        "date_mismatch",
        "competition_conflict",
    }
    assert item["bet_count"] == 0


def test_review_mutation_requires_operator_token(client) -> None:
    _seed_review_mapping()

    response = client.post(
        "/matches/mapping-review/decision",
        json={
            "canonical_match_id": 101,
            "decision": "retain",
            "reason": "Verified exact fixture identity",
            "operator": "test-operator",
        },
    )

    assert response.status_code == 401


def test_review_mutation_accepts_secret_file(client, monkeypatch, tmp_path) -> None:
    _seed_review_mapping()
    token_file = tmp_path / "identity-review-token"
    token_file.write_text("file-review-token\n", encoding="utf-8")
    monkeypatch.delenv("IDENTITY_REVIEW_TOKEN")
    monkeypatch.setenv("IDENTITY_REVIEW_TOKEN_FILE", str(token_file))

    response = client.post(
        "/matches/mapping-review/decision",
        json={
            "canonical_match_id": 101,
            "decision": "retain",
            "reason": "Verified exact fixture identity",
            "operator": "test-operator",
        },
        headers={"X-Identity-Review-Token": "file-review-token"},
    )

    assert response.status_code == 200, response.text


def test_review_replacement_is_atomic_and_audited(client) -> None:
    _seed_review_mapping()

    response = client.post(
        "/matches/mapping-review/decision",
        json={
            "canonical_match_id": 101,
            "decision": "replace",
            "new_golgg_match_id": "502",
            "reason": "Exact date, teams, and LPL competition verified",
            "operator": "test-operator",
        },
        headers={"X-Identity-Review-Token": "test-review-token"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["old_golgg_match_id"] == "501"
    assert response.json()["new_golgg_match_id"] == "502"
    with transaction() as connection:
        mapping = connection.execute(
            "SELECT golgg_match_id, confidence, mapped_by FROM golgg_match_mappings WHERE canonical_match_id = ?",
            (101,),
        ).fetchone()
        audit = connection.execute(
            "SELECT decision, reason, operator FROM mapping_review_decisions WHERE canonical_match_id = ?",
            (101,),
        ).fetchone()
        canonical = connection.execute(
            "SELECT status, winner_side FROM canonical_matches WHERE id = ?",
            (101,),
        ).fetchone()
    assert mapping["golgg_match_id"] == "502"
    assert mapping["confidence"] == 1.0
    assert mapping["mapped_by"] == "manual-reviewed-v1"
    assert audit["decision"] == "replace"
    assert audit["reason"] == "Exact date, teams, and LPL competition verified"
    assert audit["operator"] == "test-operator"
    assert canonical["status"] == "expired"
    assert canonical["winner_side"] is None


def test_review_conflict_rolls_back_mapping_and_audit(client) -> None:
    _seed_review_mapping(with_second_mapping=True)

    response = client.post(
        "/matches/mapping-review/decision",
        json={
            "canonical_match_id": 101,
            "decision": "replace",
            "new_golgg_match_id": "502",
            "reason": "Attempt conflicting replacement for regression test",
            "operator": "test-operator",
        },
        headers={"X-Identity-Review-Token": "test-review-token"},
    )

    assert response.status_code == 409
    with transaction() as connection:
        mapping = connection.execute(
            "SELECT golgg_match_id FROM golgg_match_mappings WHERE canonical_match_id = ?",
            (101,),
        ).fetchone()
        audit_count = connection.execute(
            "SELECT COUNT(*) AS count FROM mapping_review_decisions WHERE canonical_match_id = ?",
            (101,),
        ).fetchone()["count"]
    assert mapping["golgg_match_id"] == "501"
    assert audit_count == 0
