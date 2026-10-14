"""Tests for /api/system endpoints: bookmakers, system status."""

from fastapi.testclient import TestClient


class TestSystemStatus:
    def test_status_returns_counts(self, client: TestClient):
        resp = client.get("/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "counts" in data
        assert "last_scrape_runs" in data
        assert data["counts"]["canonical_matches"] >= 0


class TestBookmakers:
    def test_list_contains_seeded(self, client: TestClient):
        resp = client.get("/bookmakers")
        assert resp.status_code == 200
        data = resp.json()
        names = [b["name"] for b in data]
        assert "manual" in names
