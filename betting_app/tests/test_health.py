"""Tests for the health endpoint."""

from fastapi.testclient import TestClient


class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_method_not_allowed(self, client: TestClient):
        resp = client.post("/health")
        assert resp.status_code == 405
