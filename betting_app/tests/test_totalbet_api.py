from __future__ import annotations

import asyncio

from betting_app.scrapers.totalbet_api import TotalbetApiScraper


def _event(uuid: str, *, name_a: str = "Team A", name_b: str = "Team B") -> dict:
    return {
        "uuid": uuid,
        "type": "prematch",
        "status": "active",
        "path": [{"name": "League of Legends"}, {"name": "LPL"}],
        "participants": [{"name": name_a}, {"name": name_b}],
        "start_at": "2026-07-23T09:00:00+00:00",
        "markets": [
            {
                "name": "Zwycięzca meczu",
                "market_type": "prematch",
                "outcomes": [
                    {"sort": 1, "name": name_a, "odds": "1.80"},
                    {"sort": 2, "name": name_b, "odds": "1.90"},
                ],
            }
        ],
    }


def test_totalbet_scraper_fetches_until_empty_page() -> None:
    scraper = TotalbetApiScraper(pages=20, per_page=100)
    calls: list[int] = []

    def fake_fetch(page: int) -> list[dict]:
        calls.append(page)
        if page <= 3:
            return [_event(f"event-{page}", name_a=f"A{page}", name_b=f"B{page}")]
        return []

    scraper.fetch_events_page = fake_fetch  # type: ignore[method-assign]

    snapshots = asyncio.run(scraper.scrape_upcoming_matches())

    assert calls == [1, 2, 3, 4]
    assert [snapshot.raw_team_a for snapshot in snapshots] == ["A1", "A2", "A3"]
    assert scraper.per_page == 100


def test_totalbet_fetch_events_page_uses_configured_per_page(monkeypatch) -> None:
    scraper = TotalbetApiScraper(pages=1, per_page=100)
    requested_urls: list[str] = []

    class DummyResponse:
        def __enter__(self) -> DummyResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req, timeout: int):  # type: ignore[no-untyped-def]
        requested_urls.append(req.full_url)
        return DummyResponse()

    monkeypatch.setattr("betting_app.scrapers.totalbet_api.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "betting_app.scrapers.totalbet_api.json.load",
        lambda response: {"data": {"events": []}},
    )

    assert scraper.fetch_events_page(2) == []
    assert "page=2" in requested_urls[0]
    assert "per_page=100" in requested_urls[0]
