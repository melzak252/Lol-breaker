"""Small NoDriver wrapper used by bookmaker scrapers.

NoDriver is imported lazily so the rest of the betting app can be used before
the scraping dependency is installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import inspect
import shutil
import tempfile
from typing import Any

from betting_app.core.config import load_config


class NoDriverClient:
    """Lifecycle helper for a NoDriver browser session."""

    def __init__(self, headless: bool | None = None, debug_dir: str | Path | None = None) -> None:
        cfg = load_config()
        self.headless = cfg.scraper_headless if headless is None else headless
        self.debug_dir = Path(debug_dir) if debug_dir else cfg.debug_dir
        self.browser: Any | None = None
        self._user_data_dir: Path | None = None

    async def __aenter__(self) -> "NoDriverClient":
        """Start browser."""

        try:
            import nodriver as uc
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("NoDriver is not installed. Install it with: pip install nodriver") from exc

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._user_data_dir = Path(tempfile.mkdtemp(prefix="betting-nodriver-", dir="/tmp"))
        browser_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-application-cache",
            "--disk-cache-size=1",
            f"--disk-cache-dir={self._user_data_dir / 'cache'}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        self.browser = await uc.start(
            headless=self.headless,
            sandbox=False,
            user_data_dir=self._user_data_dir,
            browser_args=browser_args,
        )
        return self

    async def __aexit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        """Stop browser."""

        if self.browser is not None:
            try:
                result = self.browser.stop()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=10.0)
            finally:
                self.browser = None
        if self._user_data_dir is not None:
            shutil.rmtree(self._user_data_dir, ignore_errors=True)
            self._user_data_dir = None

    async def open(self, url: str) -> Any:
        """Open a URL in the active browser."""

        if self.browser is None:
            raise RuntimeError("Browser is not started")
        return await self.browser.get(url)

    async def scroll_to_bottom(self, tab: Any, step: int = 500, pause: float = 0.8, passes: int = 2) -> None:
        """Scroll the page to the bottom to trigger lazy-loaded content.

        Scrolls down in *step*-pixel increments, pausing *pause* seconds
        between each scroll to allow the SPA to render newly loaded cards.

        Multiple *passes* help with SPAs that keep appending content as the
        user scrolls — the first pass may reveal sections that themselves
        lazy-load more content below.
        """

        import asyncio

        try:
            for _pass in range(passes):
                last_height = await tab.evaluate("document.body.scrollHeight")
                if last_height is None:
                    return
                last_height = int(last_height)
                current = 0
                stable_count = 0
                while current < last_height:
                    current += step
                    await tab.evaluate(f"window.scrollTo(0, {current})")
                    await asyncio.sleep(pause)
                    new_height = await tab.evaluate("document.body.scrollHeight")
                    if new_height is not None:
                        new_height_int = int(new_height)
                        if new_height_int == last_height:
                            stable_count += 1
                            if stable_count >= 3:
                                break
                        else:
                            stable_count = 0
                        last_height = new_height_int
                # Scroll back to top before next pass so lazy-load triggers again
                if _pass < passes - 1:
                    await tab.evaluate("window.scrollTo(0, 0)")
                    await asyncio.sleep(1.0)
        except Exception:
            return

    async def save_debug_artifacts(self, tab: Any, prefix: str) -> tuple[str | None, str | None]:
        """Best-effort HTML/screenshot debug capture."""

        html_path = self.debug_dir / f"{prefix}.html"
        screenshot_path = self.debug_dir / f"{prefix}.png"
        saved_html: str | None = None
        saved_screenshot: str | None = None
        try:
            html = await tab.get_content()
            html_path.write_text(html, encoding="utf-8")
            saved_html = str(html_path)
        except Exception:
            saved_html = None
        try:
            await tab.save_screenshot(str(screenshot_path))
            saved_screenshot = str(screenshot_path)
        except Exception:
            saved_screenshot = None
        return saved_html, saved_screenshot
