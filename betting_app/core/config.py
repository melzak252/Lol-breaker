"""Configuration helpers for the betting application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "betting_app.sqlite3"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "data" / "betting_scraper_debug"


@dataclass(frozen=True)
class BettingConfig:
    """Runtime configuration for the local betting manager."""

    db_path: Path = DEFAULT_DB_PATH
    debug_dir: Path = DEFAULT_DEBUG_DIR
    tax_rate: float = 0.12
    min_ev: float = 0.05
    default_bankroll: float = 100.0
    scraper_headless: bool = True
    scraper_timeout_seconds: int = 30


def load_config() -> BettingConfig:
    """Load configuration from environment variables with safe defaults."""

    return BettingConfig(
        db_path=Path(os.getenv("BETTING_APP_DB", str(DEFAULT_DB_PATH))),
        debug_dir=Path(os.getenv("BETTING_APP_DEBUG_DIR", str(DEFAULT_DEBUG_DIR))),
        tax_rate=float(os.getenv("BETTING_APP_TAX_RATE", "0.12")),
        min_ev=float(os.getenv("BETTING_APP_MIN_EV", "0.05")),
        default_bankroll=float(os.getenv("BETTING_APP_BANKROLL", "100.0")),
        scraper_headless=os.getenv("BETTING_APP_HEADLESS", "1") not in {"0", "false", "False"},
        scraper_timeout_seconds=int(os.getenv("BETTING_APP_TIMEOUT", "30")),
    )
