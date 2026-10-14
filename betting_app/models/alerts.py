"""SQLAlchemy models for Value Bet alerts and notification settings."""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class AlertConfig(Base):
    """Global configuration for automated Value Bet notifications."""

    __tablename__ = "alert_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    min_ev: Mapped[float] = mapped_column(Float, default=0.05, server_default="0.05")
    min_odds: Mapped[float] = mapped_column(Float, default=1.25, server_default="1.25")
    max_odds: Mapped[float | None] = mapped_column(Float, nullable=True, default=12.0)
    cooldown_hours: Mapped[float] = mapped_column(Float, default=6.0, server_default="6.0")

    # Discord Webhook integration
    discord_enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    discord_webhook_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Telegram Bot integration
    telegram_enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    telegram_bot_token: Mapped[str | None] = mapped_column(String(200), nullable=True)
    telegram_chat_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    updated_at: Mapped[str | None] = mapped_column(String(50), nullable=True)


class ValueAlertLog(Base):
    """Audit log of dispatched (or simulated) Value Bet alerts."""

    __tablename__ = "value_alert_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    canonical_match_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    match_label: Mapped[str] = mapped_column(String(255), nullable=False)
    league: Mapped[str | None] = mapped_column(String(100), nullable=True)
    match_start_at: Mapped[str | None] = mapped_column(String(50), nullable=True)
    side: Mapped[str | None] = mapped_column(String(10), nullable=True)
    team_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    bookmaker_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_stake: Mapped[float | None] = mapped_column(Float, nullable=True)

    channels: Mapped[str] = mapped_column(String(100), default="discord")  # e.g. "discord", "telegram", "both"
    status: Mapped[str] = mapped_column(String(50), default="sent")  # sent, failed, simulated, skipped
    message_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
