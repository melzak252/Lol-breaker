"""Automation run and command models."""

from __future__ import annotations

from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    Text,
    
)
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class AutomationRun(Base):
    __tablename__ = "automation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_type: Mapped[str] = mapped_column(String(50), nullable=False)
    trigger_source: Mapped[str] = mapped_column(String(50), server_default='scheduler')
    status: Mapped[str] = mapped_column(String(50), server_default='running')
    started_at: Mapped[str | None] = mapped_column(String(50))
    finished_at: Mapped[str | None] = mapped_column(String(50))
    interval_seconds: Mapped[int | None] = mapped_column(Integer)
    next_run_at: Mapped[str | None] = mapped_column(String(50))
    host: Mapped[str | None] = mapped_column(String(200))
    pid: Mapped[int | None] = mapped_column(Integer)
    commands_total: Mapped[int] = mapped_column(Integer, server_default="0")
    commands_failed: Mapped[int] = mapped_column(Integer, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)


class AutomationCommand(Base):
    __tablename__ = "automation_commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("automation_runs.id"), nullable=False)
    command: Mapped[str] = mapped_column(String(500), nullable=False)
    started_at: Mapped[str | None] = mapped_column(String(50))
    finished_at: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), server_default='pending')
    exit_code: Mapped[int | None] = mapped_column(Integer)
    output: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
