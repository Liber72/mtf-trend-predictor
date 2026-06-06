from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import String, DateTime, Float, Integer, Boolean, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.db.base import Base, TimestampMixin

class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    model_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    h1_direction: Mapped[str | None] = mapped_column(String(8), nullable=True)
    h1_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    m5_direction: Mapped[str | None] = mapped_column(String(8), nullable=True)
    m5_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    combined_signal: Mapped[str | None] = mapped_column(String(8), nullable=True)
    combined_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    trade_executed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
