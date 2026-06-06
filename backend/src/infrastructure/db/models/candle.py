from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, DateTime, Float, Integer, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.db.base import Base, TimestampMixin

class Candle(Base, TimestampMixin):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    tick_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spread: Mapped[int | None] = mapped_column(Integer, nullable=True)
    real_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "time", name="uq_candles_symbol_timeframe_time"),
        Index("ix_candles_symbol_timeframe_time", "symbol", "timeframe", "time"),
        Index("ix_candles_time", "time"),
    )
