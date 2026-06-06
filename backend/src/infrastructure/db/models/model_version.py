from __future__ import annotations

from typing import Any

from sqlalchemy import String, Float, Integer, Boolean, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.db.base import Base, TimestampMixin

class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    data_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    scaler_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    train_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lookback: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
