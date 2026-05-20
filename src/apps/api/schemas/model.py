"""Pydantic schemas for ML model versions and training."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response — Model version item
# ---------------------------------------------------------------------------

class ModelVersionOut(BaseModel):
    """Một model version trả về từ API."""
    id: int
    model_name: str
    timeframe: str
    version: str
    data_file: str | None = None
    artifact_path: str
    scaler_path: str | None = None
    train_ratio: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    lookback: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Train — Request / Response
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """Request body để huấn luyện model."""
    timeframe: str = Field(description="H1 hoặc M5")
    data_file: str = Field(default="", description="Tên file CSV (để trống nếu lấy từ DB)")
    lookback: int = Field(default=48, ge=12, le=96, description="Số nến nhìn lại")
    epochs: int = Field(default=100, ge=10, le=500)
    batch_size: int = Field(default=32, description="Batch size")
    train_ratio: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Tỷ lệ train (0.8 = 80% train, 20% val)",
    )


class TrainMetrics(BaseModel):
    """Các metrics sau khi train xong."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class TrainResponse(BaseModel):
    """Kết quả huấn luyện model."""
    timeframe: str
    model_path: str
    metrics: TrainMetrics
    message: str
