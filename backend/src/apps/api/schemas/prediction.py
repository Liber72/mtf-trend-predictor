"""Pydantic schemas for prediction records and realtime predict."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response — Prediction history item
# ---------------------------------------------------------------------------

class PredictionOut(BaseModel):
    """Một prediction record trả về từ API."""
    id: int
    symbol: str
    timeframe: str
    model_mode: str
    predicted_at: datetime
    h1_direction: str | None = None
    h1_probability: float | None = None
    m5_direction: str | None = None
    m5_probability: float | None = None
    combined_signal: str | None = None
    combined_confidence: float | None = None
    trade_executed: bool
    reason: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Realtime Predict — Request / Response
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body để dự đoán realtime."""
    model_mode: str = Field(
        default="dual",
        description="dual | single_m5",
    )


class TimeframePrediction(BaseModel):
    """Kết quả dự đoán cho 1 timeframe."""
    direction: str = Field(description="UP | DOWN")
    probability: float


class CombinedSignal(BaseModel):
    """Tín hiệu tổng hợp từ các timeframes."""
    signal: str = Field(description="BUY | SELL | WAIT")
    confidence: float | None = None
    reason: str | None = None


class PredictResponse(BaseModel):
    """Kết quả dự đoán realtime."""
    h1: TimeframePrediction | None = None
    m5: TimeframePrediction | None = None
    combined: CombinedSignal
    model_mode: str
