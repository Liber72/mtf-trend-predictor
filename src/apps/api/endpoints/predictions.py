"""Prediction endpoints — realtime predict, prediction history.

Dự đoán realtime sử dụng Trainer singleton (đã load model vào RAM).
Kết quả được lưu vào bảng ``predictions`` trong database.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.apps.api.schemas.common import PaginatedResponse
from src.apps.api.schemas.prediction import (
    CombinedSignal,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    TimeframePrediction,
)
from src.core.dependencies import get_db
from src.infrastructure.db.repositories.prediction_repo import PredictionRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _get_trainer():
    """Tái sử dụng Trainer singleton từ models endpoint."""
    from src.apps.api.endpoints.models import _get_trainer as get_t

    return get_t()


# ======================================================================
# POST /predictions/predict  —  Dự đoán realtime
# ======================================================================

@router.post("/predict", response_model=PredictResponse)
async def predict_realtime(
    req: PredictRequest,
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:
    """Dự đoán tín hiệu BUY/SELL/WAIT từ dữ liệu MT5 realtime.

    Yêu cầu:
    - Models đã được load (qua train hoặc load_models)
    - MT5 terminal đang chạy (để lấy data realtime)
    """
    trainer = _get_trainer()

    # Kiểm tra models đã load chưa
    if req.model_mode == "dual":
        if trainer.h1_model is None or trainer.m5_model is None:
            raise HTTPException(
                status_code=422,
                detail="Chế độ dual yêu cầu cả H1 và M5 model đã load. "
                "Hãy train hoặc load models trước.",
            )
    elif req.model_mode == "single_m5":
        if trainer.m5_model is None:
            raise HTTPException(
                status_code=422,
                detail="Chế độ single_m5 yêu cầu M5 model đã load. "
                "Hãy train hoặc load models trước.",
            )
    else:
        raise HTTPException(
            status_code=422, detail="model_mode phải là 'dual' hoặc 'single_m5'"
        )

    # Lấy data realtime từ MT5 (synchronous) trong thread pool
    try:
        import MetaTrader5 as mt5

        async def _get_mt5_data(timeframe_str: str, count: int = 350):
            """Lấy nến mới nhất từ MT5."""

            def _fetch():
                if not mt5.initialize():
                    raise RuntimeError(f"Không thể kết nối MT5: {mt5.last_error()}")

                from src.modules.market_data.crawler import TIMEFRAME_MAP

                tf = TIMEFRAME_MAP.get(timeframe_str)
                if tf is None:
                    raise ValueError(f"Timeframe không hợp lệ: {timeframe_str}")

                rates = mt5.copy_rates_from_pos("XAUUSD", tf, 0, count)
                if rates is None or len(rates) == 0:
                    raise RuntimeError(
                        f"Không có data MT5 cho {timeframe_str}: {mt5.last_error()}"
                    )

                import pandas as pd

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.columns = [
                    "Time", "Open", "High", "Low", "Close",
                    "TickVolume", "Spread", "RealVolume",
                ]
                return df

            return await asyncio.to_thread(_fetch)

        # Lấy data
        m5_df = await _get_mt5_data("M5")
        h1_df = None
        if req.model_mode == "dual":
            h1_df = await _get_mt5_data("H1")

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="MetaTrader5 library không khả dụng trên server này",
        )
    except Exception as e:
        logger.exception("Lỗi lấy data MT5")
        raise HTTPException(status_code=503, detail=f"Lỗi MT5: {e}")

    # Chạy prediction (TensorFlow inference) trong thread pool
    try:
        results = await asyncio.to_thread(
            trainer.predict, h1_df, m5_df, req.model_mode
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction thất bại: {e}")

    # Build response
    h1_pred = None
    m5_pred = None

    if "H1" in results and "direction" in results["H1"]:
        h1_pred = TimeframePrediction(
            direction=results["H1"]["direction"],
            probability=results["H1"]["probability"],
        )

    if "M5" in results and "direction" in results["M5"]:
        m5_pred = TimeframePrediction(
            direction=results["M5"]["direction"],
            probability=results["M5"]["probability"],
        )

    combined_data = results.get("combined", {})
    combined = CombinedSignal(
        signal=combined_data.get("signal", "WAIT"),
        confidence=combined_data.get("confidence"),
        reason=combined_data.get("reason"),
    )

    # Lưu prediction vào database
    try:
        repo = PredictionRepository(db)
        await repo.create(
            symbol="XAUUSD",
            timeframe="M5",
            model_mode=req.model_mode,
            predicted_at=datetime.now(timezone.utc),
            h1_direction=h1_pred.direction if h1_pred else None,
            h1_probability=h1_pred.probability if h1_pred else None,
            m5_direction=m5_pred.direction if m5_pred else None,
            m5_probability=m5_pred.probability if m5_pred else None,
            combined_signal=combined.signal,
            combined_confidence=combined.confidence,
            trade_executed=False,
            reason=combined.reason,
            extra={},
        )
    except Exception:
        logger.exception("Failed to save prediction to DB (non-critical)")

    return PredictResponse(
        h1=h1_pred,
        m5=m5_pred,
        combined=combined,
        model_mode=req.model_mode,
    )


# ======================================================================
# GET /predictions  —  Lịch sử predictions
# ======================================================================

@router.get("", response_model=PaginatedResponse[PredictionOut])
async def list_predictions(
    model_mode: str | None = Query(default=None, description="Lọc theo dual | single_m5"),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[PredictionOut]:
    """Lịch sử các dự đoán đã thực hiện."""
    repo = PredictionRepository(db)
    offset = (page - 1) * size

    items, total = await repo.list_recent(
        model_mode=model_mode, offset=offset, limit=size
    )

    return PaginatedResponse[PredictionOut](
        items=[PredictionOut.model_validate(p) for p in items],
        total=total,
        page=page,
        size=size,
        pages=math.ceil(total / size) if total > 0 else 0,
    )
