"""Model management endpoints — train, list, activate.

Training chạy lâu (phút đến giờ) nên được thực hiện trong background
thread thông qua ``asyncio.to_thread`` để không block server.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.apps.api.schemas.common import PaginatedResponse
from src.apps.api.schemas.model import (
    ModelVersionOut,
    TrainMetrics,
    TrainRequest,
    TrainResponse,
)
from src.core.dependencies import get_db
from src.infrastructure.db.repositories.model_repo import ModelVersionRepository
from src.utils.paths import project_root

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# ---------------------------------------------------------------------------
# Singleton Trainer — load 1 lần, tái sử dụng cho mọi request
# ---------------------------------------------------------------------------
_trainer = None


def _get_trainer():
    """Lazy-init Trainer singleton (import TensorFlow nặng, chỉ làm 1 lần)."""
    global _trainer
    if _trainer is None:
        from src.modules.ml.trainer import Trainer

        _trainer = Trainer()
        _trainer.load_models()
        logger.info("Trainer singleton initialised, models loaded")
    return _trainer


# ======================================================================
# POST /models/train  —  Huấn luyện model
# ======================================================================

@router.post("/train", response_model=TrainResponse)
async def train_model(
    req: TrainRequest,
    db: AsyncSession = Depends(get_db),
) -> TrainResponse:
    """Huấn luyện model LSTM cho timeframe chỉ định.

    Training là synchronous (TensorFlow) nên chạy trong thread riêng.
    Sau khi xong, model version được lưu vào database.
    """
    # Validate timeframe
    if req.timeframe not in ("H1", "M5"):
        raise HTTPException(status_code=422, detail="timeframe phải là H1 hoặc M5")

    # Fetch data from Database instead of CSV
    from src.infrastructure.db.repositories.candle_repo import CandleRepository
    import pandas as pd
    
    candle_repo = CandleRepository(db)
    # Lấy toàn bộ nến của timeframe này
    # Ta dùng limit=100000 để lấy đủ dữ liệu train
    db_candles, total = await candle_repo.list_by_symbol_timeframe(
        symbol="XAUUSD", timeframe=req.timeframe, limit=100000
    )
    
    if not db_candles:
        raise HTTPException(
            status_code=404, 
            detail=f"Không có dữ liệu {req.timeframe} trong Database. Vui lòng Crawl Data trước."
        )
    
    # Chuyển đổi sang DataFrame
    df = pd.DataFrame([{
        "Time": c.time,
        "Open": c.open,
        "High": c.high,
        "Low": c.low,
        "Close": c.close,
        "TickVolume": c.tick_volume or 0,
        "Spread": c.spread or 0,
        "RealVolume": c.real_volume or 0
    } for c in db_candles])
    
    # Đảm bảo sort theo thời gian
    df = df.sort_values('Time').reset_index(drop=True)

    trainer = _get_trainer()

    # Chạy training trong thread pool (TensorFlow là sync/CPU-bound)
    try:
        _, metrics = await asyncio.to_thread(
            trainer.train_model,
            timeframe=req.timeframe,
            data_file=None,
            lookback=req.lookback,
            epochs=req.epochs,
            batch_size=req.batch_size,
            train_ratio=req.train_ratio,
            df=df
        )
    except Exception as e:
        logger.exception("Training failed for %s", req.timeframe)
        raise HTTPException(status_code=500, detail=f"Training thất bại: {e}")

    # Lưu model version vào DB
    model_path = os.path.join(
        str(project_root()), "models", f"{req.timeframe.lower()}_model.keras"
    )
    scaler_path = os.path.join(
        str(project_root()), "models", f"{req.timeframe.lower()}_scaler.pkl"
    )

    version_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    repo = ModelVersionRepository(db)

    # Deactivate models cũ cho timeframe này, activate model mới
    await repo.deactivate_all(req.timeframe)
    await repo.create(
        model_name=f"lstm_{req.timeframe.lower()}",
        timeframe=req.timeframe,
        version=version_str,
        data_file=req.data_file,
        artifact_path=model_path,
        scaler_path=scaler_path,
        train_ratio=req.train_ratio,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lookback=req.lookback,
        metrics={
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
        },
        hyperparameters={
            "lookback": req.lookback,
            "epochs": req.epochs,
            "batch_size": req.batch_size,
            "train_ratio": req.train_ratio,
        },
        is_active=True,
    )

    return TrainResponse(
        timeframe=req.timeframe,
        model_path=model_path,
        metrics=TrainMetrics(
            accuracy=metrics.get("accuracy", 0),
            precision=metrics.get("precision", 0),
            recall=metrics.get("recall", 0),
            f1_score=metrics.get("f1_score", 0),
        ),
        message=f"Model {req.timeframe} trained successfully (v{version_str})",
    )


# ======================================================================
# GET /models  —  Danh sách model versions
# ======================================================================

@router.get("", response_model=PaginatedResponse[ModelVersionOut])
async def list_models(
    timeframe: str | None = Query(default=None, description="Lọc theo H1 hoặc M5"),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[ModelVersionOut]:
    """Liệt kê tất cả model versions."""
    repo = ModelVersionRepository(db)
    offset = (page - 1) * size

    items, total = await repo.list_by_timeframe(
        timeframe=timeframe, offset=offset, limit=size
    )

    return PaginatedResponse[ModelVersionOut](
        items=[ModelVersionOut.model_validate(m) for m in items],
        total=total,
        page=page,
        size=size,
        pages=math.ceil(total / size) if total > 0 else 0,
    )


# ======================================================================
# GET /models/{id}  —  Chi tiết model
# ======================================================================

@router.get("/{model_id}", response_model=ModelVersionOut)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
) -> ModelVersionOut:
    """Lấy chi tiết 1 model version."""
    repo = ModelVersionRepository(db)
    model = await repo.get_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return ModelVersionOut.model_validate(model)


# ======================================================================
# PATCH /models/{id}/activate  —  Kích hoạt model
# ======================================================================

@router.patch("/{model_id}/activate", response_model=ModelVersionOut)
async def activate_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
) -> ModelVersionOut:
    """Kích hoạt model version — deactivate các model cùng timeframe khác."""
    repo = ModelVersionRepository(db)
    model = await repo.get_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Deactivate tất cả model cùng timeframe
    await repo.deactivate_all(model.timeframe)

    # Activate model được chọn
    updated = await repo.update(model_id, is_active=True)

    # Reload model trong Trainer singleton
    trainer = _get_trainer()
    trainer.load_models()
    logger.info("Activated model %d (%s) and reloaded trainer", model_id, model.timeframe)

    return ModelVersionOut.model_validate(updated)
