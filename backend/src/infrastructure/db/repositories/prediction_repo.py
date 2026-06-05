"""Repository for prediction log records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.models.prediction import Prediction
from src.infrastructure.db.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    """Async CRUD + domain queries for predictions."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Prediction)

    async def list_recent(
        self,
        *,
        model_mode: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Prediction], int]:
        """Lấy danh sách predictions gần nhất, lọc tuỳ chọn."""
        filters: list[Any] = []
        if model_mode:
            filters.append(Prediction.model_mode == model_mode)
        if from_time:
            filters.append(Prediction.predicted_at >= from_time)
        if to_time:
            filters.append(Prediction.predicted_at <= to_time)

        return await self.list(
            offset=offset,
            limit=limit,
            order_by=Prediction.predicted_at.desc(),
            filters=filters if filters else None,
        )
