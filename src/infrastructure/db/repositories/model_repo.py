"""Repository for model version records."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.models.model_version import ModelVersion
from src.infrastructure.db.repositories.base import BaseRepository


class ModelVersionRepository(BaseRepository[ModelVersion]):
    """Async CRUD + domain queries for model versions."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ModelVersion)

    async def get_active_by_timeframe(self, timeframe: str) -> ModelVersion | None:
        """Lấy model đang active cho 1 timeframe."""
        stmt = (
            select(ModelVersion)
            .where(
                ModelVersion.timeframe == timeframe,
                ModelVersion.is_active == True,  # noqa: E712
            )
            .order_by(ModelVersion.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_timeframe(
        self,
        timeframe: str | None = None,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[ModelVersion], int]:
        """Lấy danh sách model versions, lọc theo timeframe nếu có."""
        filters: list[Any] = []
        if timeframe:
            filters.append(ModelVersion.timeframe == timeframe)

        return await self.list(
            offset=offset,
            limit=limit,
            order_by=ModelVersion.created_at.desc(),
            filters=filters if filters else None,
        )

    async def deactivate_all(self, timeframe: str) -> None:
        """Deactivate tất cả model versions cho 1 timeframe."""
        stmt = (
            select(ModelVersion)
            .where(
                ModelVersion.timeframe == timeframe,
                ModelVersion.is_active == True,  # noqa: E712
            )
        )
        result = await self._session.execute(stmt)
        for model in result.scalars().all():
            model.is_active = False
        await self._session.commit()
