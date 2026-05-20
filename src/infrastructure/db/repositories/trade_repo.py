"""Repository for trade records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.models.trade import Trade
from src.infrastructure.db.repositories.base import BaseRepository


class TradeRepository(BaseRepository[Trade]):
    """Async CRUD + domain queries for trades."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Trade)

    async def list_trades(
        self,
        *,
        symbol: str | None = None,
        direction: str | None = None,
        status: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Trade], int]:
        """Lấy danh sách trades với nhiều filter tuỳ chọn."""
        filters: list[Any] = []
        if symbol:
            filters.append(Trade.symbol == symbol)
        if direction:
            filters.append(Trade.direction == direction)
        if status:
            filters.append(Trade.status == status)
        if from_date:
            filters.append(Trade.entry_time >= from_date)
        if to_date:
            filters.append(Trade.entry_time <= to_date)

        return await self.list(
            offset=offset,
            limit=limit,
            order_by=Trade.entry_time.desc(),
            filters=filters if filters else None,
        )
