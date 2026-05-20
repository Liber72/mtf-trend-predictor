"""Repository for candle (market data) records.

Kế thừa BaseRepository và thêm các phương thức chuyên biệt cho
import CSV bulk và truy vấn candle theo symbol/timeframe.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.models.candle import Candle
from src.infrastructure.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class CandleRepository(BaseRepository[Candle]):
    """Async CRUD + domain queries for candles."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Candle)

    # ------------------------------------------------------------------
    # Bulk import  (upsert — bỏ qua duplicate theo unique constraint)
    # ------------------------------------------------------------------

    async def bulk_upsert(
        self,
        records: list[dict[str, Any]],
        *,
        batch_size: int = 1000,
    ) -> int:
        """Insert nhiều candles, bỏ qua nếu đã tồn tại (ON CONFLICT DO NOTHING).

        Chia thành batches để tránh quá tải memory.
        Trả về tổng số rows inserted.
        """
        total_inserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = (
                pg_insert(Candle)
                .values(batch)
                .on_conflict_do_nothing(
                    constraint="uq_candles_symbol_timeframe_time",
                )
            )
            result = await self._session.execute(stmt)
            total_inserted += result.rowcount  # type: ignore[union-attr]

        await self._session.commit()
        return total_inserted

    # ------------------------------------------------------------------
    # Domain queries
    # ------------------------------------------------------------------

    async def list_by_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        *,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Candle], int]:
        """Lấy candles lọc theo symbol, timeframe, khoảng thời gian."""
        filters: list[Any] = [
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
        ]
        if from_time:
            filters.append(Candle.time >= from_time)
        if to_time:
            filters.append(Candle.time <= to_time)

        return await self.list(
            offset=offset,
            limit=limit,
            order_by=Candle.time.desc(),
            filters=filters,
        )

    async def count_by_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Đếm số candles cho 1 symbol + timeframe."""
        from sqlalchemy import func

        stmt = (
            select(func.count())
            .select_from(Candle)
            .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one()
