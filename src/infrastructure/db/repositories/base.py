"""Generic async CRUD repository.

Lớp cơ sở cung cấp các thao tác CRUD (Create, Read, Update, Delete)
chung cho tất cả ORM models.  Các repository cụ thể kế thừa và mở rộng.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic async repository cho một ORM model."""

    def __init__(self, session: AsyncSession, model: type[ModelT]) -> None:
        self._session = session
        self._model = model

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_by_id(self, record_id: int) -> ModelT | None:
        """Lấy 1 record theo primary key."""
        return await self._session.get(self._model, record_id)

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        order_by: Any | None = None,
        filters: list[Any] | None = None,
    ) -> tuple[list[ModelT], int]:
        """Lấy danh sách records có phân trang.

        Returns:
            Tuple (items, total_count).
        """
        stmt = select(self._model)

        if filters:
            for f in filters:
                stmt = stmt.where(f)

        # Total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._session.execute(count_stmt)).scalar_one()

        # Apply ordering
        if order_by is not None:
            stmt = stmt.order_by(order_by)

        # Apply pagination
        stmt = stmt.offset(offset).limit(limit)

        result = await self._session.execute(stmt)
        items = list(result.scalars().all())

        return items, total

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(self, **kwargs: Any) -> ModelT:
        """Tạo mới 1 record và commit."""
        instance = self._model(**kwargs)
        self._session.add(instance)
        await self._session.commit()
        await self._session.refresh(instance)
        return instance

    async def create_many(self, records: list[dict[str, Any]]) -> int:
        """Bulk insert nhiều records. Trả về số records đã tạo."""
        instances = [self._model(**r) for r in records]
        self._session.add_all(instances)
        await self._session.commit()
        return len(instances)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update(self, record_id: int, **kwargs: Any) -> ModelT | None:
        """Cập nhật 1 record theo ID. Trả về None nếu không tìm thấy."""
        instance = await self.get_by_id(record_id)
        if instance is None:
            return None
        for key, value in kwargs.items():
            setattr(instance, key, value)
        await self._session.commit()
        await self._session.refresh(instance)
        return instance

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(self, record_id: int) -> bool:
        """Xoá 1 record theo ID. Trả về True nếu xoá thành công."""
        instance = await self.get_by_id(record_id)
        if instance is None:
            return False
        await self._session.delete(instance)
        await self._session.commit()
        return True
