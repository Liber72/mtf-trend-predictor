"""FastAPI dependency injection providers.

Cung cấp các dependency dùng chung cho tất cả API endpoints thông qua
FastAPI ``Depends()``.  Mỗi request sẽ nhận một AsyncSession riêng biệt
và session sẽ tự động đóng khi request kết thúc.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.settings import Settings, get_settings as _get_settings
from src.infrastructure.db.session import get_session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield một async DB session cho mỗi request, tự đóng khi xong."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


def get_settings() -> Settings:
    """Trả về cached Settings instance."""
    return _get_settings()
