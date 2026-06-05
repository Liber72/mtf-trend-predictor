"""Async engine and session factory for PostgreSQL."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from src.core.errors import ConfigurationError
from src.core.settings import get_settings
from src.infrastructure.db.base import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        config = get_settings()
        if not config.database_url:
            raise ConfigurationError(
                "DATABASE_URL hoặc bộ biến DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME là bắt buộc"
            )
        _engine = create_async_engine(
            config.database_url,
            echo=config.database_echo,
            pool_pre_ping=True,
            pool_size=config.database_pool_size,
            max_overflow=config.database_max_overflow,
        )
        _session_factory = async_sessionmaker(
            _engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


async def get_async_session() -> AsyncIterator[AsyncSession]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def create_schema() -> None:
    from src.infrastructure.db import models  # noqa: F401

    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
