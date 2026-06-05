"""System endpoints — health check, configuration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text

from src.apps.api.schemas.common import HealthResponse
from src.core.dependencies import get_settings
from src.core.settings import Settings
from src.infrastructure.db.session import get_session_factory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Health check chi tiết — bao gồm trạng thái database."""
    db_status = "not_configured"
    if settings.database_url:
        db_status = "disconnected"
        try:
            session_factory = get_session_factory()
            async with session_factory() as db:
                await db.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception:
            logger.warning("Database health check failed")

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
        version=settings.app_version,
        database=db_status,
    )
