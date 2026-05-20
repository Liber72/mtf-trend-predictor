"""System endpoints — health check, configuration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.apps.api.schemas.common import HealthResponse
from src.core.dependencies import get_db, get_settings
from src.core.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck(
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Health check chi tiết — bao gồm trạng thái database."""
    db_status = "disconnected"
    try:
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
