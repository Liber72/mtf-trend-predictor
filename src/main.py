"""FastAPI bootstrap for the modular monolith."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.apps.api.router import api_router
from src.core.logging import configure_logging
from src.core.settings import ensure_runtime_directories, get_settings
from src.infrastructure.db.session import dispose_engine, get_engine


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_directories(settings)
    configure_logging(settings.log_level)
    get_engine()
    yield
    await dispose_engine()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/health", tags=["system"])
async def root_health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
    }
