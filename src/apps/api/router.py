"""Top-level API router for the FastAPI application."""

from fastapi import APIRouter

from src.core.constants import APP_NAME, APP_VERSION

api_router = APIRouter(prefix="/api/v1")


@api_router.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
    }
