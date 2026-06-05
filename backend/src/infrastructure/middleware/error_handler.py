"""Global exception handlers for FastAPI.

Bắt các AppError và chuyển thành JSON response chuẩn thay vì để
server trả về 500 Internal Server Error mặc định.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.errors import (
    AppError,
    ConfigurationError,
    DataError,
    DatabaseError,
    ModelError,
    TradingError,
)

logger = logging.getLogger(__name__)

# Mapping exception class → HTTP status code
_STATUS_MAP: dict[type[AppError], int] = {
    ConfigurationError: 500,
    DataError: 422,
    ModelError: 500,
    TradingError: 502,
    DatabaseError: 503,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Đăng ký tất cả exception handlers vào FastAPI app."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        status_code = _STATUS_MAP.get(type(exc), 500)
        logger.error("AppError [%s]: %s", type(exc).__name__, exc)
        return JSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "detail": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "An unexpected error occurred.",
            },
        )
