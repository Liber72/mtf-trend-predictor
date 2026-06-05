"""FastAPI bootstrap for the modular monolith."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.apps.api.router import api_router
from src.core.logging import configure_logging
from src.core.settings import ensure_runtime_directories, get_settings
from src.infrastructure.db.session import dispose_engine, get_engine
from src.infrastructure.middleware.cors import register_cors
from src.infrastructure.middleware.error_handler import register_exception_handlers


settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_directories(settings)
    configure_logging(settings.log_level)
    if settings.database_url:
        get_engine()
    else:
        logger.warning(
            "Database chưa được cấu hình; backend vẫn khởi động nhưng các endpoint cần DB sẽ không hoạt động"
        )
    yield
    await dispose_engine()


TAGS_METADATA = [
    {"name": "system", "description": "System health and configuration."},
    {"name": "market-data", "description": "MT5 data crawling and CSV import operations."},
    {"name": "models", "description": "ML model training and version management."},
    {"name": "predictions", "description": "Realtime ML predictions (Dual-Timeframe)."},
    {"name": "trading", "description": "MT5 connection, auto trading, and trade history."},
    {"name": "websockets", "description": "Realtime streaming for trades and predictions."},
]

app = FastAPI(
    title="AutoTrader MTF Predictor API",
    description="""
    Backend API cho hệ thống giao dịch tự động Multi-Timeframe (M5 & H1).
    Hỗ trợ dự đoán xu hướng thị trường và thực thi giao dịch tự động qua MetaTrader 5.
    """,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware & exception handlers
register_cors(app)
register_exception_handlers(app)

# API routes
app.include_router(api_router)
