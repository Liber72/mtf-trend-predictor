"""Top-level API router for the FastAPI application.

Gắn tất cả sub-routers từ endpoints/ vào prefix /api/v1.
"""

from fastapi import APIRouter

from src.apps.api.endpoints import market_data, models, predictions, system, trading, ws

api_router = APIRouter(prefix="/api/v1")

# System endpoints: /api/v1/health
api_router.include_router(system.router)

# Market data endpoints: /api/v1/market-data/*
api_router.include_router(market_data.router)

# ML model endpoints: /api/v1/models/*
api_router.include_router(models.router)

# Prediction endpoints: /api/v1/predictions/*
api_router.include_router(predictions.router)

# Trading endpoints: /api/v1/mt5/*, /api/v1/trading/*, /api/v1/trades/*
api_router.include_router(trading.router)

# WebSocket endpoints: /api/v1/ws/*
api_router.include_router(ws.router)

