"""WebSocket endpoints for realtime data streaming."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websockets"])


# ======================================================================
# WS /ws/trades  —  Stream trade logs & auto-trade messages
# ======================================================================

@router.websocket("/trades")
async def websocket_trades(websocket: WebSocket) -> None:
    """Stream realtime trade logs and auto-trade messages.

    Client connects and receives updates every 2 seconds.
    """
    await websocket.accept()
    logger.info("WebSocket connected: /ws/trades")

    try:
        from src.apps.api.endpoints.trading import _get_trader

        trader = _get_trader()
        last_msg_count = 0
        last_trade_count = 0

        while True:
            # Lấy messages và trades mới nhất
            messages = trader.get_auto_messages()
            trades = trader.get_trade_log()

            updates: dict[str, Any] = {}

            if len(messages) > last_msg_count:
                updates["new_messages"] = messages[last_msg_count:]
                last_msg_count = len(messages)

            if len(trades) > last_trade_count:
                updates["new_trades"] = [
                    {**t, "time": t["time"].isoformat()} if "time" in t else t
                    for t in trades[last_trade_count:]
                ]
                last_trade_count = len(trades)

            if updates:
                await websocket.send_json(updates)

            await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: /ws/trades")
    except Exception as e:
        logger.error("WebSocket error (/ws/trades): %s", e)
        try:
            await websocket.close()
        except Exception:
            pass


# ======================================================================
# WS /ws/predictions  —  Stream realtime predictions
# ======================================================================

@router.websocket("/predictions")
async def websocket_predictions(websocket: WebSocket) -> None:
    """Stream realtime prediction status.

    Gửi trạng thái model đang load và kết quả mới nhất.
    """
    await websocket.accept()
    logger.info("WebSocket connected: /ws/predictions")

    try:
        from src.apps.api.endpoints.models import _get_trainer

        # Loop để không chặn ngay, cập nhật mỗi 5 giây
        while True:
            trainer = _get_trainer()
            
            status = {
                "h1_model_loaded": trainer.h1_model is not None,
                "m5_model_loaded": trainer.m5_model is not None,
                "current_mode": trainer.model_mode,
            }

            await websocket.send_json({"type": "status", "data": status})
            await asyncio.sleep(5.0)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: /ws/predictions")
    except Exception as e:
        logger.error("WebSocket error (/ws/predictions): %s", e)
        try:
            await websocket.close()
        except Exception:
            pass
