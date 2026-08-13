"""Trading endpoints — MT5 connection, trade execution, auto trading, trade history.

MetaTrader5 library là synchronous nên tất cả thao tác MT5 được chạy
trong thread pool qua ``asyncio.to_thread``.  MT5Trader singleton được
lưu ở module-level để tái sử dụng.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.apps.api.schemas.common import PaginatedResponse
from src.apps.api.schemas.trade import (
    AutoTradeStartRequest,
    AutoTradeStatusResponse,
    ExecuteTradeRequest,
    ExecuteTradeResponse,
    MT5PositionOut,
    MT5StatusResponse,
    TradeOut,
    TrailingLevelItem,
    TrailingStartRequest,
    TrailingStatusResponse,
)
from src.core.dependencies import get_db
from src.infrastructure.db.repositories.trade_repo import TradeRepository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["trading"])

# ---------------------------------------------------------------------------
# MT5Trader singleton
# ---------------------------------------------------------------------------
_trader = None


def _get_trader():
    """Lazy-init MT5Trader singleton."""
    global _trader
    if _trader is None:
        from src.modules.trading.mt5_trader import MT5Trader

        _trader = MT5Trader()
    return _trader


# ======================================================================
# POST /mt5/connect  —  Kết nối MT5
# ======================================================================

@router.post("/mt5/connect", response_model=MT5StatusResponse)
async def mt5_connect() -> MT5StatusResponse:
    """Kết nối tới MetaTrader 5 terminal."""
    trader = _get_trader()

    if trader.connected:
        account = await asyncio.to_thread(trader.get_account_info)
        return MT5StatusResponse(connected=True, account_info=account)

    success, msg = await asyncio.to_thread(trader.connect)
    if not success:
        raise HTTPException(status_code=503, detail=msg)

    account = await asyncio.to_thread(trader.get_account_info)
    logger.info("MT5 connected: %s", msg)
    return MT5StatusResponse(connected=True, account_info=account)


# ======================================================================
# POST /mt5/disconnect  —  Ngắt kết nối
# ======================================================================

@router.post("/mt5/disconnect", response_model=MT5StatusResponse)
async def mt5_disconnect() -> MT5StatusResponse:
    """Ngắt kết nối MetaTrader 5.

    Tự động dừng auto-trade thread và trailing SL nếu đang chạy.
    """
    trader = _get_trader()

    if not trader.connected:
        return MT5StatusResponse(connected=False)

    # Dừng các threads trước
    if trader._auto_trade_thread_running:
        trader.stop_auto_trade_thread()
    if trader._trailing_thread_running:
        trader.stop_trailing_thread()

    await asyncio.to_thread(trader.disconnect)
    logger.info("MT5 disconnected")
    return MT5StatusResponse(connected=False)


# ======================================================================
# GET /mt5/status  —  Trạng thái kết nối + account info
# ======================================================================

@router.get("/mt5/status", response_model=MT5StatusResponse)
async def mt5_status() -> MT5StatusResponse:
    """Lấy trạng thái kết nối MT5 và thông tin tài khoản."""
    trader = _get_trader()

    if not trader.connected:
        return MT5StatusResponse(connected=False)

    account = await asyncio.to_thread(trader.get_account_info)
    return MT5StatusResponse(connected=True, account_info=account)


# ======================================================================
# GET /mt5/positions  —  Lệnh đang mở
# ======================================================================

@router.get("/mt5/positions", response_model=list[MT5PositionOut])
async def mt5_positions() -> list[MT5PositionOut]:
    """Lấy danh sách vị thế đang mở trên MT5."""
    trader = _get_trader()

    if not trader.connected:
        raise HTTPException(status_code=422, detail="Chưa kết nối MT5")

    positions = await asyncio.to_thread(trader.get_all_positions)
    return [
        MT5PositionOut(
            ticket=p["ticket"],
            type=p["type"],
            symbol=trader.symbol,
            volume=p["volume"],
            price_open=p["price_open"],
            price_current=p["price_current"],
            sl=p["sl"],
            tp=p["tp"],
            profit=p["profit"],
        )
        for p in positions
    ]


# ======================================================================
# POST /trading/execute  —  Vào lệnh thủ công
# ======================================================================

@router.post("/trading/execute", response_model=ExecuteTradeResponse)
async def execute_trade(
    req: ExecuteTradeRequest,
    db: AsyncSession = Depends(get_db),
) -> ExecuteTradeResponse:
    """Thực hiện lệnh giao dịch thủ công trên MT5.

    Ghi lại trade vào database nếu thực hiện thành công.
    """
    trader = _get_trader()

    if not trader.connected:
        raise HTTPException(status_code=422, detail="Chưa kết nối MT5")

    if req.signal not in ("BUY", "SELL"):
        raise HTTPException(status_code=422, detail="signal phải là BUY hoặc SELL")

    executed, msg = await asyncio.to_thread(
        trader.execute_signal, req.signal, req.confidence
    )

    # Lưu vào DB nếu thực hiện thành công
    if executed:
        try:
            repo = TradeRepository(db)
            tick_info = await asyncio.to_thread(trader.get_symbol_info)
            price = tick_info["ask"] if req.signal == "BUY" else tick_info["bid"]

            await repo.create(
                symbol=trader.symbol,
                timeframe="M5",
                direction=req.signal,
                magic_number=trader.magic_number,
                entry_time=datetime.now(timezone.utc),
                entry_price=price,
                volume=trader.lot,
                stop_loss=price - trader.sl_pips * tick_info.get("point", 0.01) * 10
                if req.signal == "BUY"
                else price + trader.sl_pips * tick_info.get("point", 0.01) * 10,
                take_profit=price + trader.tp_pips * tick_info.get("point", 0.01) * 10
                if req.signal == "BUY"
                else price - trader.tp_pips * tick_info.get("point", 0.01) * 10,
                status="open",
                closed=False,
                extra={"confidence": req.confidence, "source": "api_manual"},
            )
        except Exception:
            logger.exception("Failed to save trade to DB (non-critical)")

    return ExecuteTradeResponse(executed=executed, message=msg)


# ======================================================================
# POST /trading/auto/start  —  Bật auto trading
# ======================================================================

@router.post("/trading/auto/start", response_model=AutoTradeStatusResponse)
async def auto_trade_start(
    req: AutoTradeStartRequest,
) -> AutoTradeStatusResponse:
    """Bật auto trading bot.

    Yêu cầu: MT5 đã kết nối, models đã load.
    """
    trader = _get_trader()

    if not trader.connected:
        raise HTTPException(status_code=422, detail="Chưa kết nối MT5")

    if trader._auto_trade_thread_running:
        return AutoTradeStatusResponse(
            running=True,
            interval=trader._auto_trade_interval,
            model_mode=trader.model_mode,
        )

    # Lấy trainer singleton
    from src.apps.api.endpoints.models import _get_trainer

    trainer_instance = _get_trainer()

    trader.start_auto_trade_thread(
        trainer=trainer_instance,
        interval=req.interval,
        model_mode=req.model_mode,
    )
    logger.info(
        "Auto trading started: interval=%.1fs, mode=%s",
        req.interval,
        req.model_mode,
    )

    return AutoTradeStatusResponse(
        running=True,
        interval=req.interval,
        model_mode=req.model_mode,
    )


# ======================================================================
# POST /trading/auto/stop  —  Tắt auto trading
# ======================================================================

@router.post("/trading/auto/stop", response_model=AutoTradeStatusResponse)
async def auto_trade_stop() -> AutoTradeStatusResponse:
    """Tắt auto trading bot."""
    trader = _get_trader()
    trader.stop_auto_trade_thread()
    logger.info("Auto trading stopped")
    return AutoTradeStatusResponse(running=False)


# ======================================================================
# GET /trading/auto/status  —  Trạng thái bot
# ======================================================================

@router.get("/trading/auto/status", response_model=AutoTradeStatusResponse)
async def auto_trade_status() -> AutoTradeStatusResponse:
    """Lấy trạng thái auto trading bot."""
    trader = _get_trader()

    if not trader._auto_trade_thread_running:
        return AutoTradeStatusResponse(running=False)

    return AutoTradeStatusResponse(
        running=True,
        interval=getattr(trader, "_auto_trade_interval", None),
        model_mode=trader.model_mode,
    )


# ======================================================================
# POST /trading/trailing/start  —  Bật trailing SL
# ======================================================================

@router.post("/trading/trailing/start", response_model=TrailingStatusResponse)
async def trailing_start(
    req: TrailingStartRequest | None = None,
) -> TrailingStatusResponse:
    """Bật trailing stop loss.

    Nếu gửi levels thì cập nhật mức trailing mới, không gửi thì dùng mặc định.
    """
    trader = _get_trader()

    if not trader.connected:
        raise HTTPException(status_code=422, detail="Chưa kết nối MT5")

    # Cập nhật levels nếu có
    if req and req.levels:
        trader.trailing_sl_levels = [
            (lv.trigger_pips, lv.sl_pips) for lv in req.levels
        ]

    trader.start_trailing_thread()
    logger.info("Trailing SL started, levels=%s", trader.trailing_sl_levels)

    return TrailingStatusResponse(
        running=True,
        levels=[
            TrailingLevelItem(trigger_pips=t, sl_pips=s)
            for t, s in trader.trailing_sl_levels
        ],
    )


# ======================================================================
# POST /trading/trailing/stop  —  Tắt trailing SL
# ======================================================================

@router.post("/trading/trailing/stop", response_model=TrailingStatusResponse)
async def trailing_stop() -> TrailingStatusResponse:
    """Tắt trailing stop loss."""
    trader = _get_trader()
    trader.stop_trailing_thread()
    logger.info("Trailing SL stopped")
    return TrailingStatusResponse(running=False)


# ======================================================================
# GET /trading/trailing/status  —  Trạng thái trailing SL
# ======================================================================

@router.get("/trading/trailing/status", response_model=TrailingStatusResponse)
async def trailing_status() -> TrailingStatusResponse:
    """Lấy trạng thái trailing SL."""
    trader = _get_trader()

    return TrailingStatusResponse(
        running=trader._trailing_thread_running,
        levels=[
            TrailingLevelItem(trigger_pips=t, sl_pips=s)
            for t, s in trader.trailing_sl_levels
        ],
    )


# ======================================================================
# GET /trades  —  Lịch sử trades
# ======================================================================

@router.get("/trades", response_model=PaginatedResponse[TradeOut])
async def list_trades(
    symbol: str | None = Query(default=None),
    direction: str | None = Query(default=None, description="BUY | SELL"),
    status: str | None = Query(default=None, description="open | closed"),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[TradeOut]:
    """Lịch sử trades đã thực hiện."""
    repo = TradeRepository(db)
    offset = (page - 1) * size

    items, total = await repo.list_trades(
        symbol=symbol,
        direction=direction,
        status=status,
        offset=offset,
        limit=size,
    )

    return PaginatedResponse[TradeOut](
        items=[TradeOut.model_validate(t) for t in items],
        total=total,
        page=page,
        size=size,
        pages=math.ceil(total / size) if total > 0 else 0,
    )


# ======================================================================
# GET /trades/{id}  —  Chi tiết trade
# ======================================================================

@router.get("/trades/{trade_id}", response_model=TradeOut)
async def get_trade(
    trade_id: int,
    db: AsyncSession = Depends(get_db),
) -> TradeOut:
    """Lấy chi tiết 1 trade."""
    repo = TradeRepository(db)
    trade = await repo.get_by_id(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return TradeOut.model_validate(trade)
