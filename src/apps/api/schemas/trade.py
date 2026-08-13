"""Pydantic schemas for trade records and MT5 trading operations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response — Trade item
# ---------------------------------------------------------------------------

class TradeOut(BaseModel):
    """Một trade record trả về từ API."""
    id: int
    symbol: str
    timeframe: str
    direction: str
    order_ticket: int | None = None
    position_ticket: int | None = None
    magic_number: int | None = None
    entry_time: datetime
    exit_time: datetime | None = None
    entry_price: float
    exit_price: float | None = None
    volume: float
    stop_loss: float | None = None
    take_profit: float | None = None
    pnl: float | None = None
    pnl_pips: float | None = None
    status: str
    closed: bool
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TradeFilter(BaseModel):
    """Query params để lọc danh sách trades."""
    symbol: str | None = None
    direction: str | None = Field(default=None, description="BUY | SELL")
    status: str | None = Field(default=None, description="open | closed")
    from_date: date | None = None
    to_date: date | None = None


# ---------------------------------------------------------------------------
# Execute Trade
# ---------------------------------------------------------------------------

class ExecuteTradeRequest(BaseModel):
    """Request body để vào lệnh thủ công."""
    signal: str = Field(description="BUY hoặc SELL")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Độ tin cậy tín hiệu (0.0 – 1.0)",
    )


class ExecuteTradeResponse(BaseModel):
    """Kết quả thực hiện lệnh."""
    executed: bool
    message: str


# ---------------------------------------------------------------------------
# MT5 Status
# ---------------------------------------------------------------------------

class MT5StatusResponse(BaseModel):
    """Trạng thái kết nối MT5."""
    connected: bool
    account_info: dict[str, Any] | None = None


class MT5PositionOut(BaseModel):
    """Một vị thế đang mở trên MT5."""
    ticket: int
    type: str = Field(description="BUY | SELL")
    symbol: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float


# ---------------------------------------------------------------------------
# Auto Trading
# ---------------------------------------------------------------------------

class AutoTradeStartRequest(BaseModel):
    """Request body để bật auto trading."""
    interval: float = Field(
        default=0.5,
        ge=0.1,
        le=10.0,
        description="Khoảng cách giữa mỗi lần kiểm tra (giây)",
    )
    model_mode: str = Field(
        default="dual",
        description="dual | single_m5",
    )


class AutoTradeStatusResponse(BaseModel):
    """Trạng thái auto trading bot."""
    running: bool
    interval: float | None = None
    model_mode: str | None = None


# ---------------------------------------------------------------------------
# Trailing Stop Loss
# ---------------------------------------------------------------------------

class TrailingLevelItem(BaseModel):
    """Một mức trailing SL: khi lời >= trigger_pips thì dời SL về sl_pips."""
    trigger_pips: float = Field(description="Số pips lời để kích hoạt mức này")
    sl_pips: float = Field(description="Dời SL về vị trí này (tính từ giá entry)")


class TrailingStartRequest(BaseModel):
    """Request body để bật trailing SL."""
    levels: list[TrailingLevelItem] | None = Field(
        default=None,
        description="Các mức trailing. None = dùng mặc định",
    )


class TrailingStatusResponse(BaseModel):
    """Trạng thái trailing SL."""
    running: bool
    levels: list[TrailingLevelItem] = Field(default_factory=list)
