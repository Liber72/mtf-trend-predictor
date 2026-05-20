"""Pydantic schemas for market candle data.

Dùng cho các endpoint crawl data từ MT5 và import CSV.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response — Candle item
# ---------------------------------------------------------------------------

class CandleOut(BaseModel):
    """Một candle trả về từ API."""
    id: int
    symbol: str
    timeframe: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int | None = None
    spread: int | None = None
    real_volume: int | None = None
    source: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Crawl — Request / Response
# ---------------------------------------------------------------------------

class CrawlRequest(BaseModel):
    """Request body để crawl data mới từ MT5."""
    symbol: str = Field(default="XAUUSD", description="Symbol cần crawl")
    timeframes: list[str] = Field(
        default=["M5", "H1"],
        description="Danh sách timeframe cần tải",
        examples=[["M5", "H1"]],
    )
    start_date: date = Field(description="Ngày bắt đầu")
    end_date: date = Field(description="Ngày kết thúc")


class CrawlTimeframeResult(BaseModel):
    """Kết quả crawl cho 1 timeframe."""
    timeframe: str
    rows: int = Field(description="Số nến đã tải")
    status: str = Field(description="success | failed | error: ...")


class CrawlResponse(BaseModel):
    """Kết quả tổng hợp của lệnh crawl."""
    symbol: str
    results: list[CrawlTimeframeResult]


# ---------------------------------------------------------------------------
# Import CSV — Request / Response
# ---------------------------------------------------------------------------

class ImportCsvRequest(BaseModel):
    """Request body để import candles từ CSV vào database."""
    file_path: str = Field(description="Đường dẫn file CSV trong project")
    symbol: str = Field(default="XAUUSD")
    timeframe: str = Field(description="Timeframe của dữ liệu", examples=["M5", "H1"])


class ImportCsvResponse(BaseModel):
    """Kết quả import CSV."""
    rows_imported: int
    symbol: str
    timeframe: str
    file_path: str


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------

class DataFileInfo(BaseModel):
    """Thông tin một file CSV trong project."""
    filename: str
    size_mb: float = Field(description="Kích thước file (MB)")
    path: str
