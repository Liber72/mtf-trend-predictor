"""Market data endpoints — crawl từ MT5, import CSV, liệt kê files.

MetaTrader5 library là synchronous nên các thao tác MT5 được chạy
trong thread pool thông qua ``asyncio.to_thread`` để không block
event loop của FastAPI.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.apps.api.schemas.candle import (
    CrawlRequest,
    CrawlResponse,
    CrawlTimeframeResult,
    DataFileInfo,
    ImportCsvRequest,
    ImportCsvResponse,
)
from src.core.dependencies import get_db
from src.infrastructure.db.repositories.candle_repo import CandleRepository
from src.modules.market_data.crawler import download_xauusd_data
from src.utils.paths import project_root

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market-data", tags=["market-data"])


def _resolve_data_file_path(file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path

    relative_path = Path(file_path)
    root_candidate = project_root() / relative_path
    if root_candidate.is_file():
        return str(root_candidate)

    if relative_path.parts and relative_path.parts[0] == "data":
        return str(root_candidate)

    data_candidate = project_root() / "data" / relative_path
    return str(data_candidate)


# ======================================================================
# POST /market-data/crawl  —  Crawl data mới từ MT5
# ======================================================================

@router.post("/crawl", response_model=CrawlResponse)
async def crawl_from_mt5(
    req: CrawlRequest,
    db: AsyncSession = Depends(get_db)
) -> CrawlResponse:
    """Crawl dữ liệu nến mới từ MetaTrader 5 và lưu vào database.

    MetaTrader5 lib là synchronous nên được chạy trong thread riêng
    để không block server. Sau khi lấy xong, dữ liệu được ghi vào DB.
    """
    results: list[CrawlTimeframeResult] = []
    repo = CandleRepository(db)

    for tf in req.timeframes:
        try:
            # Chạy synchronous crawler trong thread pool
            df = await asyncio.to_thread(
                download_xauusd_data,
                symbol=req.symbol,
                start_date=req.start_date.strftime("%Y-%m-%d"),
                end_date=req.end_date.strftime("%Y-%m-%d"),
                timeframe=tf,
            )

            if df is not None and not df.empty:
                # Chuyển đổi datetime timezone-aware cho PostgreSQL
                df["Time"] = pd.to_datetime(df["Time"], utc=True)
                
                records = []
                for _, row in df.iterrows():
                    records.append(
                        {
                            "symbol": req.symbol,
                            "timeframe": tf,
                            "time": row["Time"],
                            "open": float(row["Open"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "tick_volume": (
                                int(row["TickVolume"]) if "TickVolume" in df.columns and pd.notna(row.get("TickVolume")) else None
                            ),
                            "spread": (
                                int(row["Spread"]) if "Spread" in df.columns and pd.notna(row.get("Spread")) else None
                            ),
                            "real_volume": (
                                int(row["RealVolume"]) if "RealVolume" in df.columns and pd.notna(row.get("RealVolume")) else None
                            ),
                            "source": "mt5_crawl",
                        }
                    )
                
                # Lưu vào DB
                rows_inserted = await repo.bulk_upsert(records)
                
                results.append(
                    CrawlTimeframeResult(
                        timeframe=tf,
                        rows=rows_inserted,
                        status="success",
                    )
                )
                logger.info(
                    "Crawled and inserted %d candles for %s %s into DB", rows_inserted, req.symbol, tf
                )
            else:
                results.append(
                    CrawlTimeframeResult(timeframe=tf, rows=0, status="failed")
                )
        except Exception as e:
            logger.exception("Crawl error for %s %s", req.symbol, tf)
            results.append(
                CrawlTimeframeResult(
                    timeframe=tf, rows=0, status=f"error: {e}"
                )
            )

    return CrawlResponse(symbol=req.symbol, results=results)


# ======================================================================
# POST /market-data/import  —  Import CSV vào database
# ======================================================================

@router.post("/import", response_model=ImportCsvResponse)
async def import_csv(
    req: ImportCsvRequest,
    db: AsyncSession = Depends(get_db),
) -> ImportCsvResponse:
    """Import dữ liệu từ file CSV vào bảng ``candles``.

    File CSV phải nằm trong thư mục project và có các cột:
    ``Time, Open, High, Low, Close, TickVolume, Spread, RealVolume``.
    Các record trùng lặp (cùng symbol + timeframe + time) sẽ bị bỏ qua.
    """
    # Resolve path — hỗ trợ cả absolute và relative (từ project root)
    file_path = _resolve_data_file_path(req.file_path)

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File không tồn tại: {file_path}",
        )

    # Đọc CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Không thể đọc file CSV: {e}",
        )

    # Validate columns
    required_cols = {"Time", "Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"File CSV thiếu các cột: {missing}",
        )

    # Chuyển đổi data thành dạng dict cho bulk insert
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "symbol": req.symbol,
                "timeframe": req.timeframe,
                "time": row["Time"],
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "tick_volume": (
                    int(row["TickVolume"]) if "TickVolume" in df.columns and pd.notna(row.get("TickVolume")) else None
                ),
                "spread": (
                    int(row["Spread"]) if "Spread" in df.columns and pd.notna(row.get("Spread")) else None
                ),
                "real_volume": (
                    int(row["RealVolume"]) if "RealVolume" in df.columns and pd.notna(row.get("RealVolume")) else None
                ),
                "source": "csv",
            }
        )

    repo = CandleRepository(db)
    rows_inserted = await repo.bulk_upsert(records)

    logger.info(
        "Imported %d/%d rows from %s (%s %s)",
        rows_inserted,
        len(records),
        req.file_path,
        req.symbol,
        req.timeframe,
    )

    return ImportCsvResponse(
        rows_imported=rows_inserted,
        symbol=req.symbol,
        timeframe=req.timeframe,
        file_path=req.file_path,
    )


# ======================================================================
# GET /market-data/files  —  Liệt kê CSV files
# ======================================================================

@router.get("/files", response_model=list[DataFileInfo])
async def list_data_files() -> list[DataFileInfo]:
    """Liệt kê tất cả file CSV trong thư mục data của project."""
    base_dir = project_root() / "data"
    files: list[DataFileInfo] = []

    if not base_dir.exists():
        return files

    for filename in sorted(os.listdir(base_dir)):
        if not filename.endswith(".csv"):
            continue
        full_path = os.path.join(str(base_dir), filename)
        size_bytes = os.path.getsize(full_path)
        files.append(
            DataFileInfo(
                filename=filename,
                size_mb=round(size_bytes / (1024 * 1024), 2),
                path=full_path,
            )
        )

    return files
