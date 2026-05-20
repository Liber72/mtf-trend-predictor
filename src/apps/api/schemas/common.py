"""Common schemas shared across all API endpoints.

Bao gồm: pagination, health check, error response.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class PaginationParams(BaseModel):
    """Query parameters cho phân trang."""
    page: int = Field(default=1, ge=1, description="Số trang (bắt đầu từ 1)")
    size: int = Field(default=50, ge=1, le=500, description="Số item mỗi trang")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel, Generic[T]):
    """Response wrapper cho danh sách có phân trang."""
    items: list[T]
    total: int = Field(description="Tổng số records")
    page: int
    size: int
    pages: int = Field(description="Tổng số trang")


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response cho endpoint /health."""
    status: str = Field(examples=["ok"])
    service: str
    environment: str
    version: str
    database: str = Field(
        description="Trạng thái database",
        examples=["connected", "disconnected"],
    )


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Response chuẩn khi có lỗi."""
    error: str = Field(description="Tên loại lỗi", examples=["DataError"])
    detail: str | None = Field(
        default=None,
        description="Mô tả chi tiết lỗi",
    )
