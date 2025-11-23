"""
API models for filtering and pagination.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class DateFilter(BaseModel):
    """Date range filter"""

    start_date: Optional[date] = None
    end_date: Optional[date] = None


class PaginationParams(BaseModel):
    """Pagination parameters"""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
