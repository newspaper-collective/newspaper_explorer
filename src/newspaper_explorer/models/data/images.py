"""
Image-related data models.

Models for image validation and METS image references.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ImageValidationResult(BaseModel):
    """Result of image validation check."""

    is_valid: bool
    file_path: Path
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    error: Optional[str] = None


class ImageReference(BaseModel):
    """Reference to an image in METS XML."""

    file_id: str
    url: str
    extension: str = ".jpg"
