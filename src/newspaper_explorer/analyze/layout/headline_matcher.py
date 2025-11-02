"""
Headline Matcher - Match detected headlines to OCR text from ALTO XML.

This module matches YOLOv11 headline detections to text content in ALTO XML
by finding overlapping text blocks based on bounding box coordinates.

Uses the unified ALTOLinker for OCR text extraction.
"""

import logging
from pathlib import Path
from typing import List, Optional
import polars as pl

from newspaper_explorer.analysis.layout.schemas import (
    Detection,
    Headline,
    PageLayout,
)
from newspaper_explorer.analysis.layout.text_linker import TextLinker

logger = logging.getLogger(__name__)


class HeadlineMatcher:
    """
    Matches detected headlines to OCR text content.

    This is a specialized wrapper around TextLinker for headlines.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        min_confidence: float = 0.2,
    ):
        """
        Initialize the HeadlineMatcher.

        Args:
            overlap_threshold: Minimum IoU for matching (0.0-1.0)
            min_confidence: Minimum detection confidence to consider
        """
        self.text_linker = TextLinker(
            overlap_threshold=overlap_threshold,
            min_confidence=min_confidence,
        )

        logger.info(
            f"HeadlineMatcher initialized: overlap_threshold={overlap_threshold}, "
            f"min_confidence={min_confidence}"
        )

    def match_headlines(
        self,
        page_layout: PageLayout,
        lines_df: pl.DataFrame,
    ) -> List[Headline]:
        """
        Match headline detections to OCR text from pre-parsed DataFrame.

        Args:
            page_layout: PageLayout with detected headlines
            lines_df: Polars DataFrame with parsed lines from parquet

        Returns:
            List of Headline objects with matched text
        """
        if not page_layout.headlines:
            logger.debug(f"No headlines detected in {page_layout.page_id}")
            return []

        logger.debug(f"Matching {len(page_layout.headlines)} headlines in {page_layout.page_id}")

        # Use TextLinker to extract text for headlines
        matched_detections = self.text_linker.link_detections_to_text(
            detections=page_layout.headlines,
            lines_df=lines_df,
            page_id=page_layout.page_id,
        )

        # Convert Detection objects to Headline objects
        headlines = []
        for det in matched_detections:
            if det.text_content:  # Only create Headline if text was matched
                headline = Headline(
                    headline_id=det.detection_id,
                    detection=det,
                    ocr_text=det.text_content,
                    text_block_ids=det.alto_elements,
                    confidence=det.confidence,
                    match_score=1.0,  # TextMatcher doesn't return match_score currently
                    page_id=page_layout.page_id,
                    year=page_layout.year,
                    date=page_layout.date,
                    newspaper_title=page_layout.newspaper_title,
                )
                headlines.append(headline)

        logger.info(
            f"Matched {len(headlines)}/{len(page_layout.headlines)} headlines in {page_layout.page_id}"
        )
        return headlines
