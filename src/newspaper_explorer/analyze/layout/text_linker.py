"""
Text Linker - Link detections to text content from DataFrames.

This module provides functionality to match any detection type (headlines,
captions, tables, etc.) to OCR text content from pre-parsed DataFrames based
on bounding box overlap with text lines.

Uses the fast DataFrame path - parquet files are already parsed by ALTOParser
with rich metadata and should be loaded once and reused for batch processing.
"""

import logging
from typing import Dict, List, Tuple

import polars as pl

from newspaper_explorer.analyze.layout.schemas import BoundingBox, Detection

logger = logging.getLogger(__name__)


class TextLinker:
    """
    Universal text linker for any detection type.

    Links detection bounding boxes to text content from pre-parsed DataFrames
    by finding overlapping text lines/blocks based on IoU (Intersection over Union).

    Performance: Uses Polars DataFrames for fast filtering. Load parquet once,
    then filter by page_id for each page - much faster than re-parsing XML.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        min_confidence: float = 0.2,
    ):
        """
        Initialize the TextLinker.

        Args:
            overlap_threshold: Minimum IoU for matching (0.0-1.0)
            min_confidence: Minimum detection confidence to consider
        """
        self.overlap_threshold = overlap_threshold
        self.min_confidence = min_confidence

        logger.info(
            f"TextLinker initialized: overlap_threshold={overlap_threshold}, "
            f"min_confidence={min_confidence}"
        )

    def link_detections_to_text(
        self,
        detections: List[Detection],
        lines_df: pl.DataFrame,
        page_id: str,
    ) -> List[Detection]:
        """
        Link detections to OCR text from pre-parsed DataFrame.

        Updates Detection objects in-place with text_content and alto_elements.

        Performance: Load the parquet DataFrame once, then call this method
        for each page. Much faster than re-parsing XML for every page.

        Example:
            >>> from newspaper_explorer.data.loading.loader import DataLoader
            >>> df = DataLoader.load_parquet("data/processed/der_tag/text/der_tag_lines.parquet")
            >>> linker = TextLinker()
            >>> for page_id in page_ids:
            ...     linker.link_detections_to_text(detections, df, page_id)

        Args:
            detections: List of detections to link
            lines_df: Polars DataFrame with parsed ALTO lines (from parquet)
            page_id: Page identifier to filter lines

        Returns:
            List of Detection objects with linked text
        """
        if not detections:
            logger.debug("No detections provided")
            return []

        # Filter detections by confidence
        valid_detections = [d for d in detections if d.confidence >= self.min_confidence]

        if not valid_detections:
            logger.debug("No detections meet confidence threshold")
            return []

        logger.debug(f"Linking {len(valid_detections)} detections to text")

        # Get ALTO lines from DataFrame
        alto_lines = self._lines_from_dataframe(lines_df, page_id)

        if not alto_lines:
            logger.warning(f"No text lines found for page_id: {page_id}")
            return valid_detections

        # Link each detection to text
        linked_count = 0
        for detection in valid_detections:
            matched_text, alto_elements, match_score = self._match_detection_to_text(
                detection, alto_lines
            )

            if matched_text:
                # Update detection in-place
                detection.text_content = matched_text
                detection.alto_elements = alto_elements
                linked_count += 1
                logger.debug(
                    f"Linked {detection.class_name} ({detection.detection_id}): "
                    f"{matched_text[:50]}... (IoU: {match_score:.2f})"
                )

        logger.info(f"Linked text for {linked_count}/{len(valid_detections)} detections")
        return valid_detections

    def _match_detection_to_text(
        self, detection: Detection, alto_lines: List[Dict]
    ) -> Tuple[str, List[str], float]:
        """
        Match a detection to ALTO text lines.

        Args:
            detection: Detection to match
            alto_lines: List of ALTO line dictionaries

        Returns:
            Tuple of (matched_text, alto_element_ids, match_score)
        """
        matched_lines = []
        alto_element_ids = set()
        ious = []

        # Find overlapping lines
        for line in alto_lines:
            line_bbox = BoundingBox(
                x1=line["x"],
                y1=line["y"],
                x2=line["x"] + line["width"],
                y2=line["y"] + line["height"],
            )

            iou = detection.bbox.iou(line_bbox)
            if iou >= self.overlap_threshold:
                matched_lines.append((line, iou))
                if "line_id" in line:
                    alto_element_ids.add(line["line_id"])
                if "text_block_id" in line:
                    alto_element_ids.add(line["text_block_id"])
                ious.append(iou)

        if not matched_lines:
            return "", [], 0.0

        # Sort by vertical position (top to bottom), then horizontal (left to right)
        matched_lines.sort(key=lambda x: (x[0]["y"], x[0]["x"]))

        # Combine text
        text = " ".join(line[0]["text"] for line in matched_lines)

        # Calculate average match score
        match_score = sum(ious) / len(ious) if ious else 0.0

        return text, list(alto_element_ids), match_score

    def _lines_from_dataframe(self, df: pl.DataFrame, page_id: str) -> List[Dict]:
        """
        Extract lines from Polars DataFrame for a specific page.

        Args:
            df: DataFrame with parsed ALTO lines
            page_id: Page identifier to filter

        Returns:
            List of line dictionaries
        """
        # Filter for this page
        page_df = df.filter(pl.col("filename").str.contains(page_id))

        if page_df.is_empty():
            return []

        # Convert to list of dictionaries
        lines = []
        for row in page_df.iter_rows(named=True):
            lines.append(
                {
                    "text": row["text"],
                    "x": row["x"],
                    "y": row["y"],
                    "width": row["width"],
                    "height": row["height"],
                    "line_id": row.get("line_id"),
                    "text_block_id": row.get("text_block_id"),
                }
            )

        return lines
