"""
Text Linker - Link detections to text content from DataFrames.

This module provides functionality to match any detection type (headlines,
captions, tables, etc.) to OCR text content from pre-parsed DataFrames based
on bounding box overlap with text lines.

Uses the fast DataFrame path - parquet files are already parsed by ALTOParser
with rich metadata and should be loaded once and reused for batch processing.

IMPORTANT: Requires an image index to correctly scale ALTO coordinates (which
are in original high-resolution image space) to layout detection coordinates
(which are in downsampled image space).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

from newspaper_explorer.data.indexing.image_index import ImageIndexer
from newspaper_explorer.models.analysis.layout import BoundingBox, Detection

logger = logging.getLogger(__name__)


class TextLinker:
    """
    Universal text linker for any detection type.

    Links detection bounding boxes to text content from pre-parsed DataFrames
    by finding overlapping text lines/blocks based on IoU (Intersection over Union).

    Handles coordinate system differences:
    - Layout detections use downsampled image coordinates (from YOLO on actual images)
    - ALTO text lines use ALTO coordinate space (typically 2000-4000px)
    - Requires image index to get ALTO dimensions and calculate scale factors

    Performance: Uses Polars DataFrames for fast filtering. Load parquet once,
    then filter by page_id for each page - much faster than re-parsing XML.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        min_confidence: float = 0.2,
        source_name: Optional[str] = None,
        image_index: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize the TextLinker.

        Args:
            overlap_threshold: Minimum IoU for matching (0.0-1.0)
            min_confidence: Minimum detection confidence to consider
            source_name: Source name for loading image index (e.g., "der_tag")
            image_index: Pre-loaded image index DataFrame (if not provided, will load from source_name)
        """
        self.overlap_threshold = overlap_threshold
        self.min_confidence = min_confidence

        # Load image index for coordinate scaling
        if image_index is not None:
            self.image_index = image_index
        elif source_name:
            indexer = ImageIndexer(source_name)
            self.image_index = indexer.load_index()
            if self.image_index is None:
                logger.warning(
                    f"Image index not found for {source_name}. "
                    f"Run: newspaper-explorer data index-images --source {source_name}"
                )
        else:
            self.image_index = None
            logger.warning(
                "No image index provided. Coordinate scaling will be skipped. "
                "This may result in incorrect text matching!"
            )

        logger.info(
            f"TextLinker initialized: overlap_threshold={overlap_threshold}, "
            f"min_confidence={min_confidence}, "
            f"image_index={'loaded' if self.image_index is not None else 'not available'}"
        )

    def link_detections_to_text(
        self,
        detections: List[Detection],
        lines_df: pl.DataFrame,
        page_id: str,
        layout_width: Optional[int] = None,
        layout_height: Optional[int] = None,
    ) -> List[Detection]:
        """
        Link detections to OCR text from pre-parsed DataFrame.

        Updates Detection objects in-place with text_content and alto_elements.

        Performance: Load the parquet DataFrame once, then call this method
        for each page. Much faster than re-parsing XML for every page.

        IMPORTANT: For accurate matching, provide layout_width and layout_height
        (the dimensions of the image used for layout detection). If not provided,
        will attempt to infer from detection bounding boxes, but this is less reliable.

        Example:
            >>> from newspaper_explorer.data.loading.loader import DataLoader
            >>> from newspaper_explorer.data.indexing.image_index import ImageIndexer
            >>> df = DataLoader.load_parquet("data/processed/der_tag/text/der_tag_lines.parquet")
            >>> linker = TextLinker(source_name="der_tag")
            >>> for page_id in page_ids:
            ...     linker.link_detections_to_text(
            ...         detections, df, page_id,
            ...         layout_width=2400, layout_height=3499
            ...     )

        Args:
            detections: List of detections to link
            lines_df: Polars DataFrame with parsed ALTO lines (from parquet)
            page_id: Page identifier to filter lines
            layout_width: Width of image used for layout detection
            layout_height: Height of image used for layout detection

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

        # Get scale factors if we have the necessary information
        scale_x, scale_y = 1.0, 1.0
        if layout_width and layout_height and valid_detections[0].image_path:
            scale_x, scale_y = self._get_scale_factors(
                valid_detections[0].image_path, layout_width, layout_height
            )
        elif self.image_index is not None:
            logger.warning(
                "layout_width/layout_height not provided. Coordinate scaling may be inaccurate."
            )

        # Link each detection to text
        linked_count = 0
        for detection in valid_detections:
            matched_text, alto_elements, match_score = self._match_detection_to_text(
                detection, alto_lines, scale_x, scale_y
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

    def _get_scale_factors(
        self, image_path: str, layout_width: int, layout_height: int
    ) -> Tuple[float, float]:
        """
        Get scale factors to convert ALTO coordinates to layout detection coordinates.

        ALTO coordinates are in original high-resolution image space.
        Layout detections are in downsampled image space.

        Args:
            image_path: Path to the image used for layout detection
            layout_width: Width of the image used for layout detection
            layout_height: Height of the image used for layout detection

        Returns:
            Tuple of (scale_x, scale_y) factors
        """
        if self.image_index is None:
            logger.warning("No image index available, using 1.0 scale factors")
            return 1.0, 1.0

        # Extract relative path from image_path if it's absolute
        if image_path.startswith("/"):
            # Convert to relative path
            path_obj = Path(image_path)
            # Find the part after "images/"
            parts = path_obj.parts
            try:
                images_idx = parts.index("images")
                rel_path = "/".join(parts[images_idx + 1 :])
            except (ValueError, IndexError):
                logger.warning(f"Could not extract relative path from {image_path}")
                return 1.0, 1.0
        else:
            rel_path = image_path

        # Look up original dimensions from image index
        img_data = self.image_index.filter(pl.col("image_path") == rel_path)

        if img_data.is_empty():
            logger.warning(f"Image not found in index: {rel_path}")
            return 1.0, 1.0

        orig_width = img_data["alto_width"][0]
        orig_height = img_data["alto_height"][0]

        if orig_width is None or orig_height is None:
            logger.warning(f"No ALTO dimensions in index for: {rel_path}")
            return 1.0, 1.0

        # Calculate scale factors: layout_space / original_space
        scale_x = layout_width / orig_width
        scale_y = layout_height / orig_height

        logger.debug(
            f"Scale factors for {rel_path}: {scale_x:.4f}x{scale_y:.4f} "
            f"(layout: {layout_width}x{layout_height}, original: {orig_width}x{orig_height})"
        )

        return scale_x, scale_y

    def _match_detection_to_text(
        self, detection: Detection, alto_lines: List[Dict], scale_x: float, scale_y: float
    ) -> Tuple[str, List[str], float]:
        """
        Match a detection to ALTO text lines with coordinate scaling.

        Scales ALTO coordinates (original high-res) to layout detection space (downsampled).

        Args:
            detection: Detection to match (in layout detection coordinate space)
            alto_lines: List of ALTO line dictionaries (in original image coordinate space)
            scale_x: Scale factor for x-coordinates (layout_width / original_width)
            scale_y: Scale factor for y-coordinates (layout_height / original_height)

        Returns:
            Tuple of (matched_text, alto_element_ids, match_score)
        """
        matched_lines = []
        alto_element_ids = set()
        ious = []

        # Find overlapping lines
        for line in alto_lines:
            # Scale ALTO coordinates to layout detection space
            scaled_x = line["x"] * scale_x
            scaled_y = line["y"] * scale_y
            scaled_width = line["width"] * scale_x
            scaled_height = line["height"] * scale_y

            line_bbox = BoundingBox(
                x1=scaled_x,
                y1=scaled_y,
                x2=scaled_x + scaled_width,
                y2=scaled_y + scaled_height,
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
            df: DataFrame with parsed ALTO lines (must have page_id column)
            page_id: Page identifier to filter

        Returns:
            List of line dictionaries
        """
        # Filter for this page using page_id column
        page_df = df.filter(pl.col("page_id") == page_id)

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
