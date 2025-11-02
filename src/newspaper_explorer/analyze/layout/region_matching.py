"""
Proximity Matcher - Match elements based on spatial proximity.

This module provides a general framework for matching any type of detected element
to any other type based on spatial relationships (nearest neighbor with constraints).

Examples:
- Match captions to images
- Match headlines to text blocks
- Match advertisements to related content
"""

import logging
import numpy as np
from typing import List, Optional, Tuple, Literal
import polars as pl

from newspaper_explorer.analyze.layout.schemas import Detection, BoundingBox
from newspaper_explorer.analyze.layout.text_linker import TextLinker

logger = logging.getLogger(__name__)


class ProximityMatcher:
    """
    Matches elements to other elements based on spatial proximity.

    General framework for finding the nearest element of type B
    that is spatially related to element of type A.

    Handles both text extraction (via TextLinker) and spatial matching.
    """

    def __init__(
        self,
        search_radius: int = 150,
        relative_position: Literal["below", "above", "left", "right", "any"] = "below",
        overlap_threshold: float = 0.3,
    ):
        """
        Initialize the ProximityMatcher.

        Args:
            search_radius: Maximum distance to search for matches (pixels)
            relative_position: Spatial constraint for matching:
                - 'below': Match must be below source element
                - 'above': Match must be above source element
                - 'left': Match must be left of source element
                - 'right': Match must be right of source element
                - 'any': No spatial constraint (nearest neighbor)
            overlap_threshold: IoU threshold for text matching via TextLinker
        """
        self.search_radius = search_radius
        self.relative_position = relative_position

        # Initialize text linker for OCR extraction
        self.text_linker = TextLinker(overlap_threshold=overlap_threshold)

        logger.info(
            f"ProximityMatcher initialized: search_radius={search_radius}, "
            f"relative_position={relative_position}, overlap_threshold={overlap_threshold}"
        )

    def match_elements(
        self,
        source_elements: List[Detection],
        target_elements: List[Detection],
        lines_df: Optional[pl.DataFrame] = None,
        page_id: Optional[str] = None,
        extract_text: bool = True,
    ) -> List[Tuple[Detection, Optional[Detection]]]:
        """
        Match target elements to source elements based on spatial proximity.

        Generic matching pipeline: optionally extracts OCR text for targets,
        then matches them spatially to sources using nearest neighbor.

        Args:
            source_elements: Elements to find matches for (e.g., images)
            target_elements: Elements to match to sources (e.g., captions)
            lines_df: Polars DataFrame with parsed lines (for text extraction)
            page_id: Page identifier (for text extraction)
            extract_text: Whether to extract OCR text for target elements

        Returns:
            List of (source, target) tuples. Target is None if no match found.
        """
        if not source_elements:
            logger.debug("No source elements to match")
            return []

        # Step 1: Extract OCR text for target elements (if requested)
        targets_with_text = target_elements
        if extract_text and target_elements and lines_df is not None:
            logger.debug(f"Extracting OCR text for {len(target_elements)} target elements")
            targets_with_text = self.text_linker.link_detections_to_text(
                detections=target_elements,
                lines_df=lines_df,
                page_id=page_id or "unknown",
            )

        # Step 2: Spatially match targets to sources
        if not targets_with_text:
            logger.debug("No target elements to match")
            return [(src, None) for src in source_elements]

        logger.debug(f"Matching {len(targets_with_text)} targets to {len(source_elements)} sources")

        matches = []

        for src_det in source_elements:
            # Find nearest target
            best_target = None
            best_distance = float("inf")

            for tgt_det in targets_with_text:
                # Check position constraint first
                if not self._is_valid_position(src_det.bbox, tgt_det.bbox):
                    continue

                # Calculate distance
                distance = self._calculate_distance(src_det.bbox, tgt_det.bbox)

                if distance < best_distance and distance <= self.search_radius:
                    best_distance = distance
                    best_target = tgt_det

            if best_target:
                logger.debug(
                    f"Matched target to source {src_det.detection_id}: "
                    f"{best_target.text_content[:50] if best_target.text_content else best_target.class_name}... "
                    f"(distance: {best_distance:.1f}px)"
                )

            matches.append((src_det, best_target))

        matched_count = sum(1 for _, tgt in matches if tgt is not None)
        logger.debug(f"Matched {matched_count}/{len(source_elements)} sources to targets")

        return matches

    def _calculate_distance(self, source_bbox: BoundingBox, target_bbox: BoundingBox) -> float:
        """
        Calculate distance between two bounding boxes.

        Uses center-to-center Euclidean distance.

        Args:
            source_bbox: Source element bounding box
            target_bbox: Target element bounding box

        Returns:
            Distance in pixels
        """
        dx = source_bbox.center_x - target_bbox.center_x
        dy = source_bbox.center_y - target_bbox.center_y
        return float(np.sqrt(dx**2 + dy**2))

    def _is_valid_position(self, source_bbox: BoundingBox, target_bbox: BoundingBox) -> bool:
        """
        Check if target is in valid position relative to source.

        Args:
            source_bbox: Source element bounding box
            target_bbox: Target element bounding box

        Returns:
            True if position is valid according to relative_position setting
        """
        if self.relative_position == "below":
            # Target should be below source
            return target_bbox.y1 > source_bbox.y2
        elif self.relative_position == "above":
            # Target should be above source
            return target_bbox.y2 < source_bbox.y1
        elif self.relative_position == "left":
            # Target should be left of source
            return target_bbox.x2 < source_bbox.x1
        elif self.relative_position == "right":
            # Target should be right of source
            return target_bbox.x1 > source_bbox.x2
        else:  # 'any'
            return True

    def apply_matches(
        self,
        matches: List[Tuple[Detection, Optional[Detection]]],
        target_attr: str = "caption",
    ) -> List[Detection]:
        """
        Apply match information to source detections.

        Creates new Detection objects with matched target information attached.

        Args:
            matches: List of (source, target) tuples from match_elements()
            target_attr: Which attribute to populate with target info.
                - "caption": Sets caption and caption_text (for images)
                - Could extend with other attributes in the future

        Returns:
            List of source Detection objects with match info attached
        """
        results = []

        for src_det, tgt_det in matches:
            if tgt_det:
                # Create copy with matched target
                if target_attr == "caption":
                    result = Detection(
                        detection_id=src_det.detection_id,
                        class_name=src_det.class_name,
                        confidence=src_det.confidence,
                        bbox=src_det.bbox,
                        page_id=src_det.page_id,
                        image_path=src_det.image_path,
                        text_content=src_det.text_content,
                        alto_elements=src_det.alto_elements,
                        caption=tgt_det,
                        caption_text=tgt_det.text_content,
                    )
                else:
                    # For other attributes, just copy source as-is
                    result = src_det
                results.append(result)
            else:
                # No match found, keep original
                results.append(src_det)

        return results
