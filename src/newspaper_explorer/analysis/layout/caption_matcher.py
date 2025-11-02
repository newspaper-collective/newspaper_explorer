"""
Caption Matcher - Match captions to images based on spatial proximity.

This module handles the spatial relationship between caption detections and
image detections, finding which captions belong to which images based on
distance and position.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple

from newspaper_explorer.analysis.layout.schemas import Detection, BoundingBox

logger = logging.getLogger(__name__)


class CaptionMatcher:
    """
    Matches caption detections to image detections based on spatial proximity.

    This is separate from ALTO text linking - it focuses purely on the spatial
    relationship between detected caption regions and image regions.
    """

    def __init__(
        self,
        search_radius: int = 150,
        caption_position: str = "below",
    ):
        """
        Initialize the CaptionMatcher.

        Args:
            search_radius: Maximum distance to search for captions (pixels)
            caption_position: Where to look for captions relative to images:
                - 'below': Caption must be below image (most common)
                - 'above': Caption must be above image
                - 'both': Caption can be above or below
        """
        self.search_radius = search_radius
        self.caption_position = caption_position

        logger.info(
            f"CaptionMatcher initialized: search_radius={search_radius}, "
            f"caption_position={caption_position}"
        )

    def match_captions_to_images(
        self, images: List[Detection], captions: List[Detection]
    ) -> List[Tuple[Detection, Optional[Detection]]]:
        """
        Match caption detections to image detections.

        Args:
            images: List of image detections
            captions: List of caption detections (should have text_content from ALTO)

        Returns:
            List of (image, caption) tuples. Caption is None if no match found.
        """
        if not captions:
            logger.debug("No captions detected on page")
            return [(img, None) for img in images]

        logger.debug(f"Matching {len(captions)} captions to {len(images)} images")

        matches = []

        for img_det in images:
            # Find nearest caption
            best_caption = None
            best_distance = float("inf")

            for cap_det in captions:
                # Check position constraint first
                if not self._is_valid_caption_position(img_det.bbox, cap_det.bbox):
                    continue

                # Calculate distance
                distance = self._calculate_distance(img_det.bbox, cap_det.bbox)

                if distance < best_distance and distance <= self.search_radius:
                    best_distance = distance
                    best_caption = cap_det

            if best_caption:
                logger.debug(
                    f"Matched caption to image {img_det.detection_id}: "
                    f"{best_caption.text_content[:50] if best_caption.text_content else 'No text'}... "
                    f"(distance: {best_distance:.1f}px)"
                )

            matches.append((img_det, best_caption))

        matched_count = sum(1 for _, cap in matches if cap is not None)
        logger.info(f"Matched {matched_count}/{len(images)} images to captions")

        return matches

    def _calculate_distance(self, image_bbox: BoundingBox, caption_bbox: BoundingBox) -> float:
        """
        Calculate distance between image and caption bounding boxes.

        Uses center-to-center Euclidean distance.

        Args:
            image_bbox: Image bounding box
            caption_bbox: Caption bounding box

        Returns:
            Distance in pixels
        """
        dx = image_bbox.center_x - caption_bbox.center_x
        dy = image_bbox.center_y - caption_bbox.center_y
        return np.sqrt(dx**2 + dy**2)

    def _is_valid_caption_position(
        self, image_bbox: BoundingBox, caption_bbox: BoundingBox
    ) -> bool:
        """
        Check if caption is in valid position relative to image.

        Args:
            image_bbox: Image bounding box
            caption_bbox: Caption bounding box

        Returns:
            True if position is valid according to caption_position setting
        """
        if self.caption_position == "below":
            # Caption should be below image
            return caption_bbox.y1 > image_bbox.y2
        elif self.caption_position == "above":
            # Caption should be above image
            return caption_bbox.y2 < image_bbox.y1
        else:  # 'both'
            return True

    def apply_caption_matches(
        self, matches: List[Tuple[Detection, Optional[Detection]]]
    ) -> List[Detection]:
        """
        Apply caption information to image detections.

        Creates new Detection objects with caption information attached.

        Args:
            matches: List of (image, caption) tuples from match_captions_to_images()

        Returns:
            List of image Detection objects with caption info attached
        """
        images_with_captions = []

        for img_det, cap_det in matches:
            if cap_det:
                # Create copy with caption
                img_with_caption = Detection(
                    detection_id=img_det.detection_id,
                    class_name=img_det.class_name,
                    confidence=img_det.confidence,
                    bbox=img_det.bbox,
                    page_id=img_det.page_id,
                    image_path=img_det.image_path,
                    caption=cap_det,
                    caption_text=cap_det.text_content,
                )
                images_with_captions.append(img_with_caption)
            else:
                # No caption found
                images_with_captions.append(img_det)

        return images_with_captions
