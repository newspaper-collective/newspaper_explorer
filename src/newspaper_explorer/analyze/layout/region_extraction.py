"""
Region Extractor - Extract and crop detected regions from newspaper pages.

This module handles the physical extraction and cropping of any detected
layout regions (images, text blocks, headlines, etc.) from page images.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import polars as pl

from newspaper_explorer.models.analysis.layout import Detection, PageLayout

logger = logging.getLogger(__name__)


class RegionExtractor:
    """
    Extracts and crops detected regions from newspaper pages.

    Works with any detection type (images, text, headlines, etc.).
    Can filter by region type using the class_name attribute.
    Supports coordinate-based filtering to exclude common areas like headers.
    """

    def __init__(
        self,
        padding: int = 5,
        exclude_top_percent: Optional[float] = None,
        exclude_bottom_percent: Optional[float] = None,
        min_region_height: Optional[int] = None,
        min_region_width: Optional[int] = None,
    ):
        """
        Initialize the RegionExtractor.

        Args:
            padding: Padding around cropped regions (pixels)
            exclude_top_percent: Exclude regions in top X% of page (0-100).
                                Useful for filtering newspaper headers.
            exclude_bottom_percent: Exclude regions in bottom X% of page (0-100).
                                   Useful for filtering page numbers/footers.
            min_region_height: Minimum height (pixels) for regions to extract
            min_region_width: Minimum width (pixels) for regions to extract
        """
        self.padding = padding
        self.exclude_top_percent = exclude_top_percent
        self.exclude_bottom_percent = exclude_bottom_percent
        self.min_region_height = min_region_height
        self.min_region_width = min_region_width

        logger.info(
            f"RegionExtractor initialized: padding={padding}, "
            f"exclude_top={exclude_top_percent}%, exclude_bottom={exclude_bottom_percent}%, "
            f"min_height={min_region_height}, min_width={min_region_width}"
        )

    def _filter_by_coordinates(
        self,
        detections: List[Detection],
        page_height: int,
        page_width: int,
    ) -> List[Detection]:
        """
        Filter detections based on coordinate constraints.

        Args:
            detections: List of detections to filter
            page_height: Height of the page image
            page_width: Width of the page image

        Returns:
            Filtered list of detections
        """
        if not detections:
            return detections

        filtered = []
        excluded_count = {"top": 0, "bottom": 0, "size": 0}

        for detection in detections:
            # Check top exclusion zone (for headers)
            if self.exclude_top_percent is not None:
                top_threshold = (self.exclude_top_percent / 100) * page_height
                if detection.bbox.y1 < top_threshold:
                    excluded_count["top"] += 1
                    logger.debug(
                        f"Excluded {detection.detection_id} (top zone): "
                        f"y1={detection.bbox.y1:.0f} < {top_threshold:.0f}"
                    )
                    continue

            # Check bottom exclusion zone (for footers/page numbers)
            if self.exclude_bottom_percent is not None:
                bottom_threshold = page_height - (self.exclude_bottom_percent / 100) * page_height
                if detection.bbox.y2 > bottom_threshold:
                    excluded_count["bottom"] += 1
                    logger.debug(
                        f"Excluded {detection.detection_id} (bottom zone): "
                        f"y2={detection.bbox.y2:.0f} > {bottom_threshold:.0f}"
                    )
                    continue

            # Check minimum size constraints
            if (
                self.min_region_height is not None
                and detection.bbox.height < self.min_region_height
            ):
                excluded_count["size"] += 1
                logger.debug(
                    f"Excluded {detection.detection_id} (height): "
                    f"{detection.bbox.height:.0f} < {self.min_region_height}"
                )
                continue

            if self.min_region_width is not None and detection.bbox.width < self.min_region_width:
                excluded_count["size"] += 1
                logger.debug(
                    f"Excluded {detection.detection_id} (width): "
                    f"{detection.bbox.width:.0f} < {self.min_region_width}"
                )
                continue

            filtered.append(detection)

        # Log exclusion summary
        total_excluded = sum(excluded_count.values())
        if total_excluded > 0:
            logger.info(
                f"Filtered {total_excluded}/{len(detections)} regions: "
                f"top={excluded_count['top']}, bottom={excluded_count['bottom']}, "
                f"size={excluded_count['size']}"
            )

        return filtered

    def extract_regions(
        self,
        detections: List[Detection],
        page_layout: PageLayout,
        output_dir: Path,
        region_type: Optional[Union[str, List[str]]] = None,
    ) -> List[Detection]:
        """
        Extract and crop detected regions from a page.

        Args:
            detections: List of detections to extract
            page_layout: PageLayout with page information
            output_dir: Directory to save cropped regions
            region_type: Filter by class_name (e.g., "image", "text", "headline").
                        Can be a string or list of strings. If None, extracts all.

        Returns:
            List of Detection objects with image_path populated
        """
        # Filter by region type if specified
        if region_type is not None:
            if isinstance(region_type, str):
                region_type = [region_type]
            detections = [d for d in detections if d.class_name in region_type]

        if not detections:
            logger.debug(
                f"No regions to extract from {page_layout.page_id}"
                + (f" (type: {region_type})" if region_type else "")
            )
            return []

        # Load page image to get dimensions for filtering
        page_image = cv2.imread(page_layout.image_path)
        if page_image is None:
            logger.error(f"Failed to load image: {page_layout.image_path}")
            return detections

        page_height, page_width = page_image.shape[:2]

        # Apply coordinate-based filtering
        detections = self._filter_by_coordinates(detections, page_height, page_width)

        if not detections:
            logger.debug(f"No regions remaining after filtering from {page_layout.page_id}")
            return []

        logger.info(
            f"Extracting {len(detections)} regions from {page_layout.page_id}"
            + (f" (type: {region_type})" if region_type else "")
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Crop and save each region, creating new Detection objects
        extracted_detections = []
        for detection in detections:
            crop_path = self._save_region_crop(
                page_image,
                detection,
                output_dir,
                page_layout.page_id,
            )

            # Create new Detection with image_path set (immutable pattern)
            extracted = Detection(
                detection_id=detection.detection_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                page_id=detection.page_id,
                image_path=str(crop_path) if crop_path else None,
                text_content=detection.text_content,
                alto_elements=detection.alto_elements,
                caption=detection.caption,
                caption_text=detection.caption_text,
            )
            extracted_detections.append(extracted)

        logger.info(f"Extracted {len(extracted_detections)} region crops to {output_dir}")

        return extracted_detections

    def _save_region_crop(
        self,
        page_image: np.ndarray,
        detection: Detection,
        output_dir: Path,
        page_id: str,
    ) -> Optional[Path]:
        """
        Save cropped region to disk.

        Args:
            page_image: Full page image
            detection: Detection to crop
            output_dir: Output directory
            page_id: Page identifier

        Returns:
            Path to saved crop, or None if failed
        """
        try:
            # Calculate crop coordinates with padding
            x1 = max(0, int(detection.bbox.x1) - self.padding)
            y1 = max(0, int(detection.bbox.y1) - self.padding)
            x2 = min(page_image.shape[1], int(detection.bbox.x2) + self.padding)
            y2 = min(page_image.shape[0], int(detection.bbox.y2) + self.padding)

            # Extract crop
            crop = page_image[y1:y2, x1:x2]

            if crop.size == 0:
                logger.warning(f"Empty crop for detection {detection.detection_id}")
                return None

            # Generate filename
            crop_filename = f"{page_id}_{detection.detection_id}.jpg"
            crop_path = output_dir / crop_filename

            # Save
            cv2.imwrite(str(crop_path), crop)
            logger.debug(f"Saved region crop: {crop_path}")

            return crop_path

        except Exception as e:
            logger.error(f"Failed to save region crop {detection.detection_id}: {e}")
            return None

    def save_region_metadata(
        self,
        detections: List[Detection],
        output_path: Path,
        mode: str = "overwrite",
    ):
        """
        Save region metadata to Parquet file.

        Args:
            detections: List of detections with metadata
            output_path: Output file path (should end in .parquet)
            mode: Save mode - "overwrite" (default) or "append"
        """
        if not detections:
            logger.warning("No detections to save")
            return

        # Convert to dictionaries
        detections_data = []
        for det in detections:
            data = {
                "detection_id": det.detection_id,
                "page_id": det.page_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox_x1": det.bbox.x1,
                "bbox_y1": det.bbox.y1,
                "bbox_x2": det.bbox.x2,
                "bbox_y2": det.bbox.y2,
                "bbox_width": det.bbox.width,
                "bbox_height": det.bbox.height,
                "image_path": det.image_path,
                "text_content": det.text_content,
                "caption_text": det.caption_text,
                "has_caption": det.caption_text is not None,
            }
            detections_data.append(data)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create new DataFrame
        new_df = pl.DataFrame(detections_data)

        # Append or overwrite
        if mode == "append" and output_path.exists():
            existing_df = pl.read_parquet(output_path)
            combined_df = pl.concat([existing_df, new_df])
            combined_df.write_parquet(output_path)
            logger.info(
                f"Appended {len(detections)} region metadata to {output_path} "
                f"(total: {len(combined_df)})"
            )
        else:
            new_df.write_parquet(output_path)
            logger.info(f"Saved {len(detections)} region metadata to {output_path}")
