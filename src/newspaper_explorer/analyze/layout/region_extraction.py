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

from newspaper_explorer.analysis.layout.schemas import Detection, PageLayout

logger = logging.getLogger(__name__)


class RegionExtractor:
    """
    Extracts and crops detected regions from newspaper pages.

    Works with any detection type (images, text, headlines, etc.).
    Can filter by region type using the class_name attribute.
    """

    def __init__(self, padding: int = 5):
        """
        Initialize the RegionExtractor.

        Args:
            padding: Padding around cropped regions (pixels)
        """
        self.padding = padding

        logger.info(f"RegionExtractor initialized: padding={padding}")

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

        logger.info(
            f"Extracting {len(detections)} regions from {page_layout.page_id}"
            + (f" (type: {region_type})" if region_type else "")
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load page image
        page_image = cv2.imread(page_layout.image_path)
        if page_image is None:
            logger.error(f"Failed to load image: {page_layout.image_path}")
            return detections

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
