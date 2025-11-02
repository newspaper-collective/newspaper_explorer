"""
Image Extractor - Extract images with matched captions.

This module extracts detected images from newspaper pages and matches them
with nearby caption text.

Uses ALTOLinker to extract OCR text for captions, and CaptionMatcher for
spatial matching of captions to images.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import polars as pl

from newspaper_explorer.analysis.layout.schemas import (
    Detection,
    PageLayout,
    BoundingBox,
)
from newspaper_explorer.analysis.layout.alto_linker import ALTOLinker
from newspaper_explorer.analysis.layout.caption_matcher import CaptionMatcher

logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Extracts images from newspaper pages and matches captions.
    """

    def __init__(
        self,
        caption_search_radius: int = 150,
        caption_position: str = "below",
        padding: int = 5,
        overlap_threshold: float = 0.3,
    ):
        """
        Initialize the ImageExtractor.

        Args:
            caption_search_radius: Max distance to search for captions (pixels)
            caption_position: Where to look for captions ('below', 'above', 'both')
            padding: Padding around cropped images (pixels)
            overlap_threshold: IoU threshold for caption text matching with ALTO
        """
        self.padding = padding

        # Initialize ALTO linker for caption OCR extraction
        self.alto_linker = ALTOLinker(overlap_threshold=overlap_threshold)

        # Initialize caption matcher for spatial matching
        self.caption_matcher = CaptionMatcher(
            search_radius=caption_search_radius,
            caption_position=caption_position,
        )

        logger.info(
            f"ImageExtractor initialized: caption_search_radius={caption_search_radius}, "
            f"caption_position={caption_position}, padding={padding}"
        )

    def extract_images(
        self,
        page_layout: PageLayout,
        output_dir: Path,
        alto_xml_path: Optional[Path] = None,
        lines_df: Optional[pl.DataFrame] = None,
        save_crops: bool = True,
    ) -> List[Detection]:
        """
        Extract images from a page and match captions.

        Args:
            page_layout: PageLayout with detected images
            output_dir: Directory to save cropped images
            alto_xml_path: Optional path to ALTO XML for caption text extraction
            lines_df: Optional Polars DataFrame with parsed lines
            save_crops: Whether to save image crops to disk

        Returns:
            List of Detection objects with caption information
        """
        if not page_layout.images:
            logger.debug(f"No images detected in {page_layout.page_id}")
            return []

        logger.info(f"Extracting {len(page_layout.images)} images from {page_layout.page_id}")

        # Extract OCR text for captions first using ALTO linker
        if page_layout.captions and (alto_xml_path is not None or lines_df is not None):
            logger.debug(f"Extracting OCR text for {len(page_layout.captions)} captions")
            page_layout.captions = self.alto_linker.link_detections_to_text(
                detections=page_layout.captions,
                alto_xml_path=alto_xml_path,
                lines_df=lines_df,
                page_id=page_layout.page_id,
            )

        # Match captions to images spatially using caption matcher
        caption_matches = self.caption_matcher.match_captions_to_images(
            images=page_layout.images,
            captions=page_layout.captions,
        )

        # Apply matches to create images with caption info
        images_with_captions = self.caption_matcher.apply_caption_matches(caption_matches)

        # Extract image crops if requested
        if save_crops:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load page image
            page_image = cv2.imread(page_layout.image_path)
            if page_image is None:
                logger.error(f"Failed to load image: {page_layout.image_path}")
                return images_with_captions

            for img_det in images_with_captions:
                crop_path = self._save_image_crop(
                    page_image,
                    img_det,
                    output_dir,
                    page_layout.page_id,
                )
                img_det.image_path = str(crop_path) if crop_path else None

        logger.info(
            f"Extracted {len(images_with_captions)} images, "
            f"{sum(1 for img in images_with_captions if img.caption_text)} with captions"
        )

        return images_with_captions

    def _save_image_crop(
        self,
        page_image: np.ndarray,
        img_det: Detection,
        output_dir: Path,
        page_id: str,
    ) -> Optional[Path]:
        """
        Save cropped image to disk.

        Args:
            page_image: Full page image
            img_det: Image detection
            output_dir: Output directory
            page_id: Page identifier

        Returns:
            Path to saved crop, or None if failed
        """
        try:
            # Calculate crop coordinates with padding
            x1 = max(0, int(img_det.bbox.x1) - self.padding)
            y1 = max(0, int(img_det.bbox.y1) - self.padding)
            x2 = min(page_image.shape[1], int(img_det.bbox.x2) + self.padding)
            y2 = min(page_image.shape[0], int(img_det.bbox.y2) + self.padding)

            # Extract crop
            crop = page_image[y1:y2, x1:x2]

            if crop.size == 0:
                logger.warning(f"Empty crop for detection {img_det.detection_id}")
                return None

            # Generate filename
            crop_filename = f"{page_id}_{img_det.detection_id}.jpg"
            crop_path = output_dir / crop_filename

            # Save
            cv2.imwrite(str(crop_path), crop)
            logger.debug(f"Saved image crop: {crop_path}")

            return crop_path

        except Exception as e:
            logger.error(f"Failed to save image crop {img_det.detection_id}: {e}")
            return None

    def save_image_metadata(
        self,
        images: List[Detection],
        output_path: Path,
        format: str = "parquet",
    ):
        """
        Save image metadata to file.

        Args:
            images: List of image detections with captions
            output_path: Output file path
            format: Output format ('parquet' or 'json')
        """
        if not images:
            logger.warning("No images to save")
            return

        # Convert to dictionaries
        images_data = []
        for img in images:
            data = {
                "detection_id": img.detection_id,
                "page_id": img.page_id,
                "confidence": img.confidence,
                "bbox_x1": img.bbox.x1,
                "bbox_y1": img.bbox.y1,
                "bbox_x2": img.bbox.x2,
                "bbox_y2": img.bbox.y2,
                "bbox_width": img.bbox.width,
                "bbox_height": img.bbox.height,
                "image_path": img.image_path,
                "caption_text": img.caption_text,
                "has_caption": img.caption_text is not None,
            }
            images_data.append(data)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            df = pl.DataFrame(images_data)
            df.write_parquet(output_path)
            logger.info(f"Saved {len(images)} image metadata to {output_path}")

        elif format == "json":
            import json

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(images_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(images)} image metadata to {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}")
