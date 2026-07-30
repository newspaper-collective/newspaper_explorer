"""
YOLOv11 Document Layout Detection.

Library module for detecting document layout elements using YOLOv11 model
fine-tuned on the DocLayNet dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# isort: off
# Must set environment BEFORE importing ultralytics
from newspaper_explorer.config import external_tools  # noqa: F401

from ultralytics import settings
from ultralytics.models import YOLO
# isort: on

# Update settings to avoid unnecessary downloads
settings.update({"sync": False})

from newspaper_explorer.data.indexing.image_index import ImageIndexer
from newspaper_explorer.data.utils.ids import (
    extract_issue_id_from_page_id,
    generate_detection_id,
    generate_page_id,
    parse_page_id,
)
from newspaper_explorer.models.analysis.layout import BoundingBox, Detection, PageLayout

logger = logging.getLogger(__name__)


# DocLayNet classes detected by the model
DOCLAYNET_CLASSES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]


def _extract_detections_from_result(det_res, page_id: str, image_path: str) -> List[Detection]:
    """
    Extract Detection objects from YOLO result.

    Uses unified ID system with UUID-based detection IDs and foreign keys.
    """
    detections = []

    if det_res.boxes is not None and len(det_res.boxes) > 0:
        boxes = det_res.boxes.xyxy
        confidences = det_res.boxes.conf
        classes = det_res.boxes.cls

        # Convert to numpy if tensors
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
            confidences = confidences.cpu().numpy()
            classes = classes.cpu().numpy()

        # Extract foreign keys from page_id
        source_id = None
        issue_id = None
        try:
            # Parse page_id to extract source and issue
            page_components = parse_page_id(page_id)
            source_id = str(page_components.source) if page_components.source is not None else None
            issue_id = extract_issue_id_from_page_id(page_id)
        except Exception as e:
            logger.debug(f"Could not parse page_id {page_id}: {e}")

        for idx, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            cls_idx = int(cls)
            class_name = (
                DOCLAYNET_CLASSES[cls_idx]
                if cls_idx < len(DOCLAYNET_CLASSES)
                else f"class_{cls_idx}"
            )

            # Generate stable UUID-based detection ID
            detection_id = generate_detection_id(page_id, class_name)

            detection = Detection(
                detection_id=detection_id,
                class_name=class_name,
                confidence=float(conf),
                bbox=BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                ),
                page_id=page_id,
                source_id=source_id,
                issue_id=issue_id,
                image_path=image_path,
            )
            detections.append(detection)

    return detections


class LayoutDetector:
    """
    YOLOv11 Document Layout Detector.

    Detects 11 document structure categories: Caption, Footnote, Formula, List-item,
    Page-footer, Page-header, Picture, Section-header, Table, Text, and Title.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: Union[str, List[int]] = "cuda:0",
        conf_threshold: float = 0.2,
        imgsz: int = 1280,
        batch_size: int = 8,
        source_name: Optional[str] = None,
    ):
        """
        Initialize the LayoutDetector.

        Args:
            model_size: Model size: 'nano', 'small', or 'medium' (default: 'medium')
            device: Device for inference. Options:
                - 'cuda:0' - single GPU
                - [0, 1, 2, 3] - list of GPU IDs (WARNING: not supported, will use first GPU)
                - 'cpu' - CPU only
            conf_threshold: Confidence threshold for detections
            imgsz: Image size for prediction (default: 1280, as per training)
            batch_size: Batch size for inference
            source_name: Source name (e.g., 'der_tag') for proper ID generation.
                        If provided, uses image index to generate globally consistent IDs.

        Note: Parallel image preloading (70% faster) is always enabled.
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.model_size = model_size
        self.source_name = source_name
        self.source_id = None  # Will be set from image indexer
        self.image_index = None

        # Define model files
        model_files = {
            "nano": "yolo11n_doc_layout.pt",
            "small": "yolo11s_doc_layout.pt",
            "medium": "yolo11m_doc_layout.pt",
        }

        # Load model from HuggingFace
        logger.info(f"Loading YOLOv11 {model_size} model from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download

            # Define the local directory to save models
            download_path = Path("./models/layout")
            download_path.mkdir(parents=True, exist_ok=True)

            # Select model file
            selected_model_file = model_files.get(model_size.lower())
            if not selected_model_file:
                raise ValueError(
                    f"Invalid model_size: {model_size}. Choose from: nano, small, medium"
                )

            # Download the model from HuggingFace
            self.model_path = hf_hub_download(
                repo_id="Armaggheddon/yolo11-document-layout",
                filename=selected_model_file,
                repo_type="model",
                local_dir=download_path,
            )
            logger.info("Model loaded successfully from HuggingFace")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from HuggingFace: {e}")

        # Load single model instance
        # Multi-GPU not supported due to YOLO limitations - use largest batch size on one GPU instead
        if isinstance(self.device, list):
            logger.warning(
                f"Multi-GPU list {self.device} provided, but using only {self.device[0]}"
            )
            logger.warning("Recommendation: Use --device cuda:0 with large --batch-size instead")
            self.device = f"cuda:{self.device[0]}"

        self.model = YOLO(self.model_path)

        # Load or create image index if source_name provided
        if self.source_name:
            logger.info(f"Loading image index for source: {self.source_name}")
            indexer = ImageIndexer(self.source_name)

            # Store source_id from indexer (may be ZDB ID or source_name)
            self.source_id = indexer.source_id

            # Try to load existing index
            self.image_index = indexer.load_index()

            # If no index exists, create one
            if self.image_index is None:
                logger.info(f"No image index found for {self.source_name}, creating one...")
                self.image_index = indexer.create_index()
                logger.info(f"Created image index with {len(self.image_index)} images")
            else:
                logger.info(f"Loaded image index with {len(self.image_index)} images")

        logger.info(
            f"LayoutDetector initialized: model={model_size}, device={device}, "
            f"batch_size={batch_size}, conf_threshold={conf_threshold}"
        )

    def _generate_page_id(self, image_path: Path) -> str:
        """
        Generate page_id from image path with fallback to path-based ID.

        If source_name and image index are available, generates proper
        globally consistent IDs using the ID generation system.
        Otherwise falls back to path-based IDs.

        Args:
            image_path: Path to the image file

        Returns:
            page_id string
        """
        # If we have source_id and image index, use proper ID generation
        if self.source_id and self.image_index is not None:
            try:
                # Get relative path from images directory
                parts = image_path.parts
                images_idx = parts.index("images")
                rel_path = Path(*parts[images_idx + 1 :])
                rel_path_str = str(rel_path)

                # Look up in image index
                matches = self.image_index.filter(self.image_index["image_path"] == rel_path_str)

                if len(matches) > 0:
                    row = matches.row(0, named=True)
                    # Generate proper page_id using global ID function with source_id
                    from datetime import datetime

                    # Skip rows with invalid data (None values, malformed data)
                    if not all(
                        [
                            row.get("date"),
                            row.get("issue_number"),
                            row.get("edition"),
                            row.get("page_number"),
                        ]
                    ):
                        logger.debug(f"Skipping image with incomplete metadata: {rel_path_str}")
                        raise ValueError("Incomplete metadata")

                    date = datetime.strptime(row["date"], "%Y-%m-%d")
                    page_id = generate_page_id(
                        source=self.source_id,  # Use source_id (may be ZDB ID)
                        date=date,
                        issue_number=row["issue_number"],
                        edition=row["edition"],
                        page_number=row["page_number"],
                    )
                    return page_id
                else:
                    logger.debug(f"Image not found in index: {rel_path_str}")
            except Exception as e:
                logger.debug(f"Failed to generate page_id from index: {e}")

        # Fallback: Use path-based ID (old method)
        parts = image_path.parts
        try:
            images_idx = parts.index("images")
            # Use everything after "images/" to create ID
            relative_parts = list(parts[images_idx + 1 :])
            # Remove extension from last part
            relative_parts = relative_parts[:-1] + [Path(relative_parts[-1]).stem]
            # Join with underscores
            return "_".join(relative_parts)
        except (ValueError, IndexError):
            # Fallback to stem if structure unexpected
            logger.warning(f"Could not parse path structure for {image_path}, using stem")
            return image_path.stem

    def detect_page(
        self, image_path: Union[str, Path], page_id: Optional[str] = None
    ) -> PageLayout:
        """
        Detect layout elements in a single page image.

        Args:
            image_path: Path to the page image
            page_id: Optional page identifier (if None, generates from path)

        Returns:
            PageLayout with all detections
        """
        image_path = Path(image_path)
        if page_id is None:
            page_id = self._generate_page_id(image_path)

        logger.debug(f"Detecting layout for page: {page_id}")

        # Perform detection
        det_results = self.model.predict(
            str(image_path),
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False,
        )

        # Extract detections
        detections = self._extract_detections(det_results[0], page_id, str(image_path))

        # Create PageLayout
        page_layout = PageLayout(
            page_id=page_id,
            image_path=str(image_path),
            detections=detections,
        )

        logger.debug(f"Detected {len(detections)} elements in {page_id}: {page_layout.counts}")

        return page_layout

    def detect_batch(
        self, image_paths: List[Union[str, Path]], page_ids: Optional[List[str]] = None
    ) -> List[PageLayout]:
        """
        Detect layout elements in multiple page images.

        Args:
            image_paths: List of paths to page images
            page_ids: Optional list of page identifiers (if None, generates from paths)

        Returns:
            List of PageLayout objects
        """
        if page_ids is None:
            page_ids = [self._generate_page_id(Path(p)) for p in image_paths]

        if len(page_ids) != len(image_paths):
            raise ValueError("page_ids must have same length as image_paths")

        logger.info(
            f"Detecting layout for {len(image_paths)} pages in batches of {self.batch_size}"
        )

        all_layouts = []
        failed_images = []  # Track failed images

        # Batch processing with parallel image preloading (70% faster than sequential)
        from concurrent.futures import ThreadPoolExecutor

        import cv2

        def load_image(path):
            """Load image (BGR format for OpenCV/YOLO compatibility)."""
            try:
                img = cv2.imread(str(path))
                if img is None:
                    logger.error(f"Failed to load image (imread returned None): {path}")
                return img
            except Exception as e:
                logger.error(f"Exception while loading image {path}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel image loading (4 threads for I/O)
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Process batches
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i : i + self.batch_size]
                batch_ids = page_ids[i : i + self.batch_size]

                logger.debug(
                    f"Processing batch {i // self.batch_size + 1}/{(len(image_paths) + self.batch_size - 1) // self.batch_size}"
                )

                # Preload images in parallel (overlaps disk I/O with GPU processing)
                # YOLO accepts numpy arrays (BGR format), PIL Images, or file paths
                loaded_images = list(executor.map(load_image, batch_paths))

                # Filter out failed loads
                valid_images = []
                valid_paths = []
                valid_ids = []
                for img, path, page_id in zip(loaded_images, batch_paths, batch_ids):
                    if img is not None:
                        valid_images.append(img)
                        valid_paths.append(path)
                        valid_ids.append(page_id)

                if not valid_images:
                    logger.warning(f"No valid images in batch starting at index {i}")
                    continue

                # Perform batch prediction with pre-loaded images
                try:
                    det_results = self.model.predict(
                        valid_images,  # Pass numpy arrays directly
                        imgsz=self.imgsz,
                        conf=self.conf_threshold,
                        device=self.device,
                        verbose=False,
                    )
                except Exception as e:
                    logger.error(f"Error during prediction for batch starting at index {i}")
                    logger.error(f"Batch images: {[str(p) for p in valid_paths]}")
                    logger.error(f"Exception: {e}")
                    # Try processing images one by one to identify the problematic one
                    logger.info("Attempting to process images individually to identify issue...")
                    for single_img, single_path, single_id in zip(
                        valid_images, valid_paths, valid_ids
                    ):
                        try:
                            single_result = self.model.predict(
                                [single_img],
                                imgsz=self.imgsz,
                                conf=self.conf_threshold,
                                device=self.device,
                                verbose=False,
                            )
                            # Process single result
                            detections = self._extract_detections(
                                single_result[0], single_id, str(single_path)
                            )
                            page_layout = PageLayout(
                                page_id=single_id,
                                image_path=str(single_path),
                                detections=detections,
                            )
                            all_layouts.append(page_layout)
                        except Exception as single_error:
                            logger.error(f"FAILED IMAGE: {single_path}")
                            logger.error(f"Error: {single_error}")
                            failed_images.append((str(single_path), str(single_error)))
                            # Continue with next image
                    continue

                # Process results
                for img_path, page_id, det_res in zip(valid_paths, valid_ids, det_results):
                    detections = self._extract_detections(det_res, page_id, str(img_path))

                    page_layout = PageLayout(
                        page_id=page_id,
                        image_path=str(img_path),
                        detections=detections,
                    )
                    all_layouts.append(page_layout)

        logger.info(f"Completed detection for {len(all_layouts)} pages")

        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images:")
            for img_path, error in failed_images:
                logger.warning(f"  - {img_path}: {error}")

        return all_layouts

    def _extract_detections(self, det_res, page_id: str, image_path: str = "") -> List[Detection]:
        """
        Extract detection information from YOLO result.

        Args:
            det_res: YOLO detection result object
            page_id: Page identifier
            image_path: Path to image

        Returns:
            List of Detection objects
        """
        return _extract_detections_from_result(det_res, page_id, image_path)
