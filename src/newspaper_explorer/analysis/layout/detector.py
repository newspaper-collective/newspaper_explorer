"""
YOLOv11 Document Layout Detection.

Library module for detecting document layout elements using YOLOv11 model
fine-tuned on the DocLayNet dataset.
"""

import os
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict
import numpy as np

# Disable Ultralytics settings updates to avoid disk space issues
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).parent / ".ultralytics_cache")
os.environ["ULTRALYTICS_AUTOINSTALL"] = "False"

from ultralytics import YOLO
from ultralytics import settings

# Update settings to avoid unnecessary downloads
settings.update({"sync": False})

from newspaper_explorer.analysis.layout.schemas import Detection, BoundingBox, PageLayout

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


class LayoutDetector:
    """
    YOLOv11 Document Layout Detector.

    Detects 11 document structure categories: Caption, Footnote, Formula, List-item,
    Page-footer, Page-header, Picture, Section-header, Table, Text, and Title.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = "medium",
        device: str = "cuda:0",
        conf_threshold: float = 0.2,
        imgsz: int = 1280,
        batch_size: int = 8,
    ):
        """
        Initialize the LayoutDetector.

        Args:
            model_path: Path to model file. If None, downloads from HuggingFace
            model_size: Model size: 'nano', 'small', or 'medium' (default: 'medium')
            device: Device for inference ('cuda:0', 'cpu', etc.)
            conf_threshold: Confidence threshold for detections
            imgsz: Image size for prediction (default: 1280, as per training)
            batch_size: Batch size for inference (default: 8)
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.model_size = model_size

        # Define model files
        model_files = {
            "nano": "yolo11n_doc_layout.pt",
            "small": "yolo11s_doc_layout.pt",
            "medium": "yolo11m_doc_layout.pt",
        }

        # Load model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from local path: {model_path}")
            self.model = YOLO(model_path)
        else:
            logger.info(f"Loading YOLOv11 {model_size} model from HuggingFace...")
            try:
                from huggingface_hub import hf_hub_download

                # Define the local directory to save models
                download_path = Path("./models")
                download_path.mkdir(exist_ok=True)

                # Select model file
                selected_model_file = model_files.get(model_size.lower())
                if not selected_model_file:
                    raise ValueError(
                        f"Invalid model_size: {model_size}. Choose from: nano, small, medium"
                    )

                # Download the model from HuggingFace
                model_path = hf_hub_download(
                    repo_id="Armaggheddon/yolo11-document-layout",
                    filename=selected_model_file,
                    repo_type="model",
                    local_dir=download_path,
                )

                self.model = YOLO(model_path)
                logger.info("Model loaded successfully from HuggingFace")
            except Exception as e:
                raise RuntimeError(f"Failed to load model from HuggingFace: {e}")

        logger.info(
            f"LayoutDetector initialized: model={model_size}, device={device}, "
            f"batch_size={batch_size}, conf_threshold={conf_threshold}"
        )

    def detect_page(
        self, image_path: Union[str, Path], page_id: Optional[str] = None
    ) -> PageLayout:
        """
        Detect layout elements in a single page image.

        Args:
            image_path: Path to the page image
            page_id: Optional page identifier

        Returns:
            PageLayout with all detections
        """
        image_path = Path(image_path)
        if page_id is None:
            page_id = image_path.stem

        logger.debug(f"Detecting layout for page: {page_id}")

        # Perform detection
        det_results = self.model.predict(
            str(image_path),
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        # Extract detections
        detections = self._extract_detections(det_results[0], page_id)

        # Create PageLayout
        page_layout = PageLayout(
            page_id=page_id,
            image_path=str(image_path),
            detections=detections,
        )

        # Organize by type
        for det in detections:
            cls = det.class_name.lower()
            if "title" in cls or "header" in cls:
                page_layout.headlines.append(det)
            elif "picture" in cls:
                page_layout.images.append(det)
            elif "caption" in cls:
                page_layout.captions.append(det)
            elif "table" in cls:
                page_layout.tables.append(det)
            elif "text" in cls:
                page_layout.text_blocks.append(det)

        logger.debug(
            f"Detected {len(detections)} elements in {page_id}: "
            f"{len(page_layout.headlines)} headlines, {len(page_layout.images)} images, "
            f"{len(page_layout.captions)} captions"
        )

        return page_layout

    def detect_batch(
        self, image_paths: List[Union[str, Path]], page_ids: Optional[List[str]] = None
    ) -> List[PageLayout]:
        """
        Detect layout elements in multiple page images.

        Args:
            image_paths: List of paths to page images
            page_ids: Optional list of page identifiers

        Returns:
            List of PageLayout objects
        """
        if page_ids is None:
            page_ids = [Path(p).stem for p in image_paths]

        if len(page_ids) != len(image_paths):
            raise ValueError("page_ids must have same length as image_paths")

        logger.info(
            f"Detecting layout for {len(image_paths)} pages in batches of {self.batch_size}"
        )

        all_layouts = []

        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_ids = page_ids[i : i + self.batch_size]

            logger.debug(
                f"Processing batch {i // self.batch_size + 1}/{(len(image_paths) + self.batch_size - 1) // self.batch_size}"
            )

            # Perform batch prediction
            det_results = self.model.predict(
                [str(p) for p in batch_paths],
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )

            # Process results
            for img_path, page_id, det_res in zip(batch_paths, batch_ids, det_results):
                detections = self._extract_detections(det_res, page_id)

                # Create PageLayout
                page_layout = PageLayout(
                    page_id=page_id,
                    image_path=str(img_path),
                    detections=detections,
                )

                # Organize by type
                for det in detections:
                    cls = det.class_name.lower()
                    if "title" in cls or "header" in cls:
                        page_layout.headlines.append(det)
                    elif "picture" in cls:
                        page_layout.images.append(det)
                    elif "caption" in cls:
                        page_layout.captions.append(det)
                    elif "table" in cls:
                        page_layout.tables.append(det)
                    elif "text" in cls:
                        page_layout.text_blocks.append(det)

                all_layouts.append(page_layout)

        logger.info(f"Completed detection for {len(all_layouts)} pages")
        return all_layouts

    def _extract_detections(self, det_res, page_id: str) -> List[Detection]:
        """
        Extract detection information from YOLO result.

        Args:
            det_res: YOLO detection result object
            page_id: Page identifier

        Returns:
            List of Detection objects
        """
        detections = []

        if det_res.boxes is None or len(det_res.boxes) == 0:
            return detections

        # Extract boxes, classes, and confidence scores
        boxes = det_res.boxes.xyxy.cpu().numpy()
        classes = det_res.boxes.cls.cpu().numpy()
        confidences = det_res.boxes.conf.cpu().numpy()
        names = det_res.names

        for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box

            detection = Detection(
                detection_id=f"{page_id}_det_{idx}",
                class_name=names[int(cls)],
                confidence=float(conf),
                bbox=BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                ),
                page_id=page_id,
            )
            detections.append(detection)

        return detections
