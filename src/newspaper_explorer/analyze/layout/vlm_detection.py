"""
VLM-based Document Layout Detection using vLLM.

Uses vision-language models (e.g., dots.mocr) to detect document layout
elements with bounding boxes and categories. Complements the YOLO-based
LayoutDetector by providing a VLM perspective for comparison or ensemble use.

The VLM returns structured layout output (bounding boxes + categories) which
is parsed into the same Detection/PageLayout models used by the YOLO pipeline,
enabling seamless integration with TextLinker, RegionExtractor, etc.
"""

import json
import logging
from pathlib import Path
import re
from typing import Optional, Union

from PIL import Image
from vllm import LLM, SamplingParams

from newspaper_explorer.data.indexing.image_index import ImageIndexer
from newspaper_explorer.data.utils.ids import (
    extract_issue_id_from_page_id,
    generate_detection_id,
    generate_page_id,
    parse_page_id,
)
from newspaper_explorer.models.analysis.layout import BoundingBox, Detection, PageLayout

logger = logging.getLogger(__name__)


# Default model for VLM layout detection
DEFAULT_MODEL = "rednote-hilab/dots.mocr"

# Prompt modes supported by dots.mocr
PROMPT_MODES = {
    "layout-all": "Parse the layout of this document image. Return all detected regions with"
    " their bounding boxes and categories.",
    "layout-only": "Detect the layout regions in this document image. Return only the bounding"
    " boxes and categories, no text content.",
    "ocr": "Perform OCR on this document image and return the text content in markdown format.",
}

# Category mapping from VLM output to our Detection class_names
# dots.mocr categories -> DocLayNet-compatible names
VLM_CATEGORY_MAP = {
    # Direct matches
    "title": "Title",
    "text": "Text",
    "table": "Table",
    "picture": "Picture",
    "image": "Picture",
    "figure": "Picture",
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "list": "List-item",
    "list-item": "List-item",
    # Header/footer
    "header": "Page-header",
    "page-header": "Page-header",
    "page header": "Page-header",
    "footer": "Page-footer",
    "page-footer": "Page-footer",
    "page footer": "Page-footer",
    # Section headers
    "section-header": "Section-header",
    "section header": "Section-header",
    "heading": "Section-header",
    "subtitle": "Section-header",
    # Fallbacks for various VLM outputs
    "paragraph": "Text",
    "article": "Text",
    "advertisement": "Text",
    "ad": "Text",
    "illustration": "Picture",
    "photo": "Picture",
    "photograph": "Picture",
    "chart": "Table",
    "graph": "Table",
    "diagram": "Picture",
    "separator": "Page-footer",
    "decoration": "Page-header",
}

# Minimum confidence for VLM detections (VLMs don't produce confidence scores,
# so we assign a default)
DEFAULT_VLM_CONFIDENCE = 0.8


def _normalize_category(raw_category: str) -> str:
    """
    Map VLM category string to DocLayNet-compatible class name.

    Args:
        raw_category: Category string from VLM output

    Returns:
        Normalized class name matching DOCLAYNET_CLASSES
    """
    normalized = raw_category.strip().lower()
    return VLM_CATEGORY_MAP.get(normalized, "Text")


def _parse_vlm_layout_output(raw_output: str, page_id: str, image_path: str) -> list[Detection]:
    """
    Parse VLM layout detection output into Detection objects.

    dots.mocr layout-all output format is typically structured text with
    bounding boxes in [x1, y1, x2, y2] format and category labels.
    The output can be JSON-like or use a custom markup format.

    Args:
        raw_output: Raw text output from the VLM
        page_id: Page identifier for the detections
        image_path: Path to the source image

    Returns:
        List of Detection objects
    """
    detections = []

    # Extract foreign keys from page_id
    source_id = None
    issue_id = None
    try:
        page_components = parse_page_id(page_id)
        source_id = str(page_components.source) if page_components.source is not None else None
        issue_id = extract_issue_id_from_page_id(page_id)
    except Exception as e:
        logger.debug(f"Could not parse page_id {page_id}: {e}")

    # Strategy 1: Try JSON parsing (some VLMs return JSON arrays)
    json_detections = _try_parse_json(raw_output)
    if json_detections:
        for item in json_detections:
            det = _json_item_to_detection(item, page_id, image_path, source_id, issue_id)
            if det:
                detections.append(det)
        return detections

    # Strategy 2: Parse bbox pattern: <bbox>x1, y1, x2, y2</bbox> or [x1, y1, x2, y2]
    bbox_detections = _try_parse_bbox_tags(raw_output, page_id, image_path, source_id, issue_id)
    if bbox_detections:
        return bbox_detections

    # Strategy 3: Parse coordinate patterns with category labels
    coord_detections = _try_parse_coordinate_lines(
        raw_output, page_id, image_path, source_id, issue_id
    )
    if coord_detections:
        return coord_detections

    if not detections:
        logger.warning(f"Could not parse VLM layout output for {page_id}. Raw output length: "
                       f"{len(raw_output)}")
        logger.debug(f"Raw output preview: {raw_output[:500]}")

    return detections


def _try_parse_json(raw_output: str) -> Optional[list[dict]]:
    """Try to parse VLM output as JSON."""
    # Find JSON array in output
    json_match = re.search(r'\[[\s\S]*\]', raw_output)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list) and len(data) > 0:
                return data
        except json.JSONDecodeError:
            pass
    return None


def _json_item_to_detection(
    item: dict,
    page_id: str,
    image_path: str,
    source_id: Optional[str],
    issue_id: Optional[str],
) -> Optional[Detection]:
    """Convert a JSON dict from VLM output to a Detection."""
    # Look for bbox in various key names
    bbox_keys = ["bbox", "box", "bounding_box", "coordinates", "region"]
    bbox_data = None
    for key in bbox_keys:
        if key in item:
            bbox_data = item[key]
            break

    if bbox_data is None:
        return None

    # Parse bbox (could be [x1,y1,x2,y2] or {"x1":..,"y1":..})
    try:
        if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
            x1, y1, x2, y2 = float(bbox_data[0]), float(bbox_data[1]), float(bbox_data[2]), float(bbox_data[3])
        elif isinstance(bbox_data, dict):
            x1 = float(bbox_data.get("x1", bbox_data.get("left", 0)))
            y1 = float(bbox_data.get("y1", bbox_data.get("top", 0)))
            x2 = float(bbox_data.get("x2", bbox_data.get("right", 0)))
            y2 = float(bbox_data.get("y2", bbox_data.get("bottom", 0)))
        else:
            return None
    except (ValueError, TypeError):
        return None

    # Get category
    category_keys = ["category", "label", "class", "type", "class_name"]
    raw_category = "text"
    for key in category_keys:
        if key in item:
            raw_category = str(item[key])
            break

    class_name = _normalize_category(raw_category)

    # Get optional text content
    text_keys = ["text", "content", "ocr_text"]
    text_content = None
    for key in text_keys:
        if key in item and item[key]:
            text_content = str(item[key])
            break

    # Get confidence if available
    confidence = DEFAULT_VLM_CONFIDENCE
    if "confidence" in item:
        try:
            confidence = float(item["confidence"])
        except (ValueError, TypeError):
            pass

    detection_id = generate_detection_id(page_id, class_name)

    return Detection(
        detection_id=detection_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        page_id=page_id,
        source_id=source_id,
        issue_id=issue_id,
        image_path=image_path,
        text_content=text_content,
    )


def _try_parse_bbox_tags(
    raw_output: str,
    page_id: str,
    image_path: str,
    source_id: Optional[str],
    issue_id: Optional[str],
) -> list[Detection]:
    """Parse dots.mocr-style bbox tags: <ref>category</ref><bbox>x1,y1,x2,y2</bbox>."""
    detections = []

    # Pattern: <ref>Category</ref><bbox>x1, y1, x2, y2</bbox> with optional text
    pattern = re.compile(
        r'<ref>\s*([^<]+?)\s*</ref>\s*<bbox>\s*'
        r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)'
        r'\s*</bbox>(?:\s*([^<]*))?'
    )

    for match in pattern.finditer(raw_output):
        raw_category = match.group(1)
        x1 = float(match.group(2))
        y1 = float(match.group(3))
        x2 = float(match.group(4))
        y2 = float(match.group(5))
        text_content = match.group(6).strip() if match.group(6) else None

        class_name = _normalize_category(raw_category)
        detection_id = generate_detection_id(page_id, class_name)

        detections.append(
            Detection(
                detection_id=detection_id,
                class_name=class_name,
                confidence=DEFAULT_VLM_CONFIDENCE,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                page_id=page_id,
                source_id=source_id,
                issue_id=issue_id,
                image_path=image_path,
                text_content=text_content if text_content else None,
            )
        )

    return detections


def _try_parse_coordinate_lines(
    raw_output: str,
    page_id: str,
    image_path: str,
    source_id: Optional[str],
    issue_id: Optional[str],
) -> list[Detection]:
    """Parse line-based format: 'category [x1, y1, x2, y2] optional_text'."""
    detections = []

    # Pattern: category [x1, y1, x2, y2] or category: [x1, y1, x2, y2]
    pattern = re.compile(
        r'([A-Za-z][\w\s-]*?)\s*:?\s*\[\s*'
        r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)'
        r'\s*\]'
    )

    for match in pattern.finditer(raw_output):
        raw_category = match.group(1).strip()
        x1 = float(match.group(2))
        y1 = float(match.group(3))
        x2 = float(match.group(4))
        y2 = float(match.group(5))

        class_name = _normalize_category(raw_category)
        detection_id = generate_detection_id(page_id, class_name)

        detections.append(
            Detection(
                detection_id=detection_id,
                class_name=class_name,
                confidence=DEFAULT_VLM_CONFIDENCE,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                page_id=page_id,
                source_id=source_id,
                issue_id=issue_id,
                image_path=image_path,
            )
        )

    return detections


class VLMLayoutDetector:
    """
    VLM-based Document Layout Detector using vLLM.

    Uses vision-language models to detect document structure elements.
    Returns the same Detection/PageLayout models as LayoutDetector (YOLO),
    enabling seamless integration with TextLinker, RegionExtractor, etc.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prompt_mode: str = "layout-all",
        device: str = "cuda:0",
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        max_tokens: int = 2048,
        source_name: Optional[str] = None,
    ):
        """
        Initialize the VLM Layout Detector.

        Args:
            model_name: HuggingFace model ID (default: rednote-hilab/dots.mocr)
            prompt_mode: Detection mode - 'layout-all', 'layout-only', or 'ocr'
            device: CUDA device string (e.g., 'cuda:0')
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum context length for the model
            max_tokens: Maximum output tokens per image
            source_name: Source name for proper ID generation
        """
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.max_tokens = max_tokens
        self.source_name = source_name
        self.source_id = None
        self.image_index = None

        if prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"Invalid prompt_mode: {prompt_mode}. Choose from: {list(PROMPT_MODES.keys())}"
            )

        # Parse device to get GPU index
        gpu_id = 0
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
            except (IndexError, ValueError):
                gpu_id = 0

        logger.info(f"Loading VLM model: {model_name} on {device} (GPU {gpu_id})...")

        # vLLM uses CUDA_VISIBLE_DEVICES to select the GPU.
        # Must be set BEFORE importing/initializing the engine since vLLM
        # spawns a subprocess that inherits the environment.
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="auto",
            limit_mm_per_prompt={"image": 1},
            disable_log_stats=True,
        )

        # Load image index if source_name provided
        if self.source_name:
            logger.info(f"Loading image index for source: {self.source_name}")
            indexer = ImageIndexer(self.source_name)
            self.source_id = indexer.source_id
            self.image_index = indexer.load_index()

            if self.image_index is None:
                logger.info(f"No image index found for {self.source_name}, creating one...")
                self.image_index = indexer.create_index()
                logger.info(f"Created image index with {len(self.image_index)} images")
            else:
                logger.info(f"Loaded image index with {len(self.image_index)} images")

        logger.info(
            f"VLMLayoutDetector initialized: model={model_name}, prompt_mode={prompt_mode}, "
            f"device={device}"
        )

    def _generate_page_id(self, image_path: Path) -> str:
        """
        Generate page_id from image path with fallback to path-based ID.

        Same logic as LayoutDetector for consistency.
        """
        if self.source_id and self.image_index is not None:
            try:
                parts = image_path.parts
                images_idx = parts.index("images")
                rel_path = Path(*parts[images_idx + 1:])
                rel_path_str = str(rel_path)

                matches = self.image_index.filter(self.image_index["image_path"] == rel_path_str)

                if len(matches) > 0:
                    row = matches.row(0, named=True)
                    if not all([
                        row.get("date"),
                        row.get("issue_number"),
                        row.get("edition"),
                        row.get("page_number"),
                    ]):
                        raise ValueError("Incomplete metadata")

                    from datetime import datetime
                    date = datetime.strptime(row["date"], "%Y-%m-%d")
                    page_id = generate_page_id(
                        source=self.source_id,
                        date=date,
                        issue_number=row["issue_number"],
                        edition=row["edition"],
                        page_number=row["page_number"],
                    )
                    return page_id
            except Exception as e:
                logger.debug(f"Failed to generate page_id from index: {e}")

        # Fallback: path-based ID
        parts = image_path.parts
        try:
            images_idx = parts.index("images")
            relative_parts = list(parts[images_idx + 1:])
            relative_parts = relative_parts[:-1] + [Path(relative_parts[-1]).stem]
            return "_".join(relative_parts)
        except (ValueError, IndexError):
            return image_path.stem

    def detect_page(
        self, image_path: Union[str, Path], page_id: Optional[str] = None
    ) -> PageLayout:
        """
        Detect layout elements in a single page image using VLM.

        Args:
            image_path: Path to the page image
            page_id: Optional page identifier (if None, generates from path)

        Returns:
            PageLayout with all detections
        """
        image_path = Path(image_path)
        if page_id is None:
            page_id = self._generate_page_id(image_path)

        logger.debug(f"VLM detecting layout for page: {page_id}")

        results = self._run_inference([image_path])

        if results:
            detections = _parse_vlm_layout_output(results[0], page_id, str(image_path))
        else:
            detections = []

        page_layout = PageLayout(
            page_id=page_id,
            image_path=str(image_path),
            detections=detections,
        )

        logger.debug(
            f"VLM detected {len(detections)} elements in {page_id}: {page_layout.counts}"
        )

        return page_layout

    def detect_batch(
        self,
        image_paths: list[Union[str, Path]],
        page_ids: Optional[list[str]] = None,
    ) -> list[PageLayout]:
        """
        Detect layout elements in multiple page images.

        Args:
            image_paths: List of paths to page images
            page_ids: Optional list of page identifiers

        Returns:
            List of PageLayout objects
        """
        image_paths = [Path(p) for p in image_paths]

        if page_ids is None:
            page_ids = [self._generate_page_id(p) for p in image_paths]

        logger.info(f"VLM batch detection: {len(image_paths)} images")

        raw_outputs = self._run_inference(image_paths)

        results = []
        for idx, (img_path, pid) in enumerate(zip(image_paths, page_ids)):
            if idx < len(raw_outputs):
                detections = _parse_vlm_layout_output(raw_outputs[idx], pid, str(img_path))
            else:
                logger.warning(f"No VLM output for {pid}")
                detections = []

            page_layout = PageLayout(
                page_id=pid,
                image_path=str(img_path),
                detections=detections,
            )
            results.append(page_layout)

        total_dets = sum(len(r.detections) for r in results)
        logger.info(f"VLM batch complete: {total_dets} detections from {len(results)} pages")

        return results

    def _run_inference(self, image_paths: list[Path]) -> list[str]:
        """
        Run VLM inference on a list of images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of raw output strings from the VLM
        """
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        prompt_text = PROMPT_MODES[self.prompt_mode]

        # Build chat messages with images (lets vLLM handle chat template)
        conversations = []
        for img_path in image_paths:
            pil_image = Image.open(img_path).convert("RGB")
            conversations.append([{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image}},
                    {"type": "text", "text": prompt_text},
                ],
            }])

        outputs = self.llm.chat(conversations, sampling_params=sampling_params)

        results = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            results.append(text)

        return results
