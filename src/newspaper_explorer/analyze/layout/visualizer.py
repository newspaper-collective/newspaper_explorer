"""
Visualization utilities for layout analysis debugging.

Provides functions to visualize detected regions, matched text, and bounding boxes
overlaid on newspaper page images.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from newspaper_explorer.models.analysis.layout import Detection, PageLayout, BoundingBox

logger = logging.getLogger(__name__)


# Color scheme for different element types (BGR format for OpenCV)
COLORS = {
    "Title": (255, 0, 0),  # Blue
    "Section-header": (255, 100, 0),  # Light blue
    "Picture": (0, 255, 0),  # Green
    "Caption": (0, 200, 200),  # Yellow
    "Table": (0, 0, 255),  # Red
    "Text": (0, 255, 255),  # Bright yellow (was light gray)
    "Formula": (255, 0, 255),  # Magenta
    "List-item": (150, 150, 255),  # Light purple
    "Page-header": (255, 165, 0),  # Orange (was dark gray)
    "Page-footer": (255, 165, 0),  # Orange (was dark gray)
    "Footnote": (200, 200, 0),  # Cyan
}


class LayoutVisualizer:
    """
    Visualizes layout detection results on newspaper page images.
    """

    def __init__(
        self,
        line_width: int = 3,
        font_scale: float = 0.6,
        font_thickness: int = 2,
        show_confidence: bool = True,
        show_text: bool = True,
        text_max_length: int = 30,
    ):
        """
        Initialize the visualizer.

        Args:
            line_width: Width of bounding box lines
            font_scale: Scale of text labels
            font_thickness: Thickness of text labels
            show_confidence: Whether to show confidence scores
            show_text: Whether to show matched text content
            text_max_length: Maximum length of text to display
        """
        self.line_width = line_width
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.show_confidence = show_confidence
        self.show_text = show_text
        self.text_max_length = text_max_length

        logger.info(f"LayoutVisualizer initialized")

    def visualize_page(
        self,
        page_layout: PageLayout,
        output_path: Optional[Path] = None,
        element_types: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Visualize all detected elements on a page.

        Args:
            page_layout: PageLayout with detections
            output_path: Optional path to save visualization
            element_types: Optional list of element types to visualize (None = all)

        Returns:
            Annotated image as numpy array
        """
        # Load image
        image = cv2.imread(page_layout.image_path)
        if image is None:
            logger.error(f"Failed to load image: {page_layout.image_path}")
            return None

        # Filter detections by type if specified
        detections = page_layout.detections
        if element_types:
            detections = [
                d
                for d in detections
                if any(et.lower() in d.class_name.lower() for et in element_types)
            ]

        logger.debug(f"Visualizing {len(detections)} detections on {page_layout.page_id}")

        # Draw each detection
        for detection in detections:
            image = self._draw_detection(image, detection)

        # Add legend
        image = self._draw_legend(image, detections)

        # Add title
        title = f"Page: {page_layout.page_id} | Detections: {len(detections)}"
        cv2.putText(
            image,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            logger.info(f"Saved visualization to {output_path}")

        return image

    def visualize_comparison(
        self,
        page_layout: PageLayout,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create side-by-side comparison of different element types.

        Args:
            page_layout: PageLayout with detections
            output_path: Optional path to save visualization

        Returns:
            Composite image with multiple views
        """
        # Load image
        image = cv2.imread(page_layout.image_path)
        if image is None:
            logger.error(f"Failed to load image: {page_layout.image_path}")
            return None

        # Create views for different element types
        views = []
        view_labels = []

        # Headlines view (Title, Section-header)
        headlines = page_layout.filter_by_class(["Title", "Section-header"])
        if headlines:
            headlines_img = self._create_filtered_view(image.copy(), headlines, "Headlines")
            views.append(headlines_img)
            view_labels.append(f"Headlines ({len(headlines)})")

        # Images view
        images = page_layout.filter_by_class("Picture")
        if images:
            images_img = self._create_filtered_view(image.copy(), images, "Pictures")
            views.append(images_img)
            view_labels.append(f"Images ({len(images)})")

        # Captions view
        captions = page_layout.filter_by_class("Caption")
        if captions:
            captions_img = self._create_filtered_view(image.copy(), captions, "Captions")
            views.append(captions_img)
            view_labels.append(f"Captions ({len(captions)})")

        # Tables view
        tables = page_layout.filter_by_class("Table")
        if tables:
            tables_img = self._create_filtered_view(image.copy(), tables, "Tables")
            views.append(tables_img)
            view_labels.append(f"Tables ({len(tables)})")

        if not views:
            logger.warning("No elements to visualize")
            return image

        # Create composite image (2x2 grid if 4 views, otherwise horizontal)
        if len(views) <= 2:
            composite = self._create_horizontal_composite(views, view_labels)
        else:
            composite = self._create_grid_composite(views, view_labels)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), composite)
            logger.info(f"Saved comparison visualization to {output_path}")

        return composite

    def _draw_detection(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        """Draw a single detection on the image."""
        # Get color for this class
        color = COLORS.get(detection.class_name, (128, 128, 128))

        # Draw bounding box
        pt1 = (int(detection.bbox.x1), int(detection.bbox.y1))
        pt2 = (int(detection.bbox.x2), int(detection.bbox.y2))
        cv2.rectangle(image, pt1, pt2, color, self.line_width)

        # Prepare label text
        label_parts = [detection.class_name]
        if self.show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")

        label = " ".join(label_parts)

        # Draw label background
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        label_bg_pt2 = (pt1[0] + label_size[0] + 10, pt1[1] - label_size[1] - 10)
        cv2.rectangle(image, pt1, label_bg_pt2, color, -1)

        # Draw label text
        cv2.putText(
            image,
            label,
            (pt1[0] + 5, pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
            cv2.LINE_AA,
        )

        # Draw matched text content if available
        if self.show_text and detection.text_content:
            text = detection.text_content[: self.text_max_length]
            if len(detection.text_content) > self.text_max_length:
                text += "..."

            text_y = int(detection.bbox.y2) + 20
            cv2.putText(
                image,
                text,
                (int(detection.bbox.x1), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale * 0.8,
                (0, 255, 255),
                self.font_thickness,
                cv2.LINE_AA,
            )

        return image

    def _draw_legend(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw legend showing element types and counts."""
        # Count detections by class
        class_counts = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        # Draw legend in top-right corner
        legend_x = image.shape[1] - 300
        legend_y = 50
        line_height = 30

        for idx, (class_name, count) in enumerate(sorted(class_counts.items())):
            y = legend_y + idx * line_height
            color = COLORS.get(class_name, (128, 128, 128))

            # Draw color box
            cv2.rectangle(
                image,
                (legend_x, y - 15),
                (legend_x + 20, y),
                color,
                -1,
            )

            # Draw text
            text = f"{class_name}: {count}"
            cv2.putText(
                image,
                text,
                (legend_x + 30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return image

    def _create_filtered_view(
        self, image: np.ndarray, detections: List[Detection], title: str
    ) -> np.ndarray:
        """Create a view showing only specific detections."""
        for detection in detections:
            image = self._draw_detection(image, detection)

        # Add title
        cv2.putText(
            image,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return image

    def _create_horizontal_composite(
        self, views: List[np.ndarray], labels: List[str]
    ) -> np.ndarray:
        """Create horizontal composite of views."""
        # Resize all views to same height
        target_height = min(view.shape[0] for view in views)
        resized_views = []

        for view in views:
            scale = target_height / view.shape[0]
            new_width = int(view.shape[1] * scale)
            resized = cv2.resize(view, (new_width, target_height))
            resized_views.append(resized)

        # Concatenate horizontally
        return np.hstack(resized_views)

    def _create_grid_composite(self, views: List[np.ndarray], labels: List[str]) -> np.ndarray:
        """Create 2x2 grid composite of views."""
        # Resize all views to same size
        target_size = (views[0].shape[1] // 2, views[0].shape[0] // 2)
        resized_views = [cv2.resize(view, target_size) for view in views]

        # Pad with blank if odd number of views
        if len(resized_views) % 2 == 1:
            blank = np.zeros_like(resized_views[0])
            resized_views.append(blank)

        # Create rows
        rows = []
        for i in range(0, len(resized_views), 2):
            row = np.hstack(resized_views[i : i + 2])
            rows.append(row)

        # Stack rows
        return np.vstack(rows)


def quick_visualize(
    page_layout: PageLayout,
    output_path: Path,
    element_types: Optional[List[str]] = None,
) -> None:
    """
    Quick visualization function for debugging.

    Args:
        page_layout: PageLayout to visualize
        output_path: Path to save visualization
        element_types: Optional list of element types to show
    """
    visualizer = LayoutVisualizer()
    visualizer.visualize_page(page_layout, output_path, element_types)
    logger.info(f"Saved visualization to {output_path}")
