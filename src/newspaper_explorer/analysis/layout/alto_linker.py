"""
ALTO Linker - Link detections to ALTO XML text content.

This module provides functionality to match any detection type (headlines,
captions, tables, etc.) to OCR text content from ALTO XML based on bounding
box overlap with text lines.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import polars as pl

from newspaper_explorer.analysis.layout.schemas import Detection, BoundingBox

logger = logging.getLogger(__name__)


# ALTO XML namespace (v4)
ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


class ALTOLinker:
    """
    Universal ALTO linker for any detection type.

    Links detection bounding boxes to ALTO XML text content by finding
    overlapping text lines/blocks based on IoU (Intersection over Union).
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        min_confidence: float = 0.2,
    ):
        """
        Initialize the ALTOLinker.

        Args:
            overlap_threshold: Minimum IoU for matching (0.0-1.0)
            min_confidence: Minimum detection confidence to consider
        """
        self.overlap_threshold = overlap_threshold
        self.min_confidence = min_confidence

        logger.info(
            f"ALTOLinker initialized: overlap_threshold={overlap_threshold}, "
            f"min_confidence={min_confidence}"
        )

    def link_detections_to_text(
        self,
        detections: List[Detection],
        alto_xml_path: Optional[Path] = None,
        lines_df: Optional[pl.DataFrame] = None,
        page_id: Optional[str] = None,
    ) -> List[Detection]:
        """
        Link detections to OCR text from ALTO XML.

        Updates Detection objects in-place with text_content and alto_elements.

        Args:
            detections: List of detections to link
            alto_xml_path: Path to ALTO XML file (if using XML parsing)
            lines_df: Optional Polars DataFrame with parsed lines (faster)
            page_id: Page identifier (required if using lines_df)

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

        logger.debug(f"Linking {len(valid_detections)} detections to ALTO text")

        # Get ALTO lines
        if lines_df is not None and page_id is not None:
            alto_lines = self._lines_from_dataframe(lines_df, page_id)
        elif alto_xml_path is not None:
            alto_lines = self._parse_alto_lines(alto_xml_path)
        else:
            raise ValueError("Must provide either lines_df+page_id or alto_xml_path")

        if not alto_lines:
            logger.warning("No text lines found in ALTO")
            return valid_detections

        # Link each detection to text
        linked_count = 0
        for detection in valid_detections:
            matched_text, alto_elements, match_score = self._match_detection_to_text(
                detection, alto_lines
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

    def _match_detection_to_text(
        self, detection: Detection, alto_lines: List[Dict]
    ) -> Tuple[str, List[str], float]:
        """
        Match a detection to ALTO text lines.

        Args:
            detection: Detection to match
            alto_lines: List of ALTO line dictionaries

        Returns:
            Tuple of (matched_text, alto_element_ids, match_score)
        """
        matched_lines = []
        alto_element_ids = set()
        ious = []

        # Find overlapping lines
        for line in alto_lines:
            line_bbox = BoundingBox(
                x1=line["x"],
                y1=line["y"],
                x2=line["x"] + line["width"],
                y2=line["y"] + line["height"],
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

    def _parse_alto_lines(self, alto_xml_path: Path) -> List[Dict]:
        """
        Parse ALTO XML to extract text lines with coordinates.

        Args:
            alto_xml_path: Path to ALTO XML file

        Returns:
            List of line dictionaries with text and coordinates
        """
        try:
            tree = ET.parse(alto_xml_path)
            root = tree.getroot()
        except Exception as e:
            logger.error(f"Failed to parse ALTO XML {alto_xml_path}: {e}")
            return []

        lines = []

        # Find all TextLine elements
        for text_line in root.findall(".//alto:TextLine", ALTO_NS):
            try:
                # Get coordinates
                hpos = int(text_line.get("HPOS", 0))
                vpos = int(text_line.get("VPOS", 0))
                width = int(text_line.get("WIDTH", 0))
                height = int(text_line.get("HEIGHT", 0))

                if width == 0 or height == 0:
                    continue

                # Get text from String elements
                text_parts = []
                for string_elem in text_line.findall(".//alto:String", ALTO_NS):
                    content = string_elem.get("CONTENT", "")
                    if content:
                        text_parts.append(content)

                if not text_parts:
                    continue

                text = " ".join(text_parts)

                # Get IDs
                line_id = text_line.get("ID")

                # Get parent TextBlock ID
                text_block = text_line.find("..", ALTO_NS)
                while text_block is not None:
                    if "TextBlock" in text_block.tag:
                        break
                    text_block = text_block.find("..", ALTO_NS)

                text_block_id = text_block.get("ID") if text_block is not None else None

                lines.append(
                    {
                        "text": text,
                        "x": hpos,
                        "y": vpos,
                        "width": width,
                        "height": height,
                        "line_id": line_id,
                        "text_block_id": text_block_id,
                    }
                )
            except Exception as e:
                logger.debug(f"Skipping invalid text line: {e}")
                continue

        logger.debug(f"Parsed {len(lines)} text lines from ALTO XML")
        return lines

    def _lines_from_dataframe(self, df: pl.DataFrame, page_id: str) -> List[Dict]:
        """
        Extract lines from Polars DataFrame for a specific page.

        Args:
            df: DataFrame with parsed ALTO lines
            page_id: Page identifier to filter

        Returns:
            List of line dictionaries
        """
        # Filter for this page
        page_df = df.filter(pl.col("filename").str.contains(page_id))

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
