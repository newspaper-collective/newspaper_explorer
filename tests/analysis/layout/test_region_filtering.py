"""
Test coordinate-based filtering in RegionExtractor.
"""

import pytest
from pathlib import Path
from newspaper_explorer.analyze.layout.region_extraction import RegionExtractor
from newspaper_explorer.analyze.layout.schemas import Detection, BoundingBox


class TestCoordinateFiltering:
    """Test coordinate-based region filtering."""

    def test_exclude_top_percent(self):
        """Test exclusion of regions in top percentage of page."""
        # Page is 1000px high
        # Top 10% = 0-100px should be excluded
        extractor = RegionExtractor(exclude_top_percent=10.0)

        detections = [
            Detection(
                detection_id="header_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=50, x2=200, y2=80),  # In top 10%
                page_id="test_page",
            ),
            Detection(
                detection_id="content_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=200, x2=200, y2=300),  # Below top 10%
                page_id="test_page",
            ),
        ]

        filtered = extractor._filter_by_coordinates(
            detections=detections,
            page_height=1000,
            page_width=800,
        )

        assert len(filtered) == 1
        assert filtered[0].detection_id == "content_1"

    def test_exclude_bottom_percent(self):
        """Test exclusion of regions in bottom percentage of page."""
        # Page is 1000px high
        # Bottom 5% = 950-1000px should be excluded
        extractor = RegionExtractor(exclude_bottom_percent=5.0)

        detections = [
            Detection(
                detection_id="content_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=200, x2=200, y2=300),
                page_id="test_page",
            ),
            Detection(
                detection_id="footer_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=960, x2=200, y2=990),  # In bottom 5%
                page_id="test_page",
            ),
        ]

        filtered = extractor._filter_by_coordinates(
            detections=detections,
            page_height=1000,
            page_width=800,
        )

        assert len(filtered) == 1
        assert filtered[0].detection_id == "content_1"

    def test_min_region_height(self):
        """Test minimum height constraint."""
        extractor = RegionExtractor(min_region_height=100)

        detections = [
            Detection(
                detection_id="small_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=200, x2=200, y2=250),  # 50px high
                page_id="test_page",
            ),
            Detection(
                detection_id="large_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=200, x2=200, y2=350),  # 150px high
                page_id="test_page",
            ),
        ]

        filtered = extractor._filter_by_coordinates(
            detections=detections,
            page_height=1000,
            page_width=800,
        )

        assert len(filtered) == 1
        assert filtered[0].detection_id == "large_1"

    def test_min_region_width(self):
        """Test minimum width constraint."""
        extractor = RegionExtractor(min_region_width=100)

        detections = [
            Detection(
                detection_id="narrow_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=200, x2=150, y2=300),  # 50px wide
                page_id="test_page",
            ),
            Detection(
                detection_id="wide_1",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=200, x2=250, y2=300),  # 150px wide
                page_id="test_page",
            ),
        ]

        filtered = extractor._filter_by_coordinates(
            detections=detections,
            page_height=1000,
            page_width=800,
        )

        assert len(filtered) == 1
        assert filtered[0].detection_id == "wide_1"

    def test_combined_filters(self):
        """Test multiple filters applied together."""
        extractor = RegionExtractor(
            exclude_top_percent=10.0,
            exclude_bottom_percent=5.0,
            min_region_height=80,
            min_region_width=80,
        )

        detections = [
            Detection(
                detection_id="header",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=50, x2=200, y2=150),  # Top zone
                page_id="test_page",
            ),
            Detection(
                detection_id="footer",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=960, x2=200, y2=990),  # Bottom zone
                page_id="test_page",
            ),
            Detection(
                detection_id="too_small",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=400, x2=150, y2=440),  # 50x40px
                page_id="test_page",
            ),
            Detection(
                detection_id="valid",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=400, x2=200, y2=500),  # 100x100px, middle
                page_id="test_page",
            ),
        ]

        filtered = extractor._filter_by_coordinates(
            detections=detections,
            page_height=1000,
            page_width=800,
        )

        assert len(filtered) == 1
        assert filtered[0].detection_id == "valid"

    def test_no_filters(self):
        """Test that no filtering occurs when no constraints set."""
        extractor = RegionExtractor()

        detections = [
            Detection(
                detection_id=f"det_{i}",
                class_name="picture",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=i * 100, x2=200, y2=i * 100 + 50),
                page_id="test_page",
            )
            for i in range(10)
        ]

        filtered = extractor._filter_by_coordinates(
            detections=detections,
            page_height=1000,
            page_width=800,
        )

        assert len(filtered) == len(detections)

    def test_empty_detections(self):
        """Test filtering with empty detection list."""
        extractor = RegionExtractor(exclude_top_percent=10.0)

        filtered = extractor._filter_by_coordinates(
            detections=[],
            page_height=1000,
            page_width=800,
        )

        assert len(filtered) == 0
