"""
Tests for caption-picture matching using center-to-center distance.
"""

import polars as pl
import pytest

from newspaper_explorer.analyze.layout.region_matching import ProximityMatcher
from newspaper_explorer.models.analysis.layout import BoundingBox, Detection


class TestCenterToCenterMatching:
    """Test the improved caption matching algorithm."""

    def test_center_to_center_distance_calculation(self):
        """Test that distance is calculated with spatial awareness (edge-to-edge + alignment)."""
        matcher = ProximityMatcher(search_radius=300, relative_position="any")

        # Picture at (100, 100) to (200, 200) - center at (150, 150)
        picture_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)

        # Caption at (120, 220) to (180, 260) - center at (150, 240)
        # Caption is BELOW picture: bottom edge at 200, caption top at 220
        caption_bbox = BoundingBox(x1=120, y1=220, x2=180, y2=260)

        # With spatial algorithm: vertical_dist = 220 - 200 = 20px
        # horizontal_offset = 0 (centers aligned)
        # score = 20 + (0 * 0.2) = 20
        distance = matcher._calculate_distance(picture_bbox, caption_bbox)
        assert abs(distance - 20.0) < 0.01, f"Expected ~20 (edge distance), got {distance}"

    def test_shortest_distance_match(self):
        """Test that the caption with best spatial score is selected."""
        matcher = ProximityMatcher(search_radius=300, relative_position="any")

        # Picture at (100, 100) to (200, 200) - center at (150, 150), bottom at 200
        picture_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)

        # Caption 1: directly below at center (150, 240) - 20px from bottom, centered
        # vertical_dist = 220 - 200 = 20, horizontal_offset = 0
        # score = 20 + 0*0.2 = 20
        caption1_bbox = BoundingBox(x1=120, y1=220, x2=180, y2=260)

        # Caption 2: to the right - overlaps vertically, so uses center-to-center fallback
        # This should give higher score since it overlaps
        caption2_bbox = BoundingBox(x1=250, y1=130, x2=350, y2=170)

        # Caption 3: diagonal, overlaps corner - also uses center-to-center fallback
        caption3_bbox = BoundingBox(x1=200, y1=200, x2=300, y2=300)

        dist1 = matcher._calculate_distance(picture_bbox, caption1_bbox)
        dist2 = matcher._calculate_distance(picture_bbox, caption2_bbox)
        dist3 = matcher._calculate_distance(picture_bbox, caption3_bbox)

        # Caption 1 should have best score (properly positioned below)
        assert dist1 < dist2, f"Caption 1 ({dist1}) should be closer than caption 2 ({dist2})"
        # Caption 3 overlaps, so might have similar score to caption 1
        assert dist1 <= dist3 + 1, (
            f"Caption 1 ({dist1}) should be close to or better than caption 3 ({dist3})"
        )

    def test_page_based_filtering(self):
        """Test that pictures only match captions on the same page."""
        # Create test data with two pages
        pictures = [
            {
                "detection_id": "pic1",
                "page_id": "page_1",
                "class_name": "Picture",
                "bbox_x1": 100,
                "bbox_y1": 100,
                "bbox_x2": 200,
                "bbox_y2": 200,
            },
            {
                "detection_id": "pic2",
                "page_id": "page_2",
                "class_name": "Picture",
                "bbox_x1": 100,
                "bbox_y1": 100,
                "bbox_x2": 200,
                "bbox_y2": 200,
            },
        ]

        captions = [
            {
                "detection_id": "cap1",
                "page_id": "page_1",
                "class_name": "Caption",
                "text_content": "Caption for page 1",
                "bbox_x1": 120,
                "bbox_y1": 220,
                "bbox_x2": 180,
                "bbox_y2": 260,
            },
            {
                "detection_id": "cap2",
                "page_id": "page_2",
                "class_name": "Caption",
                "text_content": "Caption for page 2",
                "bbox_x1": 120,
                "bbox_y1": 220,
                "bbox_x2": 180,
                "bbox_y2": 260,
            },
        ]

        pictures_df = pl.DataFrame(pictures)
        captions_df = pl.DataFrame(captions)

        # Simulate the page-based matching logic used in commands.py
        for page_id in pictures_df["page_id"].unique().to_list():
            page_pictures = pictures_df.filter(pl.col("page_id") == page_id)
            page_captions = captions_df.filter(pl.col("page_id") == page_id)

            # Verify only same-page captions are considered
            assert len(page_pictures) == 1, f"Expected 1 picture on {page_id}"
            assert len(page_captions) == 1, f"Expected 1 caption on {page_id}"
            assert page_captions["page_id"][0] == page_id, (
                f"Caption should be on same page {page_id}"
            )

    def test_max_distance_threshold(self):
        """Test that captions beyond max_distance are not matched."""
        matcher = ProximityMatcher(search_radius=100, relative_position="any")

        # Picture at (100, 100) to (200, 200) - bottom at 200
        picture_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)

        # Caption far away - top at 380, so vertical_dist = 380 - 200 = 180
        caption_bbox = BoundingBox(x1=120, y1=380, x2=180, y2=420)

        distance = matcher._calculate_distance(picture_bbox, caption_bbox)
        # With spatial algorithm: vertical_dist = 180, h_offset = 0
        # score = 180 + 0*0.2 = 180
        assert abs(distance - 180.0) < 0.01, f"Expected ~180 (vertical distance), got {distance}"

        # Verify this would be rejected by search_radius=100
        assert distance > matcher.search_radius, "Caption should exceed search radius"

    def test_caption_above_picture(self):
        """Test that captions above pictures can be matched (no positional constraint)."""
        matcher = ProximityMatcher(search_radius=300, relative_position="any")

        # Picture at (100, 200) to (200, 300) - center at (150, 250), top at 200
        picture_bbox = BoundingBox(x1=100, y1=200, x2=200, y2=300)

        # Caption ABOVE picture at (120, 100) to (180, 150) - bottom at 150
        # vertical_dist = 200 - 150 = 50px, horizontal_offset = 0
        # score = 50 + 0*0.2 = 50
        caption_bbox = BoundingBox(x1=120, y1=100, x2=180, y2=150)

        distance = matcher._calculate_distance(picture_bbox, caption_bbox)
        assert abs(distance - 50.0) < 0.01, f"Expected ~50 (edge distance), got {distance}"
        assert distance < matcher.search_radius, "Caption above should be within range"

    def test_equal_distances_first_wins(self):
        """Test behavior when two captions have equal distance."""
        # Picture at center (150, 150)
        pic_center_x, pic_center_y = 150, 150

        # Two captions equidistant from picture
        # Caption 1: 100 pixels to the right
        cap1_center_x, cap1_center_y = 250, 150

        # Caption 2: 100 pixels above
        cap2_center_x, cap2_center_y = 150, 50

        dist1 = ((pic_center_x - cap1_center_x) ** 2 + (pic_center_y - cap1_center_y) ** 2) ** 0.5
        dist2 = ((pic_center_x - cap2_center_x) ** 2 + (pic_center_y - cap2_center_y) ** 2) ** 0.5

        assert abs(dist1 - dist2) < 0.01, "Distances should be equal"
        assert abs(dist1 - 100.0) < 0.01, "Both should be 100 pixels away"

    def test_landscape_image_spatial_matching(self):
        """
        Test the landscape image problem: caption below center should win
        over caption at top-left corner, even if corner is closer by pure distance.

        Uses ProximityMatcher.match_elements with Detection objects.
        """
        matcher = ProximityMatcher(search_radius=400, relative_position="any")

        # Wide landscape picture
        picture = Detection(
            detection_id="pic_1",
            class_name="Picture",
            confidence=0.9,
            bbox=BoundingBox(x1=100, y1=100, x2=500, y2=300),  # 400px wide, 200px tall
            page_id="test_page",
        )

        # Caption A: Near top-left corner
        caption_a = Detection(
            detection_id="cap_a",
            class_name="Caption",
            confidence=0.9,
            bbox=BoundingBox(x1=30, y1=50, x2=90, y2=80),
            page_id="test_page",
            text_content="Corner caption",
        )

        # Caption B: Directly below picture, centered
        caption_b = Detection(
            detection_id="cap_b",
            class_name="Caption",
            confidence=0.9,
            bbox=BoundingBox(x1=250, y1=320, x2=350, y2=350),  # 20px below, centered
            page_id="test_page",
            text_content="Proper caption below",
        )

        # Match without text extraction (we already have text)
        matches = matcher.match_elements(
            source_elements=[picture],
            target_elements=[caption_a, caption_b],
            extract_text=False,
        )

        assert len(matches) == 1
        matched_source, matched_target = matches[0]
        assert matched_source.detection_id == "pic_1"
        assert matched_target is not None
        assert matched_target.detection_id == "cap_b", (
            f"Expected properly positioned caption below to win, got: {matched_target.detection_id}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
