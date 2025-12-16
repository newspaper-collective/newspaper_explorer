"""
Tests for unified ID generation utilities.
"""

from datetime import datetime

import pytest

from newspaper_explorer.data.utils.ids import (
    extract_issue_id_from_page_id,
    extract_page_id_from_text_block_id,
    extract_text_block_id_from_line_id,
    generate_article_id,
    generate_detection_id,
    generate_entity_id,
    generate_issue_id,
    generate_line_id,
    generate_page_id,
    generate_source_id,
    generate_text_block_id,
    parse_issue_id,
    parse_line_id,
    parse_page_id,
    source_id_to_filename_prefix,
)


class TestIDGeneration:
    """Test ID generation functions"""

    def test_generate_source_id(self):
        """Test source ID generation"""
        source_id = generate_source_id("der_tag")
        assert source_id == "der_tag"

    def test_source_id_to_filename_prefix(self):
        """Test converting source_id to filename prefix"""
        # ZDB ID with hyphen
        assert source_id_to_filename_prefix("3074409-X") == "3074409X"

        # Regular source name (unchanged)
        assert source_id_to_filename_prefix("der_tag") == "der_tag"

        # Multiple hyphens
        assert source_id_to_filename_prefix("1234-5-X") == "12345X"

    def test_generate_issue_id(self):
        """Test issue ID generation"""
        issue_id = generate_issue_id("der_tag", datetime(1902, 9, 5), 415, 2)
        assert issue_id == "der_tag_1902-09-05_415_2"

    def test_generate_page_id(self):
        """Test page ID generation"""
        page_id = generate_page_id("der_tag", datetime(1902, 9, 5), 415, 2, 5)
        assert page_id == "der_tag_1902-09-05_415_2_005"

    def test_generate_text_block_id(self):
        """Test text block ID generation"""
        block_id = generate_text_block_id("der_tag_1902-09-05_415_2_005", "TB_1")
        assert block_id == "der_tag_1902-09-05_415_2_005_TB_1"

    def test_generate_line_id(self):
        """Test line ID generation"""
        line_id = generate_line_id("der_tag_1902-09-05_415_2_005_TB_1", "TL_1")
        assert line_id == "der_tag_1902-09-05_415_2_005_TB_1_TL_1"

    def test_generate_detection_id(self):
        """Test detection ID generation"""
        detection_id = generate_detection_id("der_tag_1902-09-05_415_2_005", "headline")

        # Check format: should start with page_id and end with class_uuid
        assert detection_id.startswith("der_tag_1902-09-05_415_2_005")
        assert "headline" in detection_id
        # Extract UUID part (last part after underscore)
        uuid_part = detection_id.split("_")[-1]
        assert len(uuid_part) == 6  # UUID is 6 chars

    def test_generate_article_id(self):
        """Test article ID generation"""
        article_id = generate_article_id("der_tag_1902-09-05_415_2_005")

        # Check format
        assert article_id.startswith("der_tag_1902-09-05_415_2_005_art_")
        uuid_part = article_id.split("_")[-1]
        assert len(uuid_part) == 6

    def test_generate_entity_id(self):
        """Test entity ID generation"""
        entity_id = generate_entity_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")

        # Check format
        assert entity_id.startswith("der_tag_1902-09-05_415_2_005_TB_1_TL_1_ent_")
        uuid_part = entity_id.split("_")[-1]
        assert len(uuid_part) == 6


class TestIDParsing:
    """Test ID parsing functions"""

    def test_parse_page_id(self):
        """Test parsing page ID"""
        components = parse_page_id("der_tag_1902-09-05_415_2_005")

        assert components.source == "der_tag"
        assert components.date == "1902-09-05"
        assert components.issue_number == 415
        assert components.edition == 2
        assert components.page_number == 5

    def test_parse_page_id_invalid(self):
        """Test parsing invalid page ID"""
        with pytest.raises(ValueError, match="Could not find date"):
            parse_page_id("invalid_id")

    def test_parse_issue_id(self):
        """Test parsing issue ID"""
        components = parse_issue_id("der_tag_1902-09-05_415_2")

        assert components.source == "der_tag"
        assert components.date == "1902-09-05"
        assert components.issue_number == 415
        assert components.edition == 2

    def test_parse_issue_id_invalid(self):
        """Test parsing invalid issue ID"""
        with pytest.raises(ValueError, match="Could not find date"):
            parse_issue_id("invalid_id")

    def test_parse_line_id(self):
        """Test parsing line ID"""
        components = parse_line_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")

        assert components.source == "der_tag"
        assert components.date == "1902-09-05"
        assert components.issue_number == 415
        assert components.edition == 2
        assert components.page_number == 5
        assert components.block_id == "TB_1"
        assert components.line_id == "TL_1"
        assert components.full_page_id == "der_tag_1902-09-05_415_2_005"
        assert components.full_text_block_id == "der_tag_1902-09-05_415_2_005_TB_1"

    def test_parse_line_id_invalid(self):
        """Test parsing invalid line ID"""
        with pytest.raises(ValueError, match="Could not find date"):
            parse_line_id("invalid_id")


class TestIDExtraction:
    """Test ID extraction helper functions"""

    def test_extract_issue_id_from_page_id(self):
        """Test extracting issue ID from page ID"""
        issue_id = extract_issue_id_from_page_id("der_tag_1902-09-05_415_2_005")
        assert issue_id == "der_tag_1902-09-05_415_2"

    def test_extract_page_id_from_text_block_id(self):
        """Test extracting page ID from text block ID"""
        page_id = extract_page_id_from_text_block_id("der_tag_1902-09-05_415_2_005_TB_1")
        assert page_id == "der_tag_1902-09-05_415_2_005"

    def test_extract_text_block_id_from_line_id(self):
        """Test extracting text block ID from line ID"""
        block_id = extract_text_block_id_from_line_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")
        assert block_id == "der_tag_1902-09-05_415_2_005_TB_1"


class TestIDHierarchy:
    """Test the complete ID hierarchy works together"""

    def test_complete_id_chain(self):
        """Test generating complete ID hierarchy"""
        source = "der_tag"
        date = datetime(1902, 9, 5)
        issue_num = 415
        daily_num = 2
        page_num = 5

        # Generate IDs
        source_id = generate_source_id(source)
        issue_id = generate_issue_id(source, date, issue_num, daily_num)
        page_id = generate_page_id(source, date, issue_num, daily_num, page_num)
        text_block_id = generate_text_block_id(page_id, "TB_1")
        line_id = generate_line_id(text_block_id, "TL_1")

        # Verify hierarchy
        assert source_id == "der_tag"
        assert issue_id.startswith(source_id)
        assert page_id.startswith(issue_id)
        assert text_block_id.startswith(page_id)
        assert line_id.startswith(text_block_id)

        # Extract back
        assert extract_issue_id_from_page_id(page_id) == issue_id
        assert extract_page_id_from_text_block_id(text_block_id) == page_id
        assert extract_text_block_id_from_line_id(line_id) == text_block_id

    def test_id_foreign_keys(self):
        """Test that IDs can be used as foreign keys"""
        # Generate test IDs
        page_id = generate_page_id("der_tag", datetime(1902, 9, 5), 415, 2, 5)
        block_id = generate_text_block_id(page_id, "TB_1")
        line_id = generate_line_id(block_id, "TL_1")

        # Simulate a database-like structure
        lines = [
            {
                "line_id": line_id,
                "text": "Sample text",
                "text_block_id": block_id,
                "page_id": page_id,
            }
        ]

        # Verify foreign key relationships
        line = lines[0]
        assert line["text_block_id"] in line["line_id"]
        assert line["page_id"] in line["text_block_id"]
