"""Test that preprocessing preserves foreign keys."""

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor


class TestPreprocessingForeignKeys:
    """Test that preprocessing preserves foreign key columns."""

    def test_preprocessing_preserves_foreign_keys(self):
        """Test that all foreign key columns are preserved through preprocessing."""
        # Create test DataFrame with foreign keys
        df = pl.DataFrame(
            {
                # Primary key
                "line_id": ["3074409-X_1902-09-05_415_2_005_TB_1_TL_1"],
                # Foreign keys
                "source_id": ["3074409-X"],
                "issue_id": ["3074409-X_1902-09-05_415_2"],
                "page_id": ["3074409-X_1902-09-05_415_2_005"],
                "text_block_id": ["3074409-X_1902-09-05_415_2_005_TB_1"],
                # Data
                "text": ["Dies iſt ein Tefttext mit hiſtoriſchen Buchſtaben."],
                # Metadata
                "filename": ["test.xml"],
                "newspaper_title": ["Der Tag"],
            }
        )

        # Apply preprocessing
        preprocessor = TextPreprocessor(text_column="text")
        result_df = preprocessor.pipeline(
            df,
            steps=["normalize", "lowercase"],
            output_column="text_processed",
        )

        # Verify all foreign keys are preserved
        assert "line_id" in result_df.columns
        assert "source_id" in result_df.columns
        assert "issue_id" in result_df.columns
        assert "page_id" in result_df.columns
        assert "text_block_id" in result_df.columns

        # Verify foreign key values are unchanged
        assert result_df["line_id"][0] == "3074409-X_1902-09-05_415_2_005_TB_1_TL_1"
        assert result_df["source_id"][0] == "3074409-X"
        assert result_df["issue_id"][0] == "3074409-X_1902-09-05_415_2"
        assert result_df["page_id"][0] == "3074409-X_1902-09-05_415_2_005"
        assert result_df["text_block_id"][0] == "3074409-X_1902-09-05_415_2_005_TB_1"

        # Verify text was processed
        assert "text_processed" in result_df.columns
        processed_text = result_df["text_processed"][0]
        assert "iſt" not in processed_text  # Historical characters normalized
        assert "ſ" not in processed_text
        assert processed_text.islower()  # Lowercased

    def test_preprocessing_preserves_metadata(self):
        """Test that metadata columns are also preserved."""
        df = pl.DataFrame(
            {
                "line_id": ["test_line_1"],
                "source_id": ["der_tag"],
                "issue_id": ["der_tag_1902-09-05_415_2"],
                "page_id": ["der_tag_1902-09-05_415_2_005"],
                "text_block_id": ["der_tag_1902-09-05_415_2_005_TB_1"],
                "text": ["Test text"],
                "filename": ["test.xml"],
                "newspaper_title": ["Der Tag"],
                "page_number": [5],
                "year": [1902],
            }
        )

        preprocessor = TextPreprocessor(text_column="text")
        result_df = preprocessor.pipeline(
            df,
            steps=["lowercase"],
            output_column="text_processed",
        )

        # Verify metadata preserved
        assert "filename" in result_df.columns
        assert "newspaper_title" in result_df.columns
        assert "page_number" in result_df.columns
        assert "year" in result_df.columns

        assert result_df["filename"][0] == "test.xml"
        assert result_df["newspaper_title"][0] == "Der Tag"
        assert result_df["page_number"][0] == 5
        assert result_df["year"][0] == 1902

    def test_preprocessing_multiple_steps_preserves_fks(self):
        """Test that foreign keys survive complex multi-step pipeline."""
        df = pl.DataFrame(
            {
                "line_id": ["test_1", "test_2"],
                "source_id": ["3074409-X", "3074409-X"],
                "issue_id": ["3074409-X_1902-09-05_415_2", "3074409-X_1902-09-05_415_2"],
                "page_id": ["3074409-X_1902-09-05_415_2_005", "3074409-X_1902-09-05_415_2_005"],
                "text_block_id": [
                    "3074409-X_1902-09-05_415_2_005_TB_1",
                    "3074409-X_1902-09-05_415_2_005_TB_2",
                ],
                "text": [
                    "Dies iſt ein Tefttext 123",
                    "Noch ein Teft mit Sonderzeichen!",
                ],
            }
        )

        preprocessor = TextPreprocessor(text_column="text")
        result_df = preprocessor.pipeline(
            df,
            steps=["normalize", "lowercase", "remove-numbers", "remove-punctuation"],
            output_column="text_processed",
        )

        # All rows should preserve foreign keys
        assert len(result_df) == 2
        assert result_df["source_id"].to_list() == ["3074409-X", "3074409-X"]
        assert result_df["issue_id"].to_list() == [
            "3074409-X_1902-09-05_415_2",
            "3074409-X_1902-09-05_415_2",
        ]
        assert result_df["page_id"].to_list() == [
            "3074409-X_1902-09-05_415_2_005",
            "3074409-X_1902-09-05_415_2_005",
        ]
        assert result_df["text_block_id"].to_list() == [
            "3074409-X_1902-09-05_415_2_005_TB_1",
            "3074409-X_1902-09-05_415_2_005_TB_2",
        ]
