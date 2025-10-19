"""
Tests for text processing utilities.
"""

import polars as pl
import pytest

from newspaper_explorer.data.utils.text import (
    load_and_aggregate_textblocks,
    split_into_sentences,
)


def test_load_and_aggregate_textblocks_basic(tmp_path):
    """Test basic text block aggregation."""
    # Create sample line-level data
    data = {
        "text": ["Line 1", "Line 2", "Line 3", "Line 4"],
        "text_block_id": ["block1", "block1", "block2", "block2"],
        "page_id": ["page1", "page1", "page1", "page1"],
        "date": ["2001-01-01", "2001-01-01", "2001-01-01", "2001-01-01"],
        "x": [100, 100, 100, 100],
        "y": [10, 20, 50, 60],
        "filename": ["test.xml"] * 4,
        "newspaper_id": ["123"] * 4,
        "newspaper_title": ["Test Paper"] * 4,
        "year": [2001] * 4,
        "month": [1] * 4,
        "day": [1] * 4,
    }
    df = pl.DataFrame(data)

    # Save to parquet
    parquet_path = tmp_path / "test_lines.parquet"
    df.write_parquet(parquet_path)

    # Test aggregation
    result = load_and_aggregate_textblocks(parquet_path)

    # Should have 2 blocks (block1 and block2)
    assert len(result) == 2

    # Check aggregated text
    block1_text = result.filter(pl.col("text_block_id") == "block1")["text"][0]
    assert block1_text == "Line 1 Line 2"

    block2_text = result.filter(pl.col("text_block_id") == "block2")["text"][0]
    assert block2_text == "Line 3 Line 4"

    # Check line counts
    assert result["line_count"].to_list() == [2, 2]


def test_load_and_aggregate_custom_grouping(tmp_path):
    """Test custom grouping."""
    data = {
        "text": ["A", "B", "C", "D"],
        "text_block_id": ["block1", "block2", "block1", "block2"],
        "page_id": ["page1", "page1", "page2", "page2"],
        "date": ["2001-01-01"] * 4,
        "x": [100] * 4,
        "y": [10, 20, 30, 40],
        "filename": ["test.xml"] * 4,
        "newspaper_id": ["123"] * 4,
        "newspaper_title": ["Test"] * 4,
        "year": [2001] * 4,
        "month": [1] * 4,
        "day": [1] * 4,
    }
    df = pl.DataFrame(data)
    parquet_path = tmp_path / "test.parquet"
    df.write_parquet(parquet_path)

    # Group by page only
    result = load_and_aggregate_textblocks(
        parquet_path, group_by=["page_id", "date"], sort_by=["y"]
    )

    # Should have 2 pages
    assert len(result) == 2

    # Check text concatenation
    page1_text = result.filter(pl.col("page_id") == "page1")["text"][0]
    assert page1_text == "A B"


def test_load_and_aggregate_missing_file():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        load_and_aggregate_textblocks("nonexistent.parquet")


def test_split_into_sentences_basic():
    """Test sentence splitting."""
    # Skip if spacy not installed
    pytest.importorskip("spacy")

    try:
        import spacy

        spacy.load("de_core_news_sm")
    except OSError:
        pytest.skip("German spacy model not installed")

    # Create sample data
    data = {
        "text_block_id": ["block1", "block2"],
        "text": [
            "Das ist Satz eins. Das ist Satz zwei.",
            "Ein einzelner Satz.",
        ],
        "page_id": ["page1", "page1"],
        "date": ["2001-01-01", "2001-01-01"],
    }
    df = pl.DataFrame(data)

    # Split into sentences
    result = split_into_sentences(df, text_column="text", batch_size=10)

    # Should have 3 sentences total
    assert len(result) == 3

    # Check sentence counts
    block1_sentences = result.filter(pl.col("text_block_id") == "block1")
    assert len(block1_sentences) == 2
    assert block1_sentences["sentence_count"][0] == 2

    block2_sentences = result.filter(pl.col("text_block_id") == "block2")
    assert len(block2_sentences) == 1
    assert block2_sentences["sentence_count"][0] == 1

    # Check sentence IDs
    assert block1_sentences["sentence_id"].to_list() == [0, 1]
    assert block2_sentences["sentence_id"].to_list() == [0]


def test_split_into_sentences_no_spacy():
    """Test error when spacy not installed."""
    # Mock missing spacy
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"spacy": None}):
        df = pl.DataFrame({"text": ["test"]})

        # Should work - the import error is caught within the function
        # and re-raised with helpful message
        try:
            split_into_sentences(df)
            assert False, "Should have raised ImportError"
        except ImportError as e:
            assert "spaCy is required" in str(e)


def test_split_into_sentences_missing_column():
    """Test error handling for missing text column."""
    pytest.importorskip("spacy")

    df = pl.DataFrame({"other_column": ["test"]})

    with pytest.raises(ValueError, match="Column 'text' not found"):
        split_into_sentences(df, text_column="text")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
