"""
Tests for filtering functions in data preprocessing.

Tests number-only line filtering including:
- Plain numbers
- Numbers with parentheses/brackets
- Numbers with separators
- Mixed text (should be kept)
"""

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.filtering import filter_number_only_lines


class TestFilterNumberOnlyLines:
    """Test suite for filter_number_only_lines function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with various number formats."""
        data = {
            "text": [
                # Plain numbers - should be filtered
                "123",
                "45",
                "789",
                "1",
                # Numbers with parentheses - should be filtered (Issue #22)
                "(123)",
                "(45)",
                "(789)",
                # Numbers with brackets - should be filtered (Issue #22)
                "[123]",
                "[45]",
                "[789]",
                # Numbers with curly braces - should be filtered
                "{123}",
                "{45}",
                # Mixed parentheses/brackets with numbers - should be filtered
                "[(123)]",
                "([45])",
                # Numbers with separators - should be filtered
                "1.234",
                "12-34-56",
                "1,000",
                "12.34.56",
                "1:2:3",
                "1/2/3",
                # Numbers with spaces - should be filtered
                "1 2 3",
                "12 34",
                # Numbers with formatting in parentheses - should be filtered
                "(1,234)",
                "(12.34)",
                # Wrapped numbers with dashes - should be filtered
                "-123-",
                "--45--",
                # Fractions and superscripts - should be filtered
                "½",
                "3 ½",
                "3⁴",
                # Text with numbers - should be KEPT
                "Seite 123",
                "Am 12. März",
                "Die 3 Musketiere",
                "Kapitel (12) beginnt",
                "siehe [123] oben",
                "Absatz {5} erklärt",
                # Pure text - should be KEPT
                "Dies ist ein normaler Text",
                "Another line of text",
                "Überschrift",
                # Edge cases
                "",  # Empty string - should be kept (not matching pattern)
                "   ",  # Whitespace only - should be kept
                # Special number-like patterns that should be filtered
                "(1) (2) (3)",
                "[1][2][3]",
            ],
        }
        return pl.DataFrame(data)

    def test_filter_with_separators(self, sample_df):
        """Test filtering with allow_separators=True (default)."""
        result = filter_number_only_lines(sample_df, allow_separators=True)

        # Lines that should be REMOVED (filtered out)
        removed_patterns = [
            "123",
            "45",
            "789",
            "1",
            "(123)",
            "(45)",
            "(789)",
            "[123]",
            "[45]",
            "[789]",
            "{123}",
            "{45}",
            "[(123)]",
            "([45])",
            "1.234",
            "12-34-56",
            "1,000",
            "12.34.56",
            "1:2:3",
            "1/2/3",
            "1 2 3",
            "12 34",
            "(1,234)",
            "(12.34)",
            "-123-",
            "--45--",
            "½",
            "3 ½",
            "3⁴",
            "(1) (2) (3)",
            "[1][2][3]",
        ]

        for pattern in removed_patterns:
            assert pattern not in result["text"].to_list(), (
                f"'{pattern}' should be filtered out but wasn't"
            )

        # Lines that should be KEPT
        kept_patterns = [
            "Seite 123",
            "Am 12. März",
            "Die 3 Musketiere",
            "Kapitel (12) beginnt",
            "siehe [123] oben",
            "Absatz {5} erklärt",
            "Dies ist ein normaler Text",
            "Another line of text",
            "Überschrift",
            "",
            "   ",
        ]

        for pattern in kept_patterns:
            assert pattern in result["text"].to_list(), (
                f"'{pattern}' should be kept but was filtered"
            )

    def test_filter_without_separators(self, sample_df):
        """Test filtering with allow_separators=False (strict mode)."""
        result = filter_number_only_lines(sample_df, allow_separators=False)

        # In strict mode, only pure numbers should be filtered
        removed_patterns = ["123", "45", "789", "1"]

        for pattern in removed_patterns:
            assert pattern not in result["text"].to_list(), (
                f"'{pattern}' should be filtered out in strict mode"
            )

        # Everything with separators should be kept in strict mode
        kept_with_separators = [
            "(123)",
            "[45]",
            "1.234",
            "1,000",
            "3 ½",
        ]

        for pattern in kept_with_separators:
            assert pattern in result["text"].to_list(), f"'{pattern}' should be kept in strict mode"

    def test_parentheses_numbers_issue_22(self, sample_df):
        """Specific test for Issue #22: parenthesized numbers."""
        result = filter_number_only_lines(sample_df)

        # These specific cases from Issue #22 should be filtered
        assert "(123)" not in result["text"].to_list()
        assert "(45)" not in result["text"].to_list()

    def test_brackets_numbers_issue_22(self, sample_df):
        """Test for bracketed numbers (extension of Issue #22)."""
        result = filter_number_only_lines(sample_df)

        # Bracketed numbers should also be filtered
        assert "[123]" not in result["text"].to_list()
        assert "[789]" not in result["text"].to_list()

    def test_mixed_text_with_parentheses_kept(self, sample_df):
        """Ensure text containing numbers in parentheses is kept."""
        result = filter_number_only_lines(sample_df)

        # These should be kept because they contain actual text
        assert "Kapitel (12) beginnt" in result["text"].to_list()
        assert "siehe [123] oben" in result["text"].to_list()

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pl.DataFrame({"text": pl.Series([], dtype=pl.String)})
        result = filter_number_only_lines(empty_df)
        assert len(result) == 0

    def test_all_numbers_dataframe(self):
        """Test with DataFrame containing only numbers."""
        numbers_df = pl.DataFrame({"text": ["123", "456", "(789)", "[111]"]})
        result = filter_number_only_lines(numbers_df)
        assert len(result) == 0

    def test_custom_column_name(self):
        """Test with custom input column."""
        df = pl.DataFrame({"content": ["123", "text", "(456)"]})
        result = filter_number_only_lines(df, input_column="content")
        assert len(result) == 1
        assert result["content"][0] == "text"
