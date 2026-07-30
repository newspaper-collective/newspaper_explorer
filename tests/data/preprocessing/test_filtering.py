"""
Tests for filtering functions in data preprocessing.

Tests all filtering functions:
- filter_number_only_lines: number/separator-only lines
- filter_by_total_character_length: min/max character length
- filter_by_word_count: min/max word count
- filter_lines_without_alphabetic_chars: non-alphabetic lines
- filter_empty_lines: empty/whitespace-only lines
"""

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.filtering import (
    filter_by_total_character_length,
    filter_by_word_count,
    filter_empty_lines,
    filter_lines_without_alphabetic_chars,
    filter_number_only_lines,
)


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


# ---------------------------------------------------------------------------
# filter_by_total_character_length
# ---------------------------------------------------------------------------


class TestFilterByTotalCharacterLength:
    """Test suite for filter_by_total_character_length function."""

    def test_min_length_filters_short(self):
        """Lines shorter than min_length should be removed."""
        df = pl.DataFrame({"text": ["ab", "short", "a longer sentence here"]})
        result = filter_by_total_character_length(df, min_length=10)
        assert len(result) == 1
        assert result["text"][0] == "a longer sentence here"

    def test_max_length_filters_long(self):
        """Lines longer than max_length should be removed."""
        df = pl.DataFrame({"text": ["short", "medium text", "a" * 100]})
        result = filter_by_total_character_length(df, min_length=0, max_length=20)
        assert len(result) == 2
        assert "a" * 100 not in result["text"].to_list()

    def test_min_and_max_length(self):
        """Both min and max filters should apply when specified."""
        df = pl.DataFrame({"text": ["ab", "medium text", "a" * 100]})
        result = filter_by_total_character_length(df, min_length=5, max_length=50)
        assert len(result) == 1
        assert result["text"][0] == "medium text"

    def test_no_max_length(self):
        """When max_length is None, only min_length applies."""
        df = pl.DataFrame({"text": ["ab", "a" * 1000]})
        result = filter_by_total_character_length(df, min_length=3)
        assert len(result) == 1
        assert len(result["text"][0]) == 1000

    def test_custom_input_column(self):
        df = pl.DataFrame({"content": ["ab", "longer text here"]})
        result = filter_by_total_character_length(df, input_column="content", min_length=5)
        assert len(result) == 1

    def test_empty_dataframe(self):
        df = pl.DataFrame({"text": pl.Series([], dtype=pl.String)})
        result = filter_by_total_character_length(df, min_length=10)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_by_word_count
# ---------------------------------------------------------------------------


class TestFilterByWordCount:
    """Test suite for filter_by_word_count function."""

    def test_min_words_filters_short(self):
        """Lines with fewer words than min_words should be removed."""
        df = pl.DataFrame({"text": ["word", "two words", "this has four words"]})
        result = filter_by_word_count(df, min_words=2)
        assert len(result) == 2
        assert "word" not in result["text"].to_list()

    def test_max_words_filters_long(self):
        """Lines with more words than max_words should be removed."""
        df = pl.DataFrame({"text": ["short", "two words", "this has four words"]})
        result = filter_by_word_count(df, min_words=1, max_words=2)
        assert len(result) == 2
        assert "this has four words" not in result["text"].to_list()

    def test_min_and_max_words(self):
        """Both min and max filters should apply together."""
        df = pl.DataFrame({"text": ["one", "two words", "this has four words"]})
        result = filter_by_word_count(df, min_words=2, max_words=3)
        assert len(result) == 1
        assert result["text"][0] == "two words"

    def test_no_max_words(self):
        """When max_words is None, only min_words applies."""
        df = pl.DataFrame({"text": ["one", "a " * 500]})
        result = filter_by_word_count(df, min_words=2)
        assert len(result) == 1

    def test_custom_input_column(self):
        df = pl.DataFrame({"content": ["one", "two words here"]})
        result = filter_by_word_count(df, input_column="content", min_words=2)
        assert len(result) == 1

    def test_empty_dataframe(self):
        df = pl.DataFrame({"text": pl.Series([], dtype=pl.String)})
        result = filter_by_word_count(df, min_words=2)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_lines_without_alphabetic_chars
# ---------------------------------------------------------------------------


class TestFilterLinesWithoutAlphabeticChars:
    """Test suite for filter_lines_without_alphabetic_chars function."""

    def test_removes_pure_numbers(self):
        df = pl.DataFrame({"text": ["12345", "text here"]})
        result = filter_lines_without_alphabetic_chars(df)
        assert len(result) == 1
        assert result["text"][0] == "text here"

    def test_removes_pure_punctuation(self):
        df = pl.DataFrame({"text": ["!!!", "---", "Good text"]})
        result = filter_lines_without_alphabetic_chars(df)
        assert len(result) == 1
        assert result["text"][0] == "Good text"

    def test_keeps_text_with_numbers(self):
        """Lines mixing text and numbers should be kept."""
        df = pl.DataFrame({"text": ["Am 12. Maerz", "Page 42"]})
        result = filter_lines_without_alphabetic_chars(df)
        assert len(result) == 2

    def test_keeps_german_umlauts(self):
        """Unicode letters like umlauts should count as alphabetic."""
        df = pl.DataFrame({"text": ["Aendern", "123"]})
        result = filter_lines_without_alphabetic_chars(df)
        assert len(result) == 1
        assert result["text"][0] == "Aendern"

    def test_custom_input_column(self):
        df = pl.DataFrame({"content": ["123", "abc"]})
        result = filter_lines_without_alphabetic_chars(df, input_column="content")
        assert len(result) == 1
        assert result["content"][0] == "abc"

    def test_empty_dataframe(self):
        df = pl.DataFrame({"text": pl.Series([], dtype=pl.String)})
        result = filter_lines_without_alphabetic_chars(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_empty_lines
# ---------------------------------------------------------------------------


class TestFilterEmptyLines:
    """Test suite for filter_empty_lines function."""

    def test_removes_empty_strings(self):
        df = pl.DataFrame({"text": ["", "content", ""]})
        result = filter_empty_lines(df)
        assert len(result) == 1
        assert result["text"][0] == "content"

    def test_removes_whitespace_only(self):
        df = pl.DataFrame({"text": ["   ", "\t", "content"]})
        result = filter_empty_lines(df)
        assert len(result) == 1
        assert result["text"][0] == "content"

    def test_default_input_column(self):
        """When input_column=None, should default to 'text'."""
        df = pl.DataFrame({"text": ["", "hello"]})
        result = filter_empty_lines(df, input_column=None)
        assert len(result) == 1
        assert result["text"][0] == "hello"

    def test_custom_input_column(self):
        df = pl.DataFrame({"content": ["", "hello"]})
        result = filter_empty_lines(df, input_column="content")
        assert len(result) == 1
        assert result["content"][0] == "hello"

    def test_all_empty(self):
        df = pl.DataFrame({"text": ["", "  ", "\t"]})
        result = filter_empty_lines(df)
        assert len(result) == 0

    def test_empty_dataframe(self):
        df = pl.DataFrame({"text": pl.Series([], dtype=pl.String)})
        result = filter_empty_lines(df)
        assert len(result) == 0
