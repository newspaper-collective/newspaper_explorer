"""
Tests for text cleaning functions.

Covers edge cases for:
- remove_diacritics
- normalize_whitespace
- lowercase
- remove_punctuation
- remove_numbers
- remove_stopwords
- only_keep_allowed_chars

Fixtures are defined in conftest.py for reuse across preprocessing tests.
"""

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.cleaning import (
    lowercase,
    normalize_whitespace,
    only_keep_allowed_chars,
    remove_diacritics,
    remove_numbers,
    remove_punctuation,
    remove_stopwords,
)

# =============================================================================
# Tests for remove_diacritics
# =============================================================================


class TestRemoveDiacritics:
    """Tests for the remove_diacritics function."""

    def test_basic_german_umlauts(self, german_df: pl.DataFrame) -> None:
        """Test removal of German umlauts."""
        result = remove_diacritics(german_df)

        assert "text_no_diacritics" in result.columns
        values = result["text_no_diacritics"].to_list()

        assert values[0] == "Munchner Strasse"
        assert values[1] == "Die grosste Uberraschung"
        assert values[2] == "Apfel und Ol"
        assert values[3] == "Fliessendes Wasser"

    def test_custom_output_column(self, german_df: pl.DataFrame) -> None:
        """Test with custom output column name."""
        result = remove_diacritics(german_df, output_column="cleaned")

        assert "cleaned" in result.columns
        assert "text_no_diacritics" not in result.columns

    def test_custom_input_column(self) -> None:
        """Test with custom input column name."""
        df = pl.DataFrame({"content": ["Überraschung", "Äußerst"]})
        result = remove_diacritics(df, input_column="content")

        assert "content_no_diacritics" in result.columns
        assert result["content_no_diacritics"].to_list() == ["Uberraschung", "Ausserst"]

    def test_empty_strings(self) -> None:
        """Test handling of empty strings."""
        df = pl.DataFrame({"text": ["", "Test", ""]})
        result = remove_diacritics(df)

        assert result["text_no_diacritics"].to_list() == ["", "Test", ""]

    def test_null_values(self) -> None:
        """Test handling of null values."""
        df = pl.DataFrame({"text": ["Test", None, "Über"]})
        result = remove_diacritics(df)

        values = result["text_no_diacritics"].to_list()
        assert values[0] == "Test"
        assert values[1] == ""  # None becomes empty string
        assert values[2] == "Uber"

    def test_french_accents(self) -> None:
        """Test removal of French accents."""
        df = pl.DataFrame({"text": ["café", "naïve", "résumé", "crème brûlée"]})
        result = remove_diacritics(df)

        values = result["text_no_diacritics"].to_list()
        assert values[0] == "cafe"
        assert values[1] == "naive"
        assert values[2] == "resume"
        assert values[3] == "creme brulee"

    def test_preserves_original_column(self, german_df: pl.DataFrame) -> None:
        """Test that original column is preserved."""
        result = remove_diacritics(german_df)

        assert "text" in result.columns
        assert result["text"].to_list() == german_df["text"].to_list()


# =============================================================================
# Tests for normalize_whitespace
# =============================================================================


class TestNormalizeWhitespace:
    """Tests for the normalize_whitespace function."""

    def test_collapse_multiple_spaces(self, whitespace_df: pl.DataFrame) -> None:
        """Test collapsing multiple spaces."""
        result = normalize_whitespace(whitespace_df)

        values = result["text_whitespace"].to_list()
        assert values[0] == "Hello World"
        assert values[1] == "Multiple spaces here"

    def test_collapse_tabs(self, whitespace_df: pl.DataFrame) -> None:
        """Test collapsing tabs."""
        result = normalize_whitespace(whitespace_df)

        assert result["text_whitespace"][2] == "Tabs here"

    def test_collapse_newlines_default(self, whitespace_df: pl.DataFrame) -> None:
        """Test that newlines are collapsed by default."""
        result = normalize_whitespace(whitespace_df)

        assert result["text_whitespace"][3] == "Newlines and more"

    def test_preserve_newlines(self, whitespace_df: pl.DataFrame) -> None:
        """Test preserving newlines when keep_newlines=True."""
        result = normalize_whitespace(whitespace_df, keep_newlines=True)

        assert result["text_whitespace"][3] == "Newlines\nand\nmore"

    def test_mixed_whitespace_default(self, whitespace_df: pl.DataFrame) -> None:
        """Test mixed whitespace is collapsed to single space."""
        result = normalize_whitespace(whitespace_df)

        assert result["text_whitespace"][4] == "Mixed whitespace"

    def test_mixed_whitespace_preserve_newlines(self, whitespace_df: pl.DataFrame) -> None:
        """Test mixed whitespace with newline preservation."""
        result = normalize_whitespace(whitespace_df, keep_newlines=True)

        # Spaces and tabs are collapsed, newline preserved
        assert result["text_whitespace"][4] == "Mixed\nwhitespace"

    def test_strip_leading_trailing(self, whitespace_df: pl.DataFrame) -> None:
        """Test stripping leading and trailing whitespace."""
        result = normalize_whitespace(whitespace_df)

        assert result["text_whitespace"][5] == "leading and trailing"

    def test_custom_output_column(self, sample_df: pl.DataFrame) -> None:
        """Test with custom output column name."""
        result = normalize_whitespace(sample_df, output_column="normalized")

        assert "normalized" in result.columns

    def test_empty_string(self) -> None:
        """Test handling of empty strings."""
        df = pl.DataFrame({"text": ["", "  ", "\t\n"]})
        result = normalize_whitespace(df)

        assert result["text_whitespace"].to_list() == ["", "", ""]

    def test_empty_string_keep_newlines(self) -> None:
        """Test handling of empty strings with keep_newlines=True."""
        df = pl.DataFrame({"text": ["", "  ", "\t\n", None]})
        result = normalize_whitespace(df, keep_newlines=True)

        values = result["text_whitespace"].to_list()
        assert values[0] == ""  # Empty string
        assert values[1] == ""  # Only spaces
        assert values[2] == ""  # Tab and newline stripped
        assert values[3] is None  # None stays None

    def test_carriage_return_handling(self) -> None:
        """Test handling of different line break styles."""
        df = pl.DataFrame({"text": ["Windows\r\nline", "Mac\rline", "Unix\nline"]})
        result = normalize_whitespace(df, keep_newlines=True)

        values = result["text_whitespace"].to_list()
        assert values[0] == "Windows\nline"  # \r\n → \n
        assert values[1] == "Mac\nline"  # \r → \n
        assert values[2] == "Unix\nline"  # \n → \n


# =============================================================================
# Tests for lowercase
# =============================================================================


class TestLowercase:
    """Tests for the lowercase function."""

    def test_basic_lowercase(self, sample_df: pl.DataFrame) -> None:
        """Test basic lowercase conversion."""
        result = lowercase(sample_df)

        assert "text_lower" in result.columns
        assert result["text_lower"].to_list() == [
            "hello world",
            "test text",
            "another line",
        ]

    def test_german_umlauts(self, german_df: pl.DataFrame) -> None:
        """Test lowercase conversion of German umlauts."""
        result = lowercase(german_df)

        values = result["text_lower"].to_list()
        assert values[0] == "münchner straße"
        assert values[2] == "äpfel und öl"

    def test_already_lowercase(self) -> None:
        """Test text that's already lowercase."""
        df = pl.DataFrame({"text": ["already lowercase", "no changes needed"]})
        result = lowercase(df)

        assert result["text_lower"].to_list() == ["already lowercase", "no changes needed"]

    def test_mixed_case(self) -> None:
        """Test mixed case text."""
        df = pl.DataFrame({"text": ["MiXeD CaSe", "ALL CAPS", "lower case"]})
        result = lowercase(df)

        assert result["text_lower"].to_list() == ["mixed case", "all caps", "lower case"]

    def test_custom_columns(self) -> None:
        """Test with custom input and output columns."""
        df = pl.DataFrame({"content": ["UPPERCASE"]})
        result = lowercase(df, input_column="content", output_column="lowered")

        assert "lowered" in result.columns
        assert result["lowered"][0] == "uppercase"

    def test_preserves_numbers_punctuation(self) -> None:
        """Test that numbers and punctuation are preserved."""
        df = pl.DataFrame({"text": ["Hello, World! 123"]})
        result = lowercase(df)

        assert result["text_lower"][0] == "hello, world! 123"


# =============================================================================
# Tests for remove_punctuation
# =============================================================================


class TestRemovePunctuation:
    """Tests for the remove_punctuation function."""

    def test_basic_punctuation_removal(self, punctuation_df: pl.DataFrame) -> None:
        """Test basic punctuation removal."""
        result = remove_punctuation(punctuation_df)

        assert "text_nopunct" in result.columns
        values = result["text_nopunct"].to_list()

        assert values[0] == "Hello World"
        assert values[1] == "Question Answer"

    def test_semicolon_colon(self, punctuation_df: pl.DataFrame) -> None:
        """Test removal of semicolons and colons."""
        result = remove_punctuation(punctuation_df)

        assert result["text_nopunct"][2] == "Semicolonhere"

    def test_quotes(self, punctuation_df: pl.DataFrame) -> None:
        """Test removal of quotes."""
        result = remove_punctuation(punctuation_df)

        assert result["text_nopunct"][3] == "Quotes and more"

    def test_brackets(self, punctuation_df: pl.DataFrame) -> None:
        """Test removal of brackets."""
        result = remove_punctuation(punctuation_df)

        assert result["text_nopunct"][4] == "Brackets and more braces"

    def test_keep_chars_hyphen(self, punctuation_df: pl.DataFrame) -> None:
        """Test keeping hyphens while removing other punctuation."""
        result = remove_punctuation(punctuation_df, keep_chars="-")

        assert "-" in result["text_nopunct"][5]

    def test_keep_chars_multiple(self) -> None:
        """Test keeping multiple punctuation characters."""
        df = pl.DataFrame({"text": ["Hello, World! It's fine-tuned."]})
        result = remove_punctuation(df, keep_chars="-'")

        assert result["text_nopunct"][0] == "Hello World It's fine-tuned"

    def test_german_text_preserved(self) -> None:
        """Test that German umlauts are preserved."""
        df = pl.DataFrame({"text": ["Größe, Überraschung!"]})
        result = remove_punctuation(df)

        assert result["text_nopunct"][0] == "Größe Überraschung"

    def test_custom_columns(self) -> None:
        """Test with custom column names."""
        df = pl.DataFrame({"content": ["Hello, World!"]})
        result = remove_punctuation(df, input_column="content", output_column="clean")

        assert "clean" in result.columns
        assert result["clean"][0] == "Hello World"

    def test_backward_compatibility_text_column(self) -> None:
        """Test backward compatibility with text_column parameter."""
        df = pl.DataFrame({"content": ["Hello, World!"]})
        result = remove_punctuation(df, text_column="content")

        assert "content_nopunct" in result.columns


# =============================================================================
# Tests for remove_numbers
# =============================================================================


class TestRemoveNumbers:
    """Tests for the remove_numbers function."""

    def test_basic_number_removal(self, numbers_df: pl.DataFrame) -> None:
        """Test basic number removal."""
        result = remove_numbers(numbers_df)

        assert "text_nonum" in result.columns
        values = result["text_nonum"].to_list()

        assert values[0] == "Year "
        assert values[2] == "Page "

    def test_decimal_numbers(self, numbers_df: pl.DataFrame) -> None:
        """Test removal of decimal numbers."""
        result = remove_numbers(numbers_df)

        assert result["text_nonum"][1] == "Price ."

    def test_mixed_text_numbers(self, numbers_df: pl.DataFrame) -> None:
        """Test removal from mixed text and numbers."""
        result = remove_numbers(numbers_df)

        assert result["text_nonum"][3] == "Mixed textnumbers"

    def test_subscript_numbers(self, numbers_df: pl.DataFrame) -> None:
        """Test removal of subscript numbers (Unicode)."""
        result = remove_numbers(numbers_df)

        # Unicode subscript numbers should also be removed (via \p{N})
        assert "₁" not in result["text_nonum"][4]
        assert "₂" not in result["text_nonum"][4]
        assert "₃" not in result["text_nonum"][4]

    def test_roman_numerals_preserved(self, numbers_df: pl.DataFrame) -> None:
        """Test that Roman numerals (letters) are preserved."""
        result = remove_numbers(numbers_df)

        # Roman numerals are letters, not numbers
        assert "XII" in result["text_nonum"][5]

    def test_no_numbers(self) -> None:
        """Test text without numbers."""
        df = pl.DataFrame({"text": ["No numbers here", "Just text"]})
        result = remove_numbers(df)

        assert result["text_nonum"].to_list() == ["No numbers here", "Just text"]

    def test_only_numbers(self) -> None:
        """Test text that is only numbers."""
        df = pl.DataFrame({"text": ["12345", "99999"]})
        result = remove_numbers(df)

        assert result["text_nonum"].to_list() == ["", ""]

    def test_custom_columns(self) -> None:
        """Test with custom column names."""
        df = pl.DataFrame({"content": ["Page 42"]})
        result = remove_numbers(df, input_column="content", output_column="clean")

        assert result["clean"][0] == "Page "


# =============================================================================
# Tests for remove_stopwords
# =============================================================================


class TestRemoveStopwords:
    """Tests for the remove_stopwords function."""

    def test_german_stopwords(self, stopwords_df: pl.DataFrame) -> None:
        """Test removal of German stopwords."""
        result = remove_stopwords(stopwords_df, language="de")

        # "Der", "und", "die" are German stopwords
        clean_text = result["text_nostop"][0]
        assert "Mann" in clean_text
        assert "Frau" in clean_text

    def test_english_stopwords(self, stopwords_df: pl.DataFrame) -> None:
        """Test removal of English stopwords."""
        result = remove_stopwords(stopwords_df, language="en")

        # "The", "and", "the" are English stopwords
        clean_text = result["text_nostop"][1]
        assert "man" in clean_text
        assert "woman" in clean_text

    def test_german_article_removal(self, stopwords_df: pl.DataFrame) -> None:
        """Test removal of German articles."""
        result = remove_stopwords(stopwords_df, language="de")

        clean_text = result["text_nostop"][2]
        # "Das", "ist", "ein" are German stopwords
        assert "Test" in clean_text

    def test_english_article_removal(self, stopwords_df: pl.DataFrame) -> None:
        """Test removal of English articles."""
        result = remove_stopwords(stopwords_df, language="en")

        clean_text = result["text_nostop"][3]
        # "This", "is", "a" are English stopwords
        assert "test" in clean_text

    def test_case_insensitive(self) -> None:
        """Test that stopword removal is case-insensitive."""
        df = pl.DataFrame({"text": ["DER Mann UND die FRAU"]})
        result = remove_stopwords(df, language="de")

        clean_text = result["text_nostop"][0]
        assert "Mann" in clean_text
        assert "FRAU" in clean_text

    def test_unsupported_language(self) -> None:
        """Test error for unsupported language."""
        df = pl.DataFrame({"text": ["Test text"]})

        with pytest.raises(ValueError, match="Unsupported language"):
            remove_stopwords(df, language="fr")

    def test_no_stopwords(self) -> None:
        """Test text without stopwords."""
        df = pl.DataFrame({"text": ["Berlin München Hamburg"]})
        result = remove_stopwords(df, language="de")

        assert result["text_nostop"][0] == "Berlin München Hamburg"

    def test_only_stopwords(self) -> None:
        """Test text containing only stopwords."""
        df = pl.DataFrame({"text": ["der die das"]})
        result = remove_stopwords(df, language="de")

        # Should be empty or mostly empty after removal
        clean_text = result["text_nostop"][0].strip()
        # All words removed, leaving empty spaces
        assert len(clean_text.split()) == 0

    def test_custom_columns(self) -> None:
        """Test with custom column names."""
        df = pl.DataFrame({"content": ["Der Mann"]})
        result = remove_stopwords(df, input_column="content", output_column="clean", language="de")

        assert "clean" in result.columns


# =============================================================================
# Tests for only_keep_allowed_chars
# =============================================================================


class TestOnlyKeepAllowedChars:
    """Tests for the only_keep_allowed_chars function."""

    def test_normal_text_unchanged(self, ocr_artifacts_df: pl.DataFrame) -> None:
        """Test that normal text passes through unchanged."""
        result = only_keep_allowed_chars(ocr_artifacts_df)

        assert result["text_filtered"][0] == "Normal text here"

    def test_remove_special_symbols(self, ocr_artifacts_df: pl.DataFrame) -> None:
        """Test removal of special symbols."""
        result = only_keep_allowed_chars(ocr_artifacts_df)

        clean_text = result["text_filtered"][1]
        assert "™" not in clean_text
        assert "©" not in clean_text
        assert "symbols" in clean_text

    def test_remove_control_characters(self, ocr_artifacts_df: pl.DataFrame) -> None:
        """Test removal of control characters."""
        result = only_keep_allowed_chars(ocr_artifacts_df)

        clean_text = result["text_filtered"][2]
        assert "\x00" not in clean_text
        assert "\x01" not in clean_text
        assert "Control" in clean_text  # Content words preserved

    def test_remove_zero_width_characters(self, ocr_artifacts_df: pl.DataFrame) -> None:
        """Test removal of zero-width Unicode characters."""
        result = only_keep_allowed_chars(ocr_artifacts_df)

        clean_text = result["text_filtered"][3]
        assert "\u200b" not in clean_text  # Zero-width space
        assert "\u200c" not in clean_text  # Zero-width non-joiner
        assert "\u200d" not in clean_text  # Zero-width joiner

    def test_german_umlauts_preserved(self, ocr_artifacts_df: pl.DataFrame) -> None:
        """Test that German umlauts are preserved."""
        result = only_keep_allowed_chars(ocr_artifacts_df)

        clean_text = result["text_filtered"][4]
        assert "ä" in clean_text
        assert "ö" in clean_text
        assert "ü" in clean_text

    def test_currency_symbols_removed(self, ocr_artifacts_df: pl.DataFrame) -> None:
        """Test that currency symbols are removed by default."""
        result = only_keep_allowed_chars(ocr_artifacts_df)

        clean_text = result["text_filtered"][4]
        assert "§" not in clean_text
        assert "€" not in clean_text
        assert "£" not in clean_text

    def test_custom_allowed_chars(self) -> None:
        """Test with custom allowed characters pattern."""
        df = pl.DataFrame({"text": ["Test 123 äöü €100"]})
        result = only_keep_allowed_chars(df, allowed_chars=r"[a-zA-Z0-9\s]")

        # Only ASCII letters, numbers, spaces allowed
        clean_text = result["text_filtered"][0]
        assert "ä" not in clean_text
        assert "€" not in clean_text
        assert "Test" in clean_text
        assert "123" in clean_text

    def test_replace_with_space(self) -> None:
        """Test replacing invalid characters with space instead of removing."""
        df = pl.DataFrame({"text": ["Hello™World"]})
        result = only_keep_allowed_chars(df, replace_with=" ")

        # Should have space instead of ™
        clean_text = result["text_filtered"][0]
        assert "Hello" in clean_text
        assert "World" in clean_text
        assert "™" not in clean_text

    def test_whitespace_normalization(self) -> None:
        """Test that whitespace is normalized after cleanup."""
        df = pl.DataFrame({"text": ["Hello   ™™™   World"]})
        result = only_keep_allowed_chars(df)

        # Multiple spaces from removed chars should be collapsed
        clean_text = result["text_filtered"][0]
        assert "  " not in clean_text
        assert clean_text == "Hello World"

    def test_leading_trailing_stripped(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        df = pl.DataFrame({"text": ["  ™Hello World™  "]})
        result = only_keep_allowed_chars(df)

        clean_text = result["text_filtered"][0]
        assert not clean_text.startswith(" ")
        assert not clean_text.endswith(" ")

    def test_empty_string(self) -> None:
        """Test handling of empty strings."""
        df = pl.DataFrame({"text": ["", "   ", "™™™"]})
        result = only_keep_allowed_chars(df)

        values = result["text_filtered"].to_list()
        assert values[0] == ""
        assert values[1] == ""
        assert values[2] == ""

    def test_custom_columns(self) -> None:
        """Test with custom column names."""
        df = pl.DataFrame({"content": ["Hello™World"]})
        result = only_keep_allowed_chars(df, input_column="content", output_column="cleaned")

        assert "cleaned" in result.columns
        assert "™" not in result["cleaned"][0]


# =============================================================================
# Integration / Chain Tests
# =============================================================================


class TestCleaningChain:
    """Tests for chaining multiple cleaning operations."""

    def test_full_cleaning_pipeline(self) -> None:
        """Test a full cleaning pipeline with multiple operations."""
        df = pl.DataFrame(
            {
                "text": [
                    "  Der Münchner MANN, Nr. 123!  ",
                    "Die größte Überraschung™ © 2020.",
                ]
            }
        )

        # Chain operations
        result = normalize_whitespace(df)
        result = lowercase(result, input_column="text_whitespace")
        result = remove_punctuation(result, input_column="text_whitespace_lower")
        result = remove_numbers(result, input_column="text_whitespace_lower_nopunct")

        final_col = "text_whitespace_lower_nopunct_nonum"
        assert final_col in result.columns

        values = result[final_col].to_list()
        assert "münchner mann nr" in values[0].lower()

    def test_order_of_operations_matters(self) -> None:
        """Test that the order of operations affects the result."""
        df = pl.DataFrame({"text": ["HELLO, WORLD 123"]})

        # Lowercase first, then remove punctuation
        result1 = lowercase(df)
        result1 = remove_punctuation(result1, input_column="text_lower")

        # Remove punctuation first, then lowercase
        result2 = remove_punctuation(df)
        result2 = lowercase(result2, input_column="text_nopunct")

        # Both should give same final result in this case
        assert result1["text_lower_nopunct"][0] == result2["text_nopunct_lower"][0]

    def test_preserves_all_intermediate_columns(self) -> None:
        """Test that intermediate columns are preserved."""
        df = pl.DataFrame({"text": ["Hello, World!"]})

        result = lowercase(df)
        result = remove_punctuation(result, input_column="text_lower")

        # Original and all intermediate columns should exist
        assert "text" in result.columns
        assert "text_lower" in result.columns
        assert "text_lower_nopunct" in result.columns
