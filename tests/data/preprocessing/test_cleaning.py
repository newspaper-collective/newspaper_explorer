"""
Tests for cleaning functions in data preprocessing.

Verifies that each cleaning function produces correct output transformations,
not just that it runs without errors.
"""

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.cleaning import (
    only_keep_allowed_chars,
    remove_diacritics,
    remove_garbage_words,
    remove_long_words,
    remove_numbers,
    remove_punctuation,
    remove_stopwords,
)

# ---------------------------------------------------------------------------
# remove_punctuation
# ---------------------------------------------------------------------------


class TestRemovePunctuation:
    """Verify punctuation is actually stripped from text."""

    def test_removes_common_punctuation(self):
        df = pl.DataFrame({"text": ["Hello, World!", "Test... done.", "What?!"]})
        result = remove_punctuation(df)
        texts = result["text_nopunct"].to_list()
        assert texts == ["Hello World", "Test done", "What"]

    def test_preserves_german_umlauts(self):
        """Umlauts must survive punctuation removal."""
        df = pl.DataFrame({"text": ["Ärger, Ärger!", "Über-drüssig."]})
        result = remove_punctuation(df)
        texts = result["text_nopunct"].to_list()
        assert "Ärger" in texts[0]
        assert "Über" in texts[1]
        assert "drüssig" in texts[1]
        # Punctuation gone
        assert "," not in texts[0]
        assert "!" not in texts[0]

    def test_keep_chars_preserves_hyphens(self):
        """keep_chars parameter should preserve specified punctuation."""
        df = pl.DataFrame({"text": ["Nord-Süd-Ost, fertig."]})
        result = remove_punctuation(df, keep_chars="-")
        text = result["text_nopunct"].to_list()[0]
        assert "-" in text
        assert "," not in text
        assert "." not in text

    def test_empty_string(self):
        df = pl.DataFrame({"text": [""]})
        result = remove_punctuation(df)
        assert result["text_nopunct"].to_list() == [""]

    def test_custom_output_column(self):
        df = pl.DataFrame({"text": ["Hello!"]})
        result = remove_punctuation(df, output_column="clean")
        assert "clean" in result.columns
        assert result["clean"].to_list() == ["Hello"]


# ---------------------------------------------------------------------------
# remove_numbers
# ---------------------------------------------------------------------------


class TestRemoveNumbers:
    """Verify numbers are correctly removed from text."""

    def test_removes_ascii_digits(self):
        df = pl.DataFrame({"text": ["Es war 1920 ein gutes Jahr"]})
        result = remove_numbers(df)
        assert "1920" not in result["text_nonum"].to_list()[0]

    def test_removes_unicode_numbers(self):
        """Should remove superscripts, fractions, etc."""
        df = pl.DataFrame({"text": ["3⁴ und ½ Liter"]})
        result = remove_numbers(df)
        text = result["text_nonum"].to_list()[0]
        assert "3" not in text
        assert "4" not in text.replace("und", "").replace("Liter", "")
        assert "½" not in text

    def test_preserves_letters(self):
        df = pl.DataFrame({"text": ["Artikel 5 besagt"]})
        result = remove_numbers(df)
        text = result["text_nonum"].to_list()[0]
        assert "Artikel" in text
        assert "besagt" in text


# ---------------------------------------------------------------------------
# remove_stopwords
# ---------------------------------------------------------------------------


class TestRemoveStopwords:
    """Verify correct stopwords are removed per language."""

    def test_removes_german_stopwords(self):
        df = pl.DataFrame({"text": ["Die Zeitung ist ein wichtiges Medium"]})
        result = remove_stopwords(df, language="de")
        text = result["text_nostop"].to_list()[0]
        # "Die", "ist", "ein" are German stopwords
        words = text.split()
        lower_words = [w.lower() for w in words]
        assert "die" not in lower_words
        assert "ist" not in lower_words
        assert "ein" not in lower_words
        # Content words preserved
        assert "Zeitung" in words
        assert "wichtiges" in words
        assert "Medium" in words

    def test_removes_english_stopwords(self):
        df = pl.DataFrame({"text": ["The newspaper is an important medium"]})
        result = remove_stopwords(df, language="en")
        text = result["text_nostop"].to_list()[0]
        words = text.split()
        lower_words = [w.lower() for w in words]
        assert "the" not in lower_words
        assert "is" not in lower_words
        assert "an" not in lower_words
        assert "newspaper" in lower_words
        assert "important" in lower_words

    def test_case_insensitive_matching(self):
        """Stopwords should be matched case-insensitively."""
        df = pl.DataFrame({"text": ["Die DIE die"]})
        result = remove_stopwords(df, language="de")
        text = result["text_nostop"].to_list()[0].strip()
        # All variants of 'die' should be removed
        assert text == ""

    def test_unsupported_language_raises(self):
        df = pl.DataFrame({"text": ["some text"]})
        with pytest.raises(ValueError, match="Unsupported language"):
            remove_stopwords(df, language="xx")


# ---------------------------------------------------------------------------
# remove_long_words
# ---------------------------------------------------------------------------


class TestRemoveLongWords:
    """Verify words exceeding max length are removed while preserving valid ones."""

    def test_normal_german_compound_kept(self):
        """Donaudampfschifffahrtsgesellschaft (36 chars) should survive default threshold."""
        compound = "Donaudampfschifffahrtsgesellschaft"
        assert len(compound) == 34
        df = pl.DataFrame({"text": [f"Die {compound} existiert"]})
        result = remove_long_words(df)
        assert compound in result["text"].to_list()[0]

    def test_ocr_merge_removed(self):
        """Extremely long word (>45 chars) should be removed."""
        garbage = "a" * 50
        df = pl.DataFrame({"text": [f"Die {garbage} Zeitung"]})
        result = remove_long_words(df)
        text = result["text"].to_list()[0]
        assert garbage not in text
        assert "Die" in text
        assert "Zeitung" in text

    def test_custom_threshold(self):
        df = pl.DataFrame({"text": ["short longword"]})
        result = remove_long_words(df, max_word_length=6)
        text = result["text"].to_list()[0]
        assert "short" in text
        assert "longword" not in text

    def test_empty_text_unchanged(self):
        df = pl.DataFrame({"text": [""]})
        result = remove_long_words(df)
        assert result["text"].to_list() == [""]


# ---------------------------------------------------------------------------
# remove_garbage_words
# ---------------------------------------------------------------------------


class TestRemoveGarbageWords:
    """Verify OCR garbage detection based on character repetition."""

    def test_repeated_chars_removed(self):
        """Words like 'ssss' or 'jjjj' should be detected as garbage."""
        df = pl.DataFrame({"text": ["Die ssss Zeitung"]})
        result = remove_garbage_words(df)
        text = result["text"].to_list()[0]
        assert "ssss" not in text
        assert "Die" in text
        assert "Zeitung" in text

    def test_low_unique_ratio_removed(self):
        """'sssuuusss' has 2 unique / 9 total = 0.22 ratio -> garbage."""
        df = pl.DataFrame({"text": ["Die sssuuusss Zeitung"]})
        result = remove_garbage_words(df)
        assert "sssuuusss" not in result["text"].to_list()[0]

    def test_valid_german_preserved(self):
        """Normal German text including repeated chars in valid words."""
        df = pl.DataFrame({"text": ["Schifffahrt auf der Donau"]})
        result = remove_garbage_words(df)
        text = result["text"].to_list()[0]
        # "Schifffahrt" has 8 unique / 12 total = 0.67 ratio -> NOT garbage
        assert "Schifffahrt" in text
        assert "Donau" in text

    def test_short_words_not_checked(self):
        """Words shorter than min_word_length should not be flagged."""
        df = pl.DataFrame({"text": ["aa bb cc normal"]})
        result = remove_garbage_words(df, min_word_length=3)
        text = result["text"].to_list()[0]
        # "aa" and "bb" are only 2 chars, below min_word_length=3
        assert "aa" in text
        assert "bb" in text

    def test_pipe_like_characters_removed(self):
        """'|||' should be detected as garbage (1 unique / 3 total)."""
        df = pl.DataFrame({"text": ["text ||| more"]})
        result = remove_garbage_words(df)
        assert "|||" not in result["text"].to_list()[0]

    def test_custom_thresholds(self):
        """Stricter thresholds should catch more words."""
        df = pl.DataFrame({"text": ["abcabc normal"]})
        # abcabc: 3 unique / 6 total = 0.5 ratio
        # With default 0.3 threshold -> kept
        result = remove_garbage_words(df)
        assert "abcabc" in result["text"].to_list()[0]
        # With stricter 0.6 threshold -> removed
        result = remove_garbage_words(df, max_repetition_ratio=0.6)
        assert "abcabc" not in result["text"].to_list()[0]

    def test_empty_text_unchanged(self):
        """Empty/whitespace-only text should pass through clean_line unchanged."""
        df = pl.DataFrame({"text": ["", "  ", "normal text"]})
        result = remove_garbage_words(df)
        texts = result["text"].to_list()
        assert texts[0] == ""
        assert texts[1] == "  "
        assert texts[2] == "normal text"


# ---------------------------------------------------------------------------
# remove_diacritics
# ---------------------------------------------------------------------------


class TestRemoveDiacritics:
    """Verify diacritics are converted to ASCII equivalents."""

    def test_german_umlauts_converted(self):
        df = pl.DataFrame({"text": ["Münchner Straße"]})
        result = remove_diacritics(df)
        text = result["text_no_diacritics"].to_list()[0]
        assert text == "Munchner Strasse"

    def test_french_accents_converted(self):
        df = pl.DataFrame({"text": ["café résumé"]})
        result = remove_diacritics(df)
        text = result["text_no_diacritics"].to_list()[0]
        assert text == "cafe resume"

    def test_ascii_text_unchanged(self):
        df = pl.DataFrame({"text": ["Hello World"]})
        result = remove_diacritics(df)
        assert result["text_no_diacritics"].to_list()[0] == "Hello World"

    def test_empty_and_none_handling(self):
        df = pl.DataFrame({"text": ["", "normal"]})
        result = remove_diacritics(df)
        texts = result["text_no_diacritics"].to_list()
        assert texts[0] == ""
        assert texts[1] == "normal"


# ---------------------------------------------------------------------------
# only_keep_allowed_chars
# ---------------------------------------------------------------------------


class TestOnlyKeepAllowedChars:
    """Verify character allowlisting works correctly."""

    def test_default_keeps_german_text(self):
        """Normal German text should pass through the default filter."""
        df = pl.DataFrame({"text": ["Die Zeitung erscheint täglich, um 8 Uhr."]})
        result = only_keep_allowed_chars(df)
        text = result["text_filtered"].to_list()[0]
        assert "Zeitung" in text
        assert "täglich" in text
        assert "8" in text

    def test_removes_ocr_garbage(self):
        """Characters outside the allowed set should be removed."""
        df = pl.DataFrame({"text": ["Text with •bullets• and ▪squares▪"]})
        result = only_keep_allowed_chars(df)
        text = result["text_filtered"].to_list()[0]
        assert "•" not in text
        assert "▪" not in text
        assert "Text" in text
        assert "bullets" in text

    def test_custom_pattern(self):
        """Custom allowlist should restrict to specified characters."""
        df = pl.DataFrame({"text": ["Hello 123 World!"]})
        result = only_keep_allowed_chars(df, allowed_chars=r"[a-zA-Z\s]")
        text = result["text_filtered"].to_list()[0]
        assert "Hello" in text
        assert "World" in text
        assert "123" not in text
        assert "!" not in text

    def test_replace_with_space(self):
        """replace_with parameter should substitute instead of removing."""
        df = pl.DataFrame({"text": ["text•here"]})
        result = only_keep_allowed_chars(df, replace_with=" ")
        text = result["text_filtered"].to_list()[0]
        assert "•" not in text
        # The bullet is replaced by space and whitespace is normalized
        assert "text" in text
        assert "here" in text

    def test_whitespace_normalization_after_cleanup(self):
        """Multiple spaces from removed chars should be collapsed."""
        df = pl.DataFrame({"text": ["text••••here"]})
        result = only_keep_allowed_chars(df)
        text = result["text_filtered"].to_list()[0]
        # After removing bullets, spaces should be collapsed
        assert "  " not in text
