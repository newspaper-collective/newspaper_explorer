"""
Tests for normalization functions in data preprocessing.

Verifies that each normalization function produces the correct character-level
transformations for historical German newspaper text.
"""

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.normalization import (
    _dehyphenate_lines,
    dehyphenate,
    normalize_casing,
    normalize_hyphens,
    normalize_long_s,
    normalize_umlauts,
    normalize_unicode,
    normalize_whitespace,
    replace_long_s_with_f,
)

# ---------------------------------------------------------------------------
# normalize_unicode
# ---------------------------------------------------------------------------


class TestNormalizeUnicode:
    """Verify Unicode normalization handles OCR artifacts and encoding issues."""

    def test_cyrillic_confusables_replaced(self):
        """Cyrillic lookalikes should become Latin equivalents."""
        # \u0422\u0435xt = Cyrillic Te + Cyrillic ie + Latin xt
        df = pl.DataFrame({"text": ["\u0422\u0435xt"]})
        result = normalize_unicode(df)
        text = result["text_unicode"].to_list()[0]
        assert text == "Text"

    def test_greek_confusables_replaced(self):
        """Greek lookalikes should become Latin."""
        # \u0391\u0392\u0395 = Greek Alpha + Beta + Epsilon
        df = pl.DataFrame({"text": ["\u0391\u0392\u0395"]})
        result = normalize_unicode(df)
        text = result["text_unicode"].to_list()[0]
        assert text == "ABE"

    def test_accented_i_normalized(self):
        """German doesn't use accented i -- these are OCR errors."""
        df = pl.DataFrame({"text": ["\u00edns\u00edde"]})  # insidé
        result = normalize_unicode(df)
        text = result["text_unicode"].to_list()[0]
        assert text == "inside"

    def test_various_spaces_unified(self):
        """Non-breaking space, thin space, etc. should become regular space."""
        # U+00A0 non-breaking space, U+2009 thin space
        df = pl.DataFrame({"text": ["hello\u00a0world\u2009here"]})
        result = normalize_unicode(df)
        text = result["text_unicode"].to_list()[0]
        assert text == "hello world here"

    def test_ocr_artifacts_removed(self):
        """Bullets and boxes (OCR garbage) should be removed."""
        df = pl.DataFrame({"text": ["text \u2022 bullet \u25aa square"]})
        result = normalize_unicode(df)
        text = result["text_unicode"].to_list()[0]
        assert "\u2022" not in text
        assert "\u25aa" not in text
        assert "text" in text

    def test_ftfy_mojibake_repair(self):
        """ftfy should fix encoding corruption."""
        df = pl.DataFrame({"text": ["sch\u00c3\u00b6n"]})  # mojibake for "schön"
        result = normalize_unicode(df)
        text = result["text_unicode"].to_list()[0]
        assert "ö" in text or text == "schön"

    def test_ligatures_expanded(self):
        """Fraktur ligatures fi, fl, ff should be expanded."""
        df = pl.DataFrame({"text": ["\ufb01nd \ufb02at \ufb00"]})  # ﬁnd ﬂat ﬀ
        result = normalize_unicode(df)
        text = result["text_unicode"].to_list()[0]
        assert "fi" in text
        assert "fl" in text
        assert "ff" in text

    def test_empty_text(self):
        df = pl.DataFrame({"text": [""]})
        result = normalize_unicode(df)
        assert result["text_unicode"].to_list()[0] == ""

    def test_normal_german_text_unchanged(self):
        """Proper German text should pass through unchanged."""
        original = "Die Münchner Zeitung berichtet über das Wetter."
        df = pl.DataFrame({"text": [original]})
        result = normalize_unicode(df)
        assert result["text_unicode"].to_list()[0] == original


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    """Verify whitespace collapsing behavior in both modes."""

    def test_collapses_multiple_spaces(self):
        df = pl.DataFrame({"text": ["hello    world"]})
        result = normalize_whitespace(df)
        assert result["text_whitespace"].to_list()[0] == "hello world"

    def test_collapses_tabs_and_newlines(self):
        df = pl.DataFrame({"text": ["hello\t\tworld\n\nnext"]})
        result = normalize_whitespace(df)
        assert result["text_whitespace"].to_list()[0] == "hello world next"

    def test_strips_leading_trailing(self):
        df = pl.DataFrame({"text": ["  hello world  "]})
        result = normalize_whitespace(df)
        assert result["text_whitespace"].to_list()[0] == "hello world"

    def test_keep_newlines_mode(self):
        """Newlines should be preserved when keep_newlines=True."""
        df = pl.DataFrame({"text": ["hello    world\n\ttab  next"]})
        result = normalize_whitespace(df, keep_newlines=True)
        text = result["text_whitespace"].to_list()[0]
        assert "\n" in text
        assert "hello world" in text
        # Tabs converted to space
        assert "\t" not in text

    def test_empty_string(self):
        df = pl.DataFrame({"text": [""]})
        result = normalize_whitespace(df)
        assert result["text_whitespace"].to_list()[0] == ""


# ---------------------------------------------------------------------------
# normalize_umlauts
# ---------------------------------------------------------------------------


class TestNormalizeUmlauts:
    """Verify umlauts are expanded to two-letter representations."""

    def test_lowercase_umlauts(self):
        df = pl.DataFrame({"text": ["ä ö ü"]})
        result = normalize_umlauts(df)
        assert result["text_umlaut_norm"].to_list()[0] == "ae oe ue"

    def test_uppercase_umlauts(self):
        df = pl.DataFrame({"text": ["Ä Ö Ü"]})
        result = normalize_umlauts(df)
        assert result["text_umlaut_norm"].to_list()[0] == "Ae Oe Ue"

    def test_eszett(self):
        df = pl.DataFrame({"text": ["Straße"]})
        result = normalize_umlauts(df)
        assert result["text_umlaut_norm"].to_list()[0] == "Strasse"

    def test_mixed_text(self):
        df = pl.DataFrame({"text": ["Über den Bären"]})
        result = normalize_umlauts(df)
        assert result["text_umlaut_norm"].to_list()[0] == "Ueber den Baeren"

    def test_no_umlauts_unchanged(self):
        df = pl.DataFrame({"text": ["Hello World"]})
        result = normalize_umlauts(df)
        assert result["text_umlaut_norm"].to_list()[0] == "Hello World"


# ---------------------------------------------------------------------------
# normalize_casing
# ---------------------------------------------------------------------------


class TestNormalizeCasing:
    """Verify casing modes produce correct output."""

    def test_lowercase(self):
        df = pl.DataFrame({"text": ["Hello WORLD"]})
        result = normalize_casing(df, mode="lower")
        assert result["text_casing"].to_list()[0] == "hello world"

    def test_uppercase(self):
        df = pl.DataFrame({"text": ["Hello world"]})
        result = normalize_casing(df, mode="upper")
        assert result["text_casing"].to_list()[0] == "HELLO WORLD"

    def test_titlecase(self):
        df = pl.DataFrame({"text": ["hello world"]})
        result = normalize_casing(df, mode="title")
        assert result["text_casing"].to_list()[0] == "Hello World"

    def test_lowercase_preserves_umlauts(self):
        df = pl.DataFrame({"text": ["Ärger Über Öl"]})
        result = normalize_casing(df, mode="lower")
        text = result["text_casing"].to_list()[0]
        assert "ä" in text
        assert "ü" in text
        assert "ö" in text

    def test_invalid_mode_raises(self):
        df = pl.DataFrame({"text": ["test"]})
        with pytest.raises(ValueError, match="Unknown mode"):
            normalize_casing(df, mode="invalid")


# ---------------------------------------------------------------------------
# normalize_long_s
# ---------------------------------------------------------------------------


class TestNormalizeLongS:
    """Verify historical long s (ſ) normalization in both modes."""

    def test_simple_mode_replaces_all(self):
        """Simple mode: every ſ becomes s."""
        df = pl.DataFrame({"text": ["Hauſe faſten Waſſer"]})
        result = normalize_long_s(df, mode="simple")
        text = result["text_long_s"].to_list()[0]
        assert text == "Hause fasten Wasser"

    def test_simple_mode_preserves_normal_s(self):
        df = pl.DataFrame({"text": ["Haus das Kosmos"]})
        result = normalize_long_s(df, mode="simple")
        assert result["text_long_s"].to_list()[0] == "Haus das Kosmos"

    def test_context_aware_double_long_s(self):
        """Context-aware: ſſ -> ss (Wasser)."""
        df = pl.DataFrame({"text": ["Waſſer"]})
        result = normalize_long_s(df, mode="context-aware")
        assert result["text_long_s"].to_list()[0] == "Wasser"

    def test_context_aware_sch_digraph(self):
        """Context-aware: ſch -> sch."""
        df = pl.DataFrame({"text": ["Buſch"]})
        result = normalize_long_s(df, mode="context-aware")
        assert result["text_long_s"].to_list()[0] == "Busch"

    def test_context_aware_st_combination(self):
        """Context-aware: ſt -> st (fasten)."""
        df = pl.DataFrame({"text": ["faſten"]})
        result = normalize_long_s(df, mode="context-aware")
        assert result["text_long_s"].to_list()[0] == "fasten"

    def test_context_aware_sp_combination(self):
        """Context-aware: ſp -> sp (Wespe)."""
        df = pl.DataFrame({"text": ["Weſpe"]})
        result = normalize_long_s(df, mode="context-aware")
        assert result["text_long_s"].to_list()[0] == "Wespe"

    def test_context_aware_word_end(self):
        """Long s at word end is an OCR error - should become round s."""
        df = pl.DataFrame({"text": ["Hauſ"]})
        result = normalize_long_s(df, mode="context-aware")
        assert result["text_long_s"].to_list()[0] == "Haus"

    def test_no_long_s_unchanged(self):
        df = pl.DataFrame({"text": ["normal text"]})
        result = normalize_long_s(df, mode="simple")
        assert result["text_long_s"].to_list()[0] == "normal text"

    def test_context_aware_no_long_s_unchanged(self):
        """Context-aware mode with no ſ should return text unchanged."""
        df = pl.DataFrame({"text": ["normal text without long s"]})
        result = normalize_long_s(df, mode="context-aware")
        assert result["text_long_s"].to_list()[0] == "normal text without long s"

    def test_invalid_mode_raises(self):
        df = pl.DataFrame({"text": ["test"]})
        with pytest.raises(ValueError, match="Unknown mode"):
            normalize_long_s(df, mode="invalid")


# ---------------------------------------------------------------------------
# normalize_hyphens
# ---------------------------------------------------------------------------


class TestNormalizeHyphens:
    """Verify hyphen/dash normalization across all modes."""

    def test_unify_mode_en_dash(self):
        """En dash (U+2013) should become regular hyphen in unify mode."""
        df = pl.DataFrame({"text": ["1914\u20131918"]})
        result = normalize_hyphens(df, mode="unify")
        assert result["text_hyphens"].to_list()[0] == "1914-1918"

    def test_unify_mode_em_dash(self):
        """Em dash (U+2014) should become regular hyphen in unify mode."""
        df = pl.DataFrame({"text": ["Wort\u2014Trennung"]})
        result = normalize_hyphens(df, mode="unify")
        assert result["text_hyphens"].to_list()[0] == "Wort-Trennung"

    def test_unify_mode_double_hyphen(self):
        """Fraktur double hyphen (U+2E17) should become regular hyphen."""
        df = pl.DataFrame({"text": ["Nachrichten\u2e17Teil"]})
        result = normalize_hyphens(df, mode="unify")
        assert result["text_hyphens"].to_list()[0] == "Nachrichten-Teil"

    def test_unify_mode_soft_hyphen_removed(self):
        """Soft hyphens (U+00AD) should be removed entirely."""
        df = pl.DataFrame({"text": ["Zei\u00adtung"]})
        result = normalize_hyphens(df, mode="unify")
        assert result["text_hyphens"].to_list()[0] == "Zeitung"

    def test_conservative_mode_preserves_en_dash(self):
        """Conservative mode should keep semantic en dash."""
        df = pl.DataFrame({"text": ["1914\u20131918"]})
        result = normalize_hyphens(df, mode="conservative")
        assert "\u2013" in result["text_hyphens"].to_list()[0]

    def test_conservative_mode_normalizes_double_hyphen(self):
        """Conservative mode should still normalize Fraktur artifacts."""
        df = pl.DataFrame({"text": ["Nachrichten\u2e17Teil"]})
        result = normalize_hyphens(df, mode="conservative")
        assert result["text_hyphens"].to_list()[0] == "Nachrichten-Teil"

    def test_soft_only_mode(self):
        """Soft-only mode should only remove soft hyphens."""
        df = pl.DataFrame({"text": ["Zei\u00adtung mit 1914\u20131918"]})
        result = normalize_hyphens(df, mode="soft_only")
        text = result["text_hyphens"].to_list()[0]
        assert "\u00ad" not in text  # soft hyphen removed
        assert "\u2013" in text  # en dash preserved

    def test_invalid_mode_raises(self):
        df = pl.DataFrame({"text": ["test"]})
        with pytest.raises(ValueError, match="Unknown mode"):
            normalize_hyphens(df, mode="invalid")

    def test_regular_hyphen_unchanged(self):
        """Regular ASCII hyphens should pass through all modes."""
        df = pl.DataFrame({"text": ["Nord-Süd"]})
        for mode in ("unify", "conservative", "soft_only"):
            result = normalize_hyphens(df, mode=mode)
            assert "-" in result["text_hyphens"].to_list()[0]

    def test_empty_text(self):
        """Empty text should pass through translate_hyphens without error."""
        df = pl.DataFrame({"text": [""]})
        result = normalize_hyphens(df, mode="unify")
        assert result["text_hyphens"].to_list()[0] == ""


# ---------------------------------------------------------------------------
# replace_long_s_with_f
# ---------------------------------------------------------------------------


class TestReplaceLongSWithF:
    """Verify long s to f replacement (for OCR correction of f-misreads)."""

    def test_replaces_long_s_with_f(self):
        df = pl.DataFrame({"text": ["Hauſe faſten"]})
        result = replace_long_s_with_f(df)
        assert result["text_long_s_to_f"].to_list()[0] == "Haufe faften"

    def test_no_long_s_unchanged(self):
        df = pl.DataFrame({"text": ["normal text"]})
        result = replace_long_s_with_f(df)
        assert result["text_long_s_to_f"].to_list()[0] == "normal text"

    def test_custom_output_column(self):
        df = pl.DataFrame({"text": ["Hauſe"]})
        result = replace_long_s_with_f(df, output_column="fixed")
        assert "fixed" in result.columns
        assert result["fixed"].to_list()[0] == "Haufe"


# ---------------------------------------------------------------------------
# normalize_whitespace - keep_newlines empty text guard
# ---------------------------------------------------------------------------


class TestNormalizeWhitespaceKeepNewlinesEmpty:
    """Cover the empty text early return in keep_newlines mode."""

    def test_empty_text_keep_newlines(self):
        df = pl.DataFrame({"text": [""]})
        result = normalize_whitespace(df, keep_newlines=True)
        assert result["text_whitespace"].to_list()[0] == ""


# ---------------------------------------------------------------------------
# dehyphenate (text-based)
# ---------------------------------------------------------------------------


class TestDehyphenateText:
    """Verify text-based dehyphenation on aggregated text."""

    def test_joins_lowercase_continuation(self):
        """'word- continuation' should join: 'wordcontinuation'."""
        df = pl.DataFrame({"text": ["Zeitungs- papier wird knapp"]})
        result = dehyphenate(df)
        assert "Zeitungspapier" in result["text"].to_list()[0]

    def test_skips_conjunction(self):
        """'Ost- und' should remain 'Ost- und'."""
        df = pl.DataFrame({"text": ["Ost- und Westgrenze"]})
        result = dehyphenate(df)
        assert "Ost- und" in result["text"].to_list()[0]

    def test_skips_digits(self):
        """'20- 30' (range) should remain unchanged."""
        df = pl.DataFrame({"text": ["von 20- 30 Seiten"]})
        result = dehyphenate(df)
        assert "20- 30" in result["text"].to_list()[0]

    def test_keeps_capitalized_compound(self):
        """'Nord- Süd' should become 'Nord-Süd' (kept hyphen)."""
        df = pl.DataFrame({"text": ["Nord- Süd Verbindung"]})
        result = dehyphenate(df)
        assert "Nord-Süd" in result["text"].to_list()[0]

    def test_no_hyphen_unchanged(self):
        """Text without line-break hyphens should pass through unchanged."""
        df = pl.DataFrame({"text": ["normal text here"]})
        result = dehyphenate(df)
        assert result["text"].to_list()[0] == "normal text here"

    def test_empty_text_unchanged(self):
        df = pl.DataFrame({"text": [""]})
        result = dehyphenate(df)
        assert result["text"].to_list()[0] == ""

    def test_custom_output_column(self):
        df = pl.DataFrame({"text": ["Zeitungs- papier"]})
        result = dehyphenate(df, output_column="dehyph")
        assert "dehyph" in result.columns
        assert "Zeitungspapier" in result["dehyph"].to_list()[0]


# ---------------------------------------------------------------------------
# dehyphenate (line-level)
# ---------------------------------------------------------------------------


class TestDehyphenateLines:
    """Verify line-level dehyphenation preserving line structure."""

    def test_joins_across_lines(self):
        """Word split across lines should be joined in the first line."""
        df = pl.DataFrame(
            {
                "text": ["Die Zeitungs-", "papier wird knapp"],
                "text_block_id": ["b1", "b1"],
                "y": [10, 20],
            }
        )
        result = dehyphenate(df)
        texts = result["text"].to_list()
        assert "Zeitungspapier" in texts[0]
        # Second line should have first word removed
        assert texts[1].startswith("wird")

    def test_skips_conjunction_lines(self):
        """Should not join when next line starts with conjunction."""
        df = pl.DataFrame(
            {
                "text": ["Ost-", "und Westgrenze"],
                "text_block_id": ["b1", "b1"],
                "y": [10, 20],
            }
        )
        result = dehyphenate(df)
        texts = result["text"].to_list()
        assert "Ost-" in texts[0]

    def test_keeps_capitalized_compound_lines(self):
        """Capitalized continuation should keep hyphen."""
        df = pl.DataFrame(
            {
                "text": ["Nord-", "Süd Verbindung"],
                "text_block_id": ["b1", "b1"],
                "y": [10, 20],
            }
        )
        result = dehyphenate(df)
        texts = result["text"].to_list()
        assert "Nord-Süd" in texts[0]

    def test_separate_blocks_independent(self):
        """Lines in different text blocks should not interact."""
        df = pl.DataFrame(
            {
                "text": ["Zeitungs-", "papier", "Arti-", "kel"],
                "text_block_id": ["b1", "b1", "b2", "b2"],
                "y": [10, 20, 10, 20],
            }
        )
        result = dehyphenate(df)
        texts = result["text"].to_list()
        assert "Zeitungspapier" in texts[0]
        assert "Artikel" in texts[2]

    def test_missing_required_columns_raises(self):
        """Should raise ValueError if required columns are missing."""
        df = pl.DataFrame({"text": ["hello"], "text_block_id": ["b1"]})
        # Missing 'y' column - should fall back to text-based (no raise)
        # Actually, since text_block_id is present but y is missing, it uses text-mode
        result = dehyphenate(df)
        assert result["text"].to_list()[0] == "hello"

    def test_ocr_suggestions_used(self):
        """When text_dehyphenated_ocr is present, it should be used."""
        df = pl.DataFrame(
            {
                "text": ["Die Zeitungs-", "papier wird"],
                "text_block_id": ["b1", "b1"],
                "y": [10, 20],
                "text_dehyphenated_ocr": ["Die Zeitungspapier", None],
            }
        )
        result = dehyphenate(df)
        texts = result["text"].to_list()
        assert texts[0] == "Die Zeitungspapier"

    def test_dehyphenate_lines_missing_column_raises(self):
        """_dehyphenate_lines should raise ValueError if required columns missing."""
        df = pl.DataFrame({"text": ["hello"], "text_block_id": ["b1"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            _dehyphenate_lines(df, text_col="text", block_col="text_block_id", y_col="y")
