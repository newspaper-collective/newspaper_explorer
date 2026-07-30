"""
Tests for lemmatization functions in data preprocessing.

Verifies that spaCy and GermaLemma lemmatizers produce correct lemma output.
These tests require spaCy's de_core_news_sm model to be installed.
"""

from unittest.mock import patch

import polars as pl
import pytest
import spacy

from newspaper_explorer.data.preprocessing.lemmatization import (
    lemmatize_germalemma,
    lemmatize_spacy,
)

# Skip all tests if model not available
try:
    spacy.load("de_core_news_sm")
    HAS_SPACY_MODEL = True
except OSError:
    HAS_SPACY_MODEL = False

pytestmark = pytest.mark.skipif(not HAS_SPACY_MODEL, reason="de_core_news_sm not installed")


# ---------------------------------------------------------------------------
# lemmatize_spacy
# ---------------------------------------------------------------------------


class TestLemmatizeSpacy:
    """Verify spaCy lemmatizer produces correct lemmas for German text."""

    def test_verb_lemmatization(self):
        """Conjugated verbs should be reduced to infinitive."""
        df = pl.DataFrame({"text": ["Er ging nach Hause"]})
        result = lemmatize_spacy(df)
        text = result["text_lemma"].to_list()[0]
        # "ging" -> "gehen" (past tense -> infinitive)
        assert "gehen" in text.lower() or "ging" not in text

    def test_noun_plural_to_singular(self):
        """Plural nouns should be reduced to singular."""
        df = pl.DataFrame({"text": ["Die Zeitungen berichten"]})
        result = lemmatize_spacy(df)
        text = result["text_lemma"].to_list()[0]
        # "Zeitungen" -> "Zeitung"
        assert "Zeitung" in text

    def test_preserves_word_count(self):
        """Lemmatization should not change the number of tokens."""
        df = pl.DataFrame({"text": ["Der Mann ging in die Stadt"]})
        result = lemmatize_spacy(df)
        original_count = len("Der Mann ging in die Stadt".split())
        lemma_count = len(result["text_lemma"].to_list()[0].split())
        assert lemma_count == original_count

    def test_custom_output_column(self):
        df = pl.DataFrame({"text": ["Die Kinder spielten"]})
        result = lemmatize_spacy(df, output_column="lemmas")
        assert "lemmas" in result.columns

    def test_empty_text(self):
        df = pl.DataFrame({"text": [""]})
        result = lemmatize_spacy(df)
        assert result["text_lemma"].to_list()[0] == ""

    def test_batch_processing(self):
        """Multiple rows should be processed correctly."""
        df = pl.DataFrame(
            {
                "text": [
                    "Die Häuser stehen",
                    "Er lief schnell",
                    "Sie kauften Bücher",
                ]
            }
        )
        result = lemmatize_spacy(df, batch_size=2)
        assert len(result) == 3
        # All rows should have lemmatized output
        assert all(len(t) > 0 for t in result["text_lemma"].to_list())


# ---------------------------------------------------------------------------
# lemmatize_germalemma
# ---------------------------------------------------------------------------


class TestLemmatizeGermalemma:
    """Verify GermaLemma produces correct lemmas with POS tag support."""

    def test_noun_lemmatization(self):
        """GermaLemma should lemmatize nouns correctly."""
        df = pl.DataFrame({"text": ["Die Zeitungen berichten"]})
        result = lemmatize_germalemma(df)
        text = result["text_lemma"].to_list()[0]
        # "Zeitungen" (NN plural) -> "Zeitung"
        assert "Zeitung" in text

    def test_unsupported_pos_returns_original(self):
        """Tokens with unsupported POS tags should be returned as-is."""
        # Prepositions, articles, etc. are not supported by GermaLemma
        df = pl.DataFrame({"text": ["in der Stadt"]})
        result = lemmatize_germalemma(df)
        text = result["text_lemma"].to_list()[0]
        # "in" and "der" should be preserved (unsupported POS)
        assert "in" in text

    def test_custom_output_column(self):
        df = pl.DataFrame({"text": ["Die Kinder"]})
        result = lemmatize_germalemma(df, output_column="lemmas")
        assert "lemmas" in result.columns

    def test_empty_text(self):
        df = pl.DataFrame({"text": [""]})
        result = lemmatize_germalemma(df)
        assert result["text_lemma"].to_list()[0] == ""

    def test_batch_processing(self):
        """Multiple rows should be processed correctly."""
        df = pl.DataFrame(
            {
                "text": [
                    "Die Häuser stehen",
                    "Er lief schnell",
                ]
            }
        )
        result = lemmatize_germalemma(df, batch_size=1)
        assert len(result) == 2
        assert all(len(t) > 0 for t in result["text_lemma"].to_list())


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestLemmatizationErrorHandling:
    """Verify error handling for missing models."""

    def test_spacy_missing_model_raises(self):
        """lemmatize_spacy should raise OSError if model not found."""
        df = pl.DataFrame({"text": ["test"]})
        with pytest.raises(OSError):
            lemmatize_spacy(df, model="nonexistent_model_xyz")

    def test_germalemma_missing_model_raises(self):
        """lemmatize_germalemma should raise OSError if model not found."""
        df = pl.DataFrame({"text": ["test"]})
        with pytest.raises(OSError):
            lemmatize_germalemma(df, spacy_model="nonexistent_model_xyz")

    @pytest.mark.skipif(not HAS_SPACY_MODEL, reason="de_core_news_sm not installed")
    def test_germalemma_handles_value_error(self):
        """GermaLemma should fall back to original token on ValueError."""
        # Mock GermaLemma.find_lemma to raise ValueError, triggering the except path
        df = pl.DataFrame({"text": ["Die Häuser stehen"]})
        with patch("newspaper_explorer.data.preprocessing.lemmatization.GermaLemma") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.find_lemma.side_effect = ValueError("unknown lemma")
            result = lemmatize_germalemma(df)
            # Tokens with supported POS tags should fall back to original text
            text = result["text_lemma"].to_list()[0]
            assert len(text) > 0
            # find_lemma was called (for tokens with NN/NE/V/ADJ/ADV tags)
            assert mock_instance.find_lemma.call_count > 0
