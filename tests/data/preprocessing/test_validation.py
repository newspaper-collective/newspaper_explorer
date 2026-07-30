"""
Tests for OCR quality validation metrics in data preprocessing.

Verifies that quality scoring, filtering, and summarization produce
correct results based on documented thresholds.
"""

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.validation import (
    CHAR_TOKEN_RATIO_GOOD_THRESHOLD,
    CHAR_TOKEN_RATIO_REVIEW_THRESHOLD,
    OOV_RATE_GOOD_THRESHOLD,
    OOV_RATE_REVIEW_THRESHOLD,
    calculate_quality_metrics,
    filter_by_quality_score,
    summarize_quality,
)

# ---------------------------------------------------------------------------
# calculate_quality_metrics
# ---------------------------------------------------------------------------


class TestCalculateQualityMetrics:
    """Verify quality metric calculations produce correct values."""

    def test_char_token_ratio_calculation(self):
        """Ratio = total_chars / token_count."""
        # "ab cd ef" -> 8 chars (inc spaces), 3 tokens -> ratio = 8/3 ~= 2.67
        df = pl.DataFrame({"text": ["ab cd ef"]})
        result = calculate_quality_metrics(df)
        ratio = result["quality_char_token_ratio"].to_list()[0]
        expected = len("ab cd ef") / 3  # 8/3
        assert abs(ratio - expected) < 0.01

    def test_oov_rate_without_wordlist(self):
        """Without a wordlist, OOV rate should be 0.0."""
        df = pl.DataFrame({"text": ["some random text"]})
        with patch(
            "newspaper_explorer.data.preprocessing.validation.get_wordlist_path",
            return_value=Path("/nonexistent/wordlist.txt"),
        ):
            result = calculate_quality_metrics(df)
        assert result["quality_oov_rate"].to_list()[0] == 0.0

    def test_oov_rate_with_wordlist(self, tmp_path):
        """OOV rate should reflect words not in the provided wordlist."""
        # Create wordlist with "die" and "zeitung", but not "xyzgarbage"
        wordlist = tmp_path / "words.txt"
        wordlist.write_text("die\nzeitung\n")

        df = pl.DataFrame({"text": ["die zeitung xyzgarbage"]})
        result = calculate_quality_metrics(df, german_wordlist_path=str(wordlist))
        oov = result["quality_oov_rate"].to_list()[0]
        # 1 out of 3 words is OOV
        assert abs(oov - 1 / 3) < 0.01

    def test_proper_noun_density(self):
        """Capitalized words (excluding common articles) count as proper nouns."""
        # "Der Minister sprach" -> "Minister" is proper noun, "Der" is excluded
        df = pl.DataFrame({"text": ["Der Minister sprach heute"]})
        result = calculate_quality_metrics(df)
        density = result["quality_proper_noun_density"].to_list()[0]
        # "Minister" is the only proper noun out of 4 tokens
        assert abs(density - 1 / 4) < 0.01

    def test_common_articles_excluded_from_proper_nouns(self):
        """Der, Die, Das, Ein, Eine should not count as proper nouns."""
        df = pl.DataFrame({"text": ["Der Die Das Ein Eine"]})
        result = calculate_quality_metrics(df)
        density = result["quality_proper_noun_density"].to_list()[0]
        assert density == 0.0

    def test_quality_score_good_by_ratio(self):
        """Short words -> low char/token ratio -> 'good' quality (without wordlist)."""
        # Short words: "ab cd ef" = 8/3 ~= 2.67 <= 7.0 threshold
        df = pl.DataFrame({"text": ["ab cd ef gh ij"]})
        with patch(
            "newspaper_explorer.data.preprocessing.validation.get_wordlist_path",
            return_value=Path("/nonexistent/wordlist.txt"),
        ):
            result = calculate_quality_metrics(df)
        assert result["quality_score"].to_list()[0] == "good"

    def test_quality_score_poor_by_ratio(self):
        """Very long words -> high char/token ratio -> 'poor' quality (without wordlist)."""
        # One very long "word" -> ratio = len/1 = very high
        long_word = "a" * 50
        df = pl.DataFrame({"text": [long_word]})
        with patch(
            "newspaper_explorer.data.preprocessing.validation.get_wordlist_path",
            return_value=Path("/nonexistent/wordlist.txt"),
        ):
            result = calculate_quality_metrics(df)
        assert result["quality_score"].to_list()[0] == "poor"

    def test_quality_score_from_oov_when_wordlist_provided(self, tmp_path):
        """When wordlist is provided, quality should be based on OOV rate."""
        wordlist = tmp_path / "words.txt"
        # All words in vocab -> 0% OOV -> "good"
        wordlist.write_text("die\nzeitung\nist\ngut\n")
        df = pl.DataFrame({"text": ["die zeitung ist gut"]})
        result = calculate_quality_metrics(df, german_wordlist_path=str(wordlist))
        assert result["quality_score"].to_list()[0] == "good"

    def test_empty_text_returns_zero_metrics(self):
        df = pl.DataFrame({"text": [""]})
        result = calculate_quality_metrics(df)
        assert result["quality_char_token_ratio"].to_list()[0] == 0.0
        assert result["quality_oov_rate"].to_list()[0] == 0.0

    def test_whitespace_only_returns_poor(self):
        """Whitespace-only text should be scored as 'poor'."""
        df = pl.DataFrame({"text": ["   "]})
        result = calculate_quality_metrics(df)
        assert result["quality_score"].to_list()[0] == "poor"
        assert result["quality_char_token_ratio"].to_list()[0] == 0.0

    def test_wordlist_not_found_warns(self, tmp_path):
        """Non-existent wordlist should warn but not crash, OOV stays 0.0."""
        df = pl.DataFrame({"text": ["some text"]})
        result = calculate_quality_metrics(
            df, german_wordlist_path=str(tmp_path / "nonexistent.txt")
        )
        assert result["quality_oov_rate"].to_list()[0] == 0.0

    def test_oov_rate_all_oov(self, tmp_path):
        """All words OOV should give oov_rate=1.0 and 'poor' quality."""
        wordlist = tmp_path / "words.txt"
        wordlist.write_text("apfel\nbirne\n")
        df = pl.DataFrame({"text": ["xyzgarbage qwerty foobar"]})
        result = calculate_quality_metrics(df, german_wordlist_path=str(wordlist))
        oov = result["quality_oov_rate"].to_list()[0]
        assert oov == 1.0
        assert result["quality_score"].to_list()[0] == "poor"

    def test_oov_rate_review_quality(self, tmp_path):
        """OOV rate between 5-15% should get 'review' quality."""
        wordlist = tmp_path / "words.txt"
        # 9 real words + 1 OOV = 10% OOV rate -> review quality
        real = ["die", "zeitung", "ist", "ein", "gutes", "medium", "fuer", "alle", "leser"]
        wordlist.write_text("\n".join(real) + "\n")
        df = pl.DataFrame({"text": [" ".join(real + ["xyzgarbage"])]})
        result = calculate_quality_metrics(df, german_wordlist_path=str(wordlist))
        assert result["quality_score"].to_list()[0] == "review"

    def test_proper_noun_density_mixed(self):
        """Mix of proper nouns and common words."""
        df = pl.DataFrame({"text": ["Berlin ist eine Stadt von Deutschland"]})
        result = calculate_quality_metrics(df)
        density = result["quality_proper_noun_density"].to_list()[0]
        # "Berlin", "Stadt", "Deutschland" are capitalized (3 out of 6 tokens)
        # "Berlin" not in common_capitalized exclusion set
        assert density > 0

    def test_multiple_rows(self):
        """Metrics should be calculated per row."""
        df = pl.DataFrame({"text": ["short", "a much longer sentence with many words"]})
        result = calculate_quality_metrics(df)
        ratios = result["quality_char_token_ratio"].to_list()
        # Single word: ratio = char_count / 1
        assert ratios[0] == len("short") / 1
        # Multi-word: ratio = total_chars_including_spaces / word_count
        expected = len("a much longer sentence with many words") / 7
        assert ratios[1] == pytest.approx(expected)

    def test_only_spaces_no_tokens_returns_poor(self):
        """Text with only spaces: split() gives empty list, should return poor."""
        df = pl.DataFrame({"text": ["     "]})
        result = calculate_quality_metrics(df)
        assert result["quality_score"][0] == "poor"

    def test_review_quality_from_char_token_ratio(self):
        """Char/token ratio in review range (without vocab) gives 'review'."""
        # Single token "abcdefgh" = 8 chars, ratio=8.0, between GOOD(7.0) and REVIEW(8.0)
        df = pl.DataFrame({"text": ["abcdefgh"]})
        with patch(
            "newspaper_explorer.data.preprocessing.validation.get_wordlist_path",
            return_value=Path("/nonexistent/wordlist.txt"),
        ):
            result = calculate_quality_metrics(df)
        ratio = result["quality_char_token_ratio"][0]
        assert ratio == 8.0
        assert result["quality_score"][0] == "review"


# ---------------------------------------------------------------------------
# filter_by_quality_score
# ---------------------------------------------------------------------------


class TestFilterByQualityScore:
    """Verify quality-based filtering keeps/removes correct rows."""

    @pytest.fixture
    def scored_df(self):
        return pl.DataFrame(
            {
                "text": ["good text", "review text", "poor text"],
                "quality_score": ["good", "review", "poor"],
            }
        )

    def test_filter_good_only(self, scored_df):
        result = filter_by_quality_score(scored_df, min_quality="good")
        assert len(result) == 1
        assert result["quality_score"].to_list() == ["good"]

    def test_filter_review_and_above(self, scored_df):
        result = filter_by_quality_score(scored_df, min_quality="review")
        assert len(result) == 2
        assert set(result["quality_score"].to_list()) == {"good", "review"}

    def test_filter_poor_keeps_all(self, scored_df):
        result = filter_by_quality_score(scored_df, min_quality="poor")
        assert len(result) == 3

    def test_invalid_quality_raises(self, scored_df):
        with pytest.raises(ValueError, match="Invalid min_quality"):
            filter_by_quality_score(scored_df, min_quality="excellent")


# ---------------------------------------------------------------------------
# summarize_quality
# ---------------------------------------------------------------------------


class TestSummarizeQuality:
    """Verify summary statistics are computed correctly."""

    def test_summary_metrics(self):
        df = pl.DataFrame(
            {
                "quality_char_token_ratio": [5.0, 7.0, 9.0],
                "quality_oov_rate": [0.02, 0.10, 0.25],
                "quality_proper_noun_density": [0.1, 0.2, 0.3],
                "quality_score": ["good", "review", "poor"],
            }
        )
        summary = summarize_quality(df)

        assert abs(summary["avg_char_token_ratio"] - 7.0) < 0.01
        assert abs(summary["avg_oov_rate"] - 0.1233) < 0.01
        assert abs(summary["pct_good"] - 1 / 3) < 0.01
        assert abs(summary["pct_review"] - 1 / 3) < 0.01
        assert abs(summary["pct_poor"] - 1 / 3) < 0.01

    def test_summary_with_missing_columns(self):
        """Should handle DataFrames with only some quality columns."""
        df = pl.DataFrame({"quality_char_token_ratio": [5.0, 7.0]})
        summary = summarize_quality(df)
        assert "avg_char_token_ratio" in summary
        assert "avg_oov_rate" not in summary

    def test_threshold_constants_documented_values(self):
        """Verify threshold constants match the documented values from normalize.md."""
        assert OOV_RATE_GOOD_THRESHOLD == 0.05
        assert OOV_RATE_REVIEW_THRESHOLD == 0.15
        assert CHAR_TOKEN_RATIO_GOOD_THRESHOLD == 7.0
        assert CHAR_TOKEN_RATIO_REVIEW_THRESHOLD == 8.0
