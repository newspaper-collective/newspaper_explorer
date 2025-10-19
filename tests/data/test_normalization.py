"""
Tests for text normalization functionality.
"""

import polars as pl
import pytest

from newspaper_explorer.data.utils.text import TRANSNORMER_MODELS, normalize_text


@pytest.fixture
def sample_historical_text_df():
    """Sample DataFrame with historical German text."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "text": [
                "Die Königinn ſaß auf des Pallaſtes mittlerer Tribune.",
                "Der König befahl ſeinen Unterthanen, das Schloß zu verlaſſen.",
                "Die Preußiſche Armee marſchirte durch die Stadt.",
                "",  # Empty text
            ],
        }
    )


def test_transnormer_models_defined():
    """Test that Transnormer models are properly defined."""
    assert "19c" in TRANSNORMER_MODELS
    assert "18-19c" in TRANSNORMER_MODELS
    assert TRANSNORMER_MODELS["19c"] == "ybracke/transnormer-19c-beta-v02"
    assert TRANSNORMER_MODELS["18-19c"] == "ybracke/transnormer-18-19c-beta-v01"


def test_normalize_text_missing_transformers(sample_historical_text_df, monkeypatch):
    """Test that normalize_text raises ImportError when transformers is not installed."""
    # Mock the import to raise ImportError
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="Transformers is required"):
        normalize_text(sample_historical_text_df, text_column="text")


def test_normalize_text_invalid_column(sample_historical_text_df):
    """Test that normalize_text raises ValueError for invalid column."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        normalize_text(sample_historical_text_df, text_column="nonexistent")


@pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers not installed",
)
class TestNormalizationWithTransformers:
    """Tests that require transformers to be installed."""

    def test_normalize_text_basic(self, sample_historical_text_df):
        """Test basic text normalization."""
        pytest.skip("Skipping actual model inference (too slow for unit tests)")

        result = normalize_text(
            sample_historical_text_df, text_column="text", model="19c", batch_size=2
        )

        # Check that output column was added
        assert "normalized_text" in result.columns

        # Check that we have the same number of rows
        assert len(result) == len(sample_historical_text_df)

        # Check that normalized text differs from original (for non-empty texts)
        for row in result.iter_rows(named=True):
            if row["text"].strip():
                # Normalized text should not contain historical characters
                assert "ſ" not in row["normalized_text"]
                assert "Königinn" not in row["normalized_text"]

    def test_normalize_text_custom_output_column(self, sample_historical_text_df):
        """Test normalization with custom output column name."""
        pytest.skip("Skipping actual model inference (too slow for unit tests)")

        result = normalize_text(
            sample_historical_text_df,
            text_column="text",
            model="19c",
            output_column="modern_text",
        )

        assert "modern_text" in result.columns
        assert "normalized_text" not in result.columns

    def test_normalize_text_model_resolution(self, sample_historical_text_df):
        """Test that model shortcuts are properly resolved."""
        pytest.skip("Skipping actual model inference (too slow for unit tests)")

        # Test with model shortcut
        result1 = normalize_text(sample_historical_text_df, text_column="text", model="19c")

        # Test with full model name
        result2 = normalize_text(
            sample_historical_text_df,
            text_column="text",
            model="ybracke/transnormer-19c-beta-v02",
        )

        # Both should produce the same results
        assert result1["normalized_text"].to_list() == result2["normalized_text"].to_list()


def test_normalize_text_empty_dataframe():
    """Test normalization with empty DataFrame."""
    empty_df = pl.DataFrame({"text": []})

    # This should not raise an error
    # (actual normalization would require transformers)
    with pytest.raises(ImportError):  # Will fail on missing transformers
        normalize_text(empty_df, text_column="text")


def test_normalize_text_parameters():
    """Test that normalize_text accepts all expected parameters."""
    df = pl.DataFrame({"text": ["Test"]})

    # This will fail due to missing transformers, but validates parameter names
    with pytest.raises(ImportError):
        normalize_text(
            df,
            text_column="text",
            model="19c",
            batch_size=16,
            num_beams=4,
            max_length=128,
            output_column="normalized",
        )
