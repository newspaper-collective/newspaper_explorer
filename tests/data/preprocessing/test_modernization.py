"""
Tests for modernization functions in data preprocessing.

Tests Transnormer (neural model) and DTA-CAB (API) normalization.
All external dependencies (GPU, API, HuggingFace models) are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.modernization import (
    DTA_CAB_API_URL,
    MIN_CSV_PARTS,
    TRANSNORMER_MODELS,
    _call_dtacab_api,
    _chunk_texts_with_cache,
    _detect_device_and_gpus,
    _get_progress_bar_config,
    _JsonCache,
    _log_cache_stats,
    _merge_results_with_cache,
    _normalize_single_text_dtacab,
    _parse_csv_response,
    _parse_dtacab_response,
    _reassemble_chunks,
    _resolve_model_name,
    _setup_dtacab_cache,
    _setup_transnormer_cache,
    _TextCache,
    dta_cab,
    transnormer,
)

# ---------------------------------------------------------------------------
# _TextCache
# ---------------------------------------------------------------------------


class TestTextCache:
    """Verify file-based text caching by SHA256 hash."""

    def test_save_and_get(self, tmp_path):
        cache = _TextCache(tmp_path / "cache")
        cache.save("hello", "world")
        assert cache.get("hello") == "world"

    def test_get_cache_miss(self, tmp_path):
        cache = _TextCache(tmp_path / "cache")
        assert cache.get("nonexistent") is None

    def test_cache_disabled(self, tmp_path):
        cache = _TextCache(tmp_path / "cache", use_cache=False)
        cache.save("hello", "world")
        assert cache.get("hello") is None

    def test_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache"
        _TextCache(cache_dir)
        assert cache_dir.exists()

    def test_corrupt_file_returns_none(self, tmp_path):
        """Corrupt cache file should return None, not crash."""
        cache = _TextCache(tmp_path / "cache")
        # Write invalid binary data
        cache_path = cache._get_cache_path("test")
        cache_path.write_bytes(b"\x80\x81\x82\x83")
        assert cache.get("test") is None

    def test_save_write_error_silent(self, tmp_path):
        """OSError on save should be logged but not raised."""
        cache = _TextCache(tmp_path / "cache")
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            # Should not raise
            cache.save("hello", "world")

    def test_cache_key_deterministic(self, tmp_path):
        cache = _TextCache(tmp_path / "cache")
        key1 = cache._get_cache_key("same text")
        key2 = cache._get_cache_key("same text")
        assert key1 == key2

    def test_different_texts_different_keys(self, tmp_path):
        cache = _TextCache(tmp_path / "cache")
        key1 = cache._get_cache_key("text a")
        key2 = cache._get_cache_key("text b")
        assert key1 != key2


# ---------------------------------------------------------------------------
# _JsonCache
# ---------------------------------------------------------------------------


class TestJsonCache:
    """Verify JSON-based cache for DTA-CAB results."""

    def test_save_and_get(self, tmp_path):
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        cache.save("input", "normalized")
        assert cache.get("input") == "normalized"

    def test_stores_json_format(self, tmp_path):
        """Saved file should be valid JSON with original, normalized, format keys."""
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        cache.save("input text", "output text")
        cache_path = cache._get_cache_path("input text")
        data = json.loads(cache_path.read_text())
        assert data["original"] == "input text"
        assert data["normalized"] == "output text"
        assert data["format"] == "csv"

    def test_corrupt_json_returns_none(self, tmp_path):
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        cache_path = cache._get_cache_path("test")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("not valid json {{{")
        assert cache.get("test") is None

    def test_missing_normalized_key(self, tmp_path):
        """If JSON lacks 'normalized' key, should return None."""
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        cache_path = cache._get_cache_path("test")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"original": "test"}))
        assert cache.get("test") is None

    def test_cache_disabled(self, tmp_path):
        cache = _JsonCache(tmp_path / "cache", use_cache=False, format_name="csv")
        cache.save("hello", "world")
        assert cache.get("hello") is None

    def test_save_write_error_silent(self, tmp_path):
        """OSError on save should be logged but not raised."""
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            cache.save("hello", "world")


# ---------------------------------------------------------------------------
# _get_progress_bar_config
# ---------------------------------------------------------------------------


class TestGetProgressBarConfig:
    """Verify progress bar configuration modes."""

    def test_single_gpu_mode(self):
        desc, pos, leave = _get_progress_bar_config(None)
        assert desc == "Normalizing"
        assert pos is None
        assert leave is True

    def test_multi_gpu_mode(self):
        desc, pos, leave = _get_progress_bar_config(0)
        assert desc == "GPU 0"
        assert pos == 0
        assert leave is False

    def test_multi_gpu_other_id(self):
        desc, pos, leave = _get_progress_bar_config(3)
        assert desc == "GPU 3"
        assert pos == 3


# ---------------------------------------------------------------------------
# _resolve_model_name
# ---------------------------------------------------------------------------


class TestResolveModelName:
    """Verify model shorthand resolution."""

    def test_known_shorthand(self):
        name = _resolve_model_name("19c")
        assert name == TRANSNORMER_MODELS["19c"]

    def test_known_shorthand_18_19c(self):
        name = _resolve_model_name("18-19c")
        assert name == TRANSNORMER_MODELS["18-19c"]

    def test_custom_model_passthrough(self):
        name = _resolve_model_name("custom/my-model")
        assert name == "custom/my-model"


# ---------------------------------------------------------------------------
# _setup_transnormer_cache
# ---------------------------------------------------------------------------


class TestSetupTransnormerCache:
    """Verify cache directory setup."""

    def test_custom_cache_dir(self, tmp_path):
        cache = _setup_transnormer_cache(tmp_path / "my_cache", "19c", use_cache=True)
        assert isinstance(cache, _TextCache)
        assert cache.cache_dir == tmp_path / "my_cache"

    def test_default_cache_dir(self):
        cache = _setup_transnormer_cache(None, "19c", use_cache=True)
        assert isinstance(cache, _TextCache)
        assert "transnormer" in str(cache.cache_dir)

    def test_cache_disabled(self, tmp_path):
        cache = _setup_transnormer_cache(tmp_path / "cache", "19c", use_cache=False)
        assert cache.use_cache is False


# ---------------------------------------------------------------------------
# _detect_device_and_gpus
# ---------------------------------------------------------------------------


class TestDetectDeviceAndGpus:
    """Verify device auto-detection and GPU count validation."""

    def test_explicit_cpu(self):
        device, gpus = _detect_device_and_gpus("cpu", 1)
        assert device == "cpu"
        assert gpus == 1

    def test_explicit_cuda(self):
        device, gpus = _detect_device_and_gpus("cuda", 1)
        assert device == "cuda"
        assert gpus == 1

    def test_integer_device(self):
        device, gpus = _detect_device_and_gpus(0, 1)
        assert device == "cuda:0"
        assert gpus == 1

    @patch("newspaper_explorer.data.preprocessing.modernization.torch")
    def test_auto_detect_no_cuda(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        device, gpus = _detect_device_and_gpus(None, 1)
        assert device == "cpu"
        assert gpus == 1

    @patch("newspaper_explorer.data.preprocessing.modernization.torch")
    def test_multi_gpu_fewer_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        device, gpus = _detect_device_and_gpus("cuda", 4)
        assert device == "cuda"
        assert gpus == 2


# ---------------------------------------------------------------------------
# _log_cache_stats
# ---------------------------------------------------------------------------


class TestLogCacheStats:
    """Verify cache logging (no assertions on log content, just no crash)."""

    def test_with_cache(self):
        _log_cache_stats(100, 50, use_cache=True)

    def test_without_cache(self):
        _log_cache_stats(100, 0, use_cache=False)

    def test_zero_chunks(self):
        _log_cache_stats(0, 0, use_cache=True)


# ---------------------------------------------------------------------------
# _reassemble_chunks
# ---------------------------------------------------------------------------


class TestReassembleChunks:
    """Verify chunk reassembly back into original rows."""

    def test_single_chunk_per_row(self):
        result = _reassemble_chunks(
            chunk_to_row_map=[0, 1, 2],
            normalized_chunks=["a", "b", "c"],
            num_texts=3,
        )
        assert result == ["a", "b", "c"]

    def test_multiple_chunks_per_row(self):
        result = _reassemble_chunks(
            chunk_to_row_map=[0, 0, 1],
            normalized_chunks=["part1", "part2", "single"],
            num_texts=2,
        )
        assert result[0] == "part1 part2"
        assert result[1] == "single"

    def test_empty_chunks(self):
        result = _reassemble_chunks(
            chunk_to_row_map=[],
            normalized_chunks=[],
            num_texts=0,
        )
        assert result == []


# ---------------------------------------------------------------------------
# _merge_results_with_cache
# ---------------------------------------------------------------------------


class TestMergeResultsWithCache:
    """Verify merging processing results with cached results."""

    def test_merges_and_saves(self, tmp_path):
        cache = _TextCache(tmp_path / "cache")
        cached = {0: "cached_result"}
        chunks = ["chunk0", "chunk1", "chunk2"]
        processing_results = [(1, "result1"), (2, "result2")]

        _merge_results_with_cache(processing_results, cached, chunks, cache)

        assert cached == {0: "cached_result", 1: "result1", 2: "result2"}
        # Verify saved to cache
        assert cache.get("chunk1") == "result1"
        assert cache.get("chunk2") == "result2"


# ---------------------------------------------------------------------------
# DTA-CAB helpers
# ---------------------------------------------------------------------------


class TestSetupDtacabCache:
    """Verify DTA-CAB cache setup."""

    def test_custom_cache_dir(self, tmp_path):
        cache = _setup_dtacab_cache(tmp_path / "dtacab", "csv", use_cache=True)
        assert isinstance(cache, _JsonCache)
        assert cache.cache_dir == tmp_path / "dtacab"

    def test_default_cache_dir(self):
        cache = _setup_dtacab_cache(None, "csv", use_cache=True)
        assert isinstance(cache, _JsonCache)
        assert "dtacab" in str(cache.cache_dir)


class TestParseCsvResponse:
    """Verify CSV response parsing from DTA-CAB."""

    def test_basic_csv(self):
        response = "Die\tDie\tdie\tART\nZeitung\tZeitung\tZeitung\tNN"
        result = _parse_csv_response(response)
        assert result == "Die Zeitung"

    def test_skips_comments(self):
        response = "# comment\nword\tnorm"
        result = _parse_csv_response(response)
        assert result == "norm"

    def test_skips_empty_lines(self):
        response = "a\tb\n\nc\td"
        result = _parse_csv_response(response)
        assert result == "b d"

    def test_skips_short_lines(self):
        """Lines with fewer than MIN_CSV_PARTS columns should be skipped."""
        response = "onlyonecolumn\na\tb"
        result = _parse_csv_response(response)
        assert result == "b"


class TestParseDtacabResponse:
    """Verify response format dispatch."""

    def test_csv_format(self):
        result = _parse_dtacab_response("a\tb\n", "csv", "original")
        assert result == "b"

    def test_txt_format(self):
        result = _parse_dtacab_response("  normalized text  ", "txt", "original")
        assert result == "normalized text"

    def test_unsupported_format(self):
        result = _parse_dtacab_response("data", "xml", "original text")
        assert result == "original text"


class TestNormalizeSingleTextDtacab:
    """Verify single-text DTA-CAB normalization with caching."""

    def test_empty_text_passthrough(self, tmp_path):
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        assert _normalize_single_text_dtacab("", cache, "csv", 30) == ""
        assert _normalize_single_text_dtacab("   ", cache, "csv", 30) == "   "

    def test_cache_hit(self, tmp_path):
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        cache.save("input", "cached_result")
        result = _normalize_single_text_dtacab("input", cache, "csv", 30)
        assert result == "cached_result"

    @patch("newspaper_explorer.data.preprocessing.modernization._call_dtacab_api")
    def test_api_called_on_miss(self, mock_api, tmp_path):
        mock_api.return_value = "api_result"
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        result = _normalize_single_text_dtacab("new_text", cache, "csv", 30)
        assert result == "api_result"
        mock_api.assert_called_once_with("new_text", "csv", 30)

    @patch("newspaper_explorer.data.preprocessing.modernization._call_dtacab_api")
    def test_timeout_returns_original(self, mock_api, tmp_path):
        import requests

        mock_api.side_effect = requests.exceptions.Timeout()
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        result = _normalize_single_text_dtacab("text", cache, "csv", 30)
        assert result == "text"

    @patch("newspaper_explorer.data.preprocessing.modernization._call_dtacab_api")
    def test_request_error_returns_original(self, mock_api, tmp_path):
        import requests

        mock_api.side_effect = requests.exceptions.ConnectionError()
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        result = _normalize_single_text_dtacab("text", cache, "csv", 30)
        assert result == "text"


class TestCallDtacabApi:
    """Verify DTA-CAB API call and response parsing."""

    @patch("newspaper_explorer.data.preprocessing.modernization.requests.get")
    def test_calls_api_with_correct_params(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "original\tnormalized"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = _call_dtacab_api("test text", "csv", 30)
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["q"] == "test text"
        assert call_kwargs[1]["params"]["fmt"] == "csv"
        assert call_kwargs[1]["timeout"] == 30


# ---------------------------------------------------------------------------
# transnormer (main function)
# ---------------------------------------------------------------------------


class TestTransnormer:
    """Verify transnormer main function with mocked model."""

    @patch("newspaper_explorer.data.preprocessing.modernization._process_with_single_device")
    @patch("newspaper_explorer.data.preprocessing.modernization._chunk_texts_with_cache")
    @patch("newspaper_explorer.data.preprocessing.modernization._detect_device_and_gpus")
    def test_all_cached(self, mock_detect, mock_chunk, mock_process):
        """When all chunks are cached, model inference should be skipped."""
        mock_detect.return_value = ("cpu", 1)
        # All chunks cached
        mock_chunk.return_value = (["chunk1", "chunk2"], [0, 1], {0: "norm1", 1: "norm2"})

        df = pl.DataFrame({"text": ["hello", "world"]})
        result = transnormer(df, use_cache=True, device="cpu")

        mock_process.assert_not_called()
        assert "text_transnormer" in result.columns
        assert result["text_transnormer"].to_list() == ["norm1", "norm2"]

    @patch("newspaper_explorer.data.preprocessing.modernization._merge_results_with_cache")
    @patch("newspaper_explorer.data.preprocessing.modernization._process_with_single_device")
    @patch("newspaper_explorer.data.preprocessing.modernization._chunk_texts_with_cache")
    @patch("newspaper_explorer.data.preprocessing.modernization._detect_device_and_gpus")
    def test_single_device_processing(self, mock_detect, mock_chunk, mock_process, mock_merge):
        """Single device processing when cache misses exist."""
        mock_detect.return_value = ("cpu", 1)
        mock_chunk.return_value = (["chunk1"], [0], {})
        mock_process.return_value = [(0, "normalized")]
        mock_merge.side_effect = lambda results, cached, chunks, cache: cached.update(
            {idx: text for idx, text in results}
        )

        df = pl.DataFrame({"text": ["hello"]})
        result = transnormer(df, use_cache=False, device="cpu")

        mock_process.assert_called_once()
        assert "text_transnormer" in result.columns

    @patch("newspaper_explorer.data.preprocessing.modernization._merge_results_with_cache")
    @patch("newspaper_explorer.data.preprocessing.modernization._process_with_multi_gpu")
    @patch("newspaper_explorer.data.preprocessing.modernization._chunk_texts_with_cache")
    @patch("newspaper_explorer.data.preprocessing.modernization._detect_device_and_gpus")
    def test_multi_gpu_processing(self, mock_detect, mock_chunk, mock_multi, mock_merge):
        """Multi-GPU processing when num_gpus > 1."""
        mock_detect.return_value = ("cuda", 2)
        mock_chunk.return_value = (["c1", "c2"], [0, 1], {})
        mock_multi.return_value = [(0, "n1"), (1, "n2")]
        mock_merge.side_effect = lambda results, cached, chunks, cache: cached.update(
            {idx: text for idx, text in results}
        )

        df = pl.DataFrame({"text": ["hello", "world"]})
        result = transnormer(df, device="cuda", num_gpus=2, use_cache=False)

        mock_multi.assert_called_once()
        assert "text_transnormer" in result.columns


# ---------------------------------------------------------------------------
# dta_cab (main function)
# ---------------------------------------------------------------------------


class TestDtaCab:
    """Verify dta_cab main function with mocked API."""

    @patch("newspaper_explorer.data.preprocessing.modernization._normalize_single_text_dtacab")
    def test_processes_all_texts(self, mock_normalize):
        mock_normalize.side_effect = lambda text, *args: f"norm_{text}"

        df = pl.DataFrame({"text": ["hello", "world"]})
        result = dta_cab(df, use_cache=False)

        assert "text_dtacab" in result.columns
        assert result["text_dtacab"].to_list() == ["norm_hello", "norm_world"]

    @patch("newspaper_explorer.data.preprocessing.modernization._normalize_single_text_dtacab")
    def test_custom_output_column(self, mock_normalize):
        mock_normalize.side_effect = lambda text, *args: text

        df = pl.DataFrame({"text": ["hello"]})
        result = dta_cab(df, output_column="custom_col", use_cache=False)

        assert "custom_col" in result.columns

    @patch("newspaper_explorer.data.preprocessing.modernization._normalize_single_text_dtacab")
    def test_batch_processing(self, mock_normalize):
        """Should process texts in batches."""
        mock_normalize.side_effect = lambda text, *args: text

        df = pl.DataFrame({"text": ["a", "b", "c", "d", "e"]})
        result = dta_cab(df, batch_size=2, use_cache=False)

        assert len(result) == 5
        assert mock_normalize.call_count == 5

    @patch("newspaper_explorer.data.preprocessing.modernization._normalize_single_text_dtacab")
    def test_cache_hit_counts(self, mock_normalize, tmp_path):
        """With use_cache=True, cache hits should be counted and logged."""
        mock_normalize.side_effect = lambda text, *args: f"norm_{text}"

        df = pl.DataFrame({"text": ["hello", "world"]})

        # Pre-populate a real cache with one entry so cache.get hits
        cache = _JsonCache(tmp_path / "cache", format_name="csv")
        cache.save("hello", "cached_hello")

        with patch(
            "newspaper_explorer.data.preprocessing.modernization._setup_dtacab_cache",
            return_value=cache,
        ):
            result = dta_cab(df, use_cache=True)
            assert "text_dtacab" in result.columns
            assert len(result) == 2


# ---------------------------------------------------------------------------
# _chunk_texts_with_cache
# ---------------------------------------------------------------------------


class TestChunkTextsWithCache:
    """Verify text chunking and cache lookup."""

    def test_empty_text_handled(self, tmp_path):
        """Empty texts should produce a single space chunk."""
        cache = _TextCache(tmp_path / "cache")
        chunks, chunk_map, cached = _chunk_texts_with_cache(["", "  "], 512, cache)
        assert len(chunks) == 2
        assert all(c == " " for c in chunks)
        assert chunk_map == [0, 1]

    def test_cache_hit_detected(self, tmp_path):
        """Cached chunks should appear in cached_results dict."""
        cache = _TextCache(tmp_path / "cache")
        cache.save("hello world", "cached_result")
        chunks, chunk_map, cached = _chunk_texts_with_cache(["hello world"], 512, cache)
        assert 0 in cached
        assert cached[0] == "cached_result"

    def test_cache_miss(self, tmp_path):
        """Uncached chunks should not appear in cached_results."""
        cache = _TextCache(tmp_path / "cache")
        chunks, chunk_map, cached = _chunk_texts_with_cache(["new text"], 512, cache)
        assert len(cached) == 0
        assert chunks == ["new text"]
        assert chunk_map == [0]
