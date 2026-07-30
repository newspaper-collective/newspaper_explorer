"""
Tests for TextPreprocessor pipeline integration.

Verifies that the pipeline correctly chains steps, handles column routing,
and produces expected transformations end-to-end.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from newspaper_explorer.data.preprocessing.pipeline import STEP_CONFIG, TextPreprocessor
from newspaper_explorer.data.preprocessing.presets import (
    ALL_PIPELINES,
    ANALYSIS_PIPELINES,
    GENERAL_PIPELINES,
    get_extra_steps,
    get_preset,
    get_preset_info,
    list_presets,
)

# ---------------------------------------------------------------------------
# TextPreprocessor.pipeline() - step chaining
# ---------------------------------------------------------------------------


class TestPipelineStepChaining:
    """Verify that pipeline correctly chains multiple preprocessing steps."""

    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_single_step_produces_output_column(self, preprocessor):
        df = pl.DataFrame({"text": ["Hello World"]})
        result = preprocessor.pipeline(df, steps=["normalize_unicode"], output_column="processed")
        assert "processed" in result.columns

    def test_two_steps_chain_correctly(self, preprocessor):
        """Second step should operate on the output of the first."""
        df = pl.DataFrame({"text": ["  Hauſe  "]})
        result = preprocessor.pipeline(
            df,
            steps=["normalize_long_s", "normalize_whitespace"],
            output_column="processed",
        )
        text = result["processed"].to_list()[0]
        # long s normalized AND whitespace stripped
        assert text == "Hause"
        assert "ſ" not in text

    def test_filter_step_reduces_rows(self, preprocessor):
        """Filter steps should remove rows that don't match criteria."""
        df = pl.DataFrame({"text": ["Good text here", "", "   ", "Also good"]})
        result = preprocessor.pipeline(
            df,
            steps=["filter_empty_lines"],
            output_column="processed",
        )
        assert len(result) == 2

    def test_temp_columns_cleaned_up(self, preprocessor):
        """Intermediate _tmp_ columns should not remain in output."""
        df = pl.DataFrame({"text": ["hello world"]})
        result = preprocessor.pipeline(
            df,
            steps=["normalize_unicode", "normalize_casing"],
            output_column="processed",
        )
        temp_cols = [c for c in result.columns if c.startswith("_tmp_")]
        assert temp_cols == []

    def test_original_columns_preserved(self, preprocessor):
        """All original DataFrame columns should survive the pipeline."""
        df = pl.DataFrame(
            {
                "text": ["Hauſe"],
                "line_id": ["L001"],
                "date": ["1920-01-15"],
            }
        )
        result = preprocessor.pipeline(
            df,
            steps=["normalize_long_s"],
            output_column="processed",
        )
        assert "line_id" in result.columns
        assert "date" in result.columns
        assert result["line_id"].to_list() == ["L001"]


# ---------------------------------------------------------------------------
# TextPreprocessor.pipeline() - step with args (dict form)
# ---------------------------------------------------------------------------


class TestPipelineStepArgs:
    """Verify steps with custom arguments are applied correctly."""

    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_dict_step_with_args(self, preprocessor):
        """Pipeline should accept dict-form steps with custom args."""
        df = pl.DataFrame({"text": ["Hello World"]})
        result = preprocessor.pipeline(
            df,
            steps=[{"name": "normalize_casing", "args": {"mode": "upper"}}],
            output_column="processed",
        )
        assert result["processed"].to_list()[0] == "HELLO WORLD"

    def test_filter_with_custom_threshold(self, preprocessor):
        """Filter steps should respect custom thresholds."""
        df = pl.DataFrame({"text": ["short", "a longer sentence with more words"]})
        result = preprocessor.pipeline(
            df,
            steps=[{"name": "filter_by_word_count", "args": {"min_words": 3}}],
            output_column="processed",
        )
        assert len(result) == 1
        assert "longer" in result["processed"].to_list()[0]


# ---------------------------------------------------------------------------
# End-to-end: realistic preprocessing chains
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    """Verify realistic multi-step pipelines produce expected output."""

    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    @pytest.fixture
    def historical_df(self):
        """Simulate historical newspaper OCR data."""
        return pl.DataFrame(
            {
                "text": [
                    "Die Münchner Zeitung berichtet über daſ Wetter.",
                    "123",  # page number - should be filtered
                    "Nachrichten⸗Teil der Abend⸗Ausgabe",
                    "",  # empty line
                    "Über die ſchöne Kunſt",
                ],
            }
        )

    def test_basic_cleanup(self, preprocessor, historical_df):
        """basic preset: unicode + long_s + whitespace + filter_empty."""
        result = preprocessor.pipeline(
            historical_df,
            steps=get_preset("basic"),
            output_column="processed",
        )
        texts = result["processed"].to_list()
        # Empty line should be filtered out
        assert "" not in [t.strip() for t in texts]
        # Long s should be normalized
        assert all("ſ" not in t for t in texts)

    def test_standard_cleanup(self, preprocessor, historical_df):
        """standard preset: basic + allowed_chars + dehyphenate + filter_empty."""
        result = preprocessor.pipeline(
            historical_df,
            steps=get_preset("standard"),
            output_column="processed",
        )
        texts = result["processed"].to_list()
        # Empty lines filtered
        assert "" not in [t.strip() for t in texts]
        # Long s normalized
        assert all("ſ" not in t for t in texts)

    def test_custom_multi_step_transformation(self, preprocessor):
        """Verify a specific multi-step chain produces exact expected output."""
        df = pl.DataFrame({"text": ["  Die  GROSSE  Zeitung  "]})
        result = preprocessor.pipeline(
            df,
            steps=["normalize_whitespace", "normalize_casing"],
            output_column="processed",
        )
        assert result["processed"].to_list()[0] == "die grosse zeitung"


# ---------------------------------------------------------------------------
# STEP_CONFIG completeness
# ---------------------------------------------------------------------------


class TestStepConfig:
    """Verify STEP_CONFIG is consistent and complete."""

    def test_all_preset_steps_exist_in_config(self):
        """Every step referenced in a preset must be defined in STEP_CONFIG."""
        for preset_name, preset in ALL_PIPELINES.items():
            for step in preset["steps"]:
                step_name = step if isinstance(step, str) else step["name"]
                assert step_name in STEP_CONFIG, (
                    f"Step '{step_name}' from preset '{preset_name}' not in STEP_CONFIG"
                )

    def test_all_config_steps_have_func(self):
        """Every STEP_CONFIG entry must have a 'func' key."""
        for step_name, config in STEP_CONFIG.items():
            assert "func" in config, f"Step '{step_name}' missing 'func' in STEP_CONFIG"
            assert callable(config["func"]), f"Step '{step_name}' func is not callable"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    """Verify preset definitions are valid."""

    def test_get_preset_returns_list(self):
        for name in ALL_PIPELINES:
            steps = get_preset(name)
            assert isinstance(steps, list)
            assert len(steps) > 0

    def test_get_preset_unknown_raises(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent_preset")

    def test_minimal_is_least_steps(self):
        """Minimal preset should have the fewest steps."""
        minimal = get_preset("minimal")
        for name in ALL_PIPELINES:
            if name != "minimal":
                assert len(minimal) <= len(get_preset(name))

    def test_entities_preset_no_lowercase(self):
        """Entity preset should NOT include normalize_casing (case-sensitive NER)."""
        steps = get_preset("entities")
        step_names = [s if isinstance(s, str) else s["name"] for s in steps]
        assert "normalize_casing" not in step_names

    def test_topics_preset_includes_stopwords(self):
        """Topic preset should include stopword removal."""
        steps = get_preset("topics")
        step_names = [s if isinstance(s, str) else s["name"] for s in steps]
        assert "remove_stopwords" in step_names

    def test_emotions_preset_no_lowercase(self):
        """Emotion preset should NOT lowercase (emphasis matters)."""
        steps = get_preset("emotions")
        step_names = [s if isinstance(s, str) else s["name"] for s in steps]
        assert "normalize_casing" not in step_names


# ---------------------------------------------------------------------------
# list_presets
# ---------------------------------------------------------------------------


class TestListPresets:
    """Verify list_presets returns correct categories."""

    def test_list_all(self):
        result = list_presets("all")
        assert result == ALL_PIPELINES

    def test_list_general(self):
        result = list_presets("general")
        assert result == GENERAL_PIPELINES

    def test_list_analysis(self):
        result = list_presets("analysis")
        assert result == ANALYSIS_PIPELINES

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown category"):
            list_presets("invalid")


# ---------------------------------------------------------------------------
# get_preset_info
# ---------------------------------------------------------------------------


class TestGetPresetInfo:
    """Verify get_preset_info returns correct preset metadata."""

    def test_valid_preset_has_description(self):
        info = get_preset_info("standard")
        assert "description" in info
        assert "steps" in info
        assert "use_case" in info

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_info("nonexistent")

    def test_returns_copy(self):
        """Modifying returned info should not affect original."""
        info = get_preset_info("standard")
        info["description"] = "modified"
        assert get_preset_info("standard")["description"] != "modified"


# ---------------------------------------------------------------------------
# get_extra_steps
# ---------------------------------------------------------------------------


class TestGetExtraSteps:
    """Verify get_extra_steps returns only steps not in base preset."""

    def test_identical_preset_returns_empty(self):
        result = get_extra_steps("standard", base_preset="standard")
        assert result == []

    def test_keywords_has_extra_steps(self):
        result = get_extra_steps("keywords", base_preset="standard")
        assert len(result) > 0
        # All returned steps should NOT be in standard
        standard_names = {s if isinstance(s, str) else s["name"] for s in get_preset("standard")}
        for step in result:
            name = step if isinstance(step, str) else step["name"]
            assert name not in standard_names


# ---------------------------------------------------------------------------
# _apply_step - edge cases
# ---------------------------------------------------------------------------


class TestApplyStepEdgeCases:
    """Verify _apply_step error handling and special cases."""

    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_unknown_step_raises(self, preprocessor):
        """Passing an unknown step name should raise ValueError."""
        df = pl.DataFrame({"text": ["hello"]})
        with pytest.raises(ValueError, match="Unknown preprocessing step"):
            preprocessor.pipeline(df, steps=["nonexistent_step_xyz"], output_column="processed")

    def test_transnormer_special_handling(self, preprocessor):
        """Transnormer step should call func with GPU-specific args."""
        df = pl.DataFrame({"text": ["historical text"]})
        mock_func = MagicMock(return_value=df.with_columns(pl.lit("normalized").alias("out")))

        with patch.dict(
            STEP_CONFIG,
            {"modernization_transnormer": {"func": mock_func, "special": "transnormer"}},
        ):
            result = preprocessor.pipeline(
                df,
                steps=["modernization_transnormer"],
                output_column="out",
                batch_size=16,
                num_beams=2,
                num_gpus=1,
            )
            mock_func.assert_called_once()
            call_kwargs = mock_func.call_args
            assert call_kwargs[1]["batch_size"] == 16
            assert call_kwargs[1]["num_beams"] == 2
            assert call_kwargs[1]["num_gpus"] == 1


# ---------------------------------------------------------------------------
# load_previous_metadata
# ---------------------------------------------------------------------------


class TestLoadPreviousMetadata:
    """Verify loading previous preprocessing metadata."""

    def test_no_metadata_returns_none(self, tmp_path):
        """When no metadata file exists, should return None."""
        preprocessor = TextPreprocessor()
        result = preprocessor.load_previous_metadata(tmp_path / "nonexistent.parquet")
        assert result is None

    @patch("newspaper_explorer.data.preprocessing.pipeline.load_metadata")
    @patch("newspaper_explorer.data.preprocessing.pipeline.find_metadata_for_parquet")
    def test_metadata_found(self, mock_find, mock_load, tmp_path):
        """When metadata file exists with PreprocessingMetadata, should return dict."""
        from newspaper_explorer.models.data.metadata import PreprocessingMetadata

        mock_find.return_value = tmp_path / "test.json"
        mock_metadata = MagicMock(spec=PreprocessingMetadata)
        mock_metadata.preprocessing_id = "test-id"
        mock_metadata.steps = ["step1"]
        mock_metadata.to_dict.return_value = {"steps": ["step1"]}
        mock_load.return_value = mock_metadata

        preprocessor = TextPreprocessor()
        result = preprocessor.load_previous_metadata(tmp_path / "test.parquet")
        assert result == {"steps": ["step1"]}

    @patch("newspaper_explorer.data.preprocessing.pipeline.load_metadata")
    @patch("newspaper_explorer.data.preprocessing.pipeline.find_metadata_for_parquet")
    def test_metadata_load_error(self, mock_find, mock_load, tmp_path):
        """When metadata loading fails, should return None."""
        mock_find.return_value = tmp_path / "test.json"
        mock_load.side_effect = ValueError("corrupt metadata")
        preprocessor = TextPreprocessor()
        result = preprocessor.load_previous_metadata(tmp_path / "test.parquet")
        assert result is None


# ---------------------------------------------------------------------------
# _create_metadata
# ---------------------------------------------------------------------------


class TestCreateMetadata:
    """Verify metadata creation for preprocessing results."""

    @patch("newspaper_explorer.data.preprocessing.pipeline.load_source_config")
    def test_creates_valid_metadata(self, mock_load_config):
        """Should create valid PreprocessingMetadata with all fields."""
        mock_load_config.return_value = {"dataset_name": "test_source"}
        preprocessor = TextPreprocessor(source="test_source")
        input_df = pl.DataFrame({"text": ["hello", "world"]})
        output_df = pl.DataFrame({"text": ["hello"], "processed": ["hello"]})

        metadata = preprocessor._create_metadata(
            steps=["normalize_unicode", "filter_empty_lines"],
            parameters={"text_column": "text"},
            input_df=input_df,
            output_df=output_df,
            duration_seconds=1.5,
        )

        assert metadata.source == "test_source"
        assert metadata.steps == ["normalize_unicode", "filter_empty_lines"]
        assert metadata.input_data["row_count"] == 2
        assert metadata.output_data["row_count"] == 1
        assert metadata.duration_seconds == 1.5
        assert metadata.status == "completed"

    def test_requires_source(self):
        """Should raise ValueError if source is None."""
        preprocessor = TextPreprocessor()
        with pytest.raises(ValueError, match="source must be set"):
            preprocessor._create_metadata(
                steps=["normalize_unicode"],
                parameters={},
                input_df=pl.DataFrame({"text": ["a"]}),
                output_df=pl.DataFrame({"text": ["a"]}),
                duration_seconds=0.1,
            )


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """Verify the full run() workflow with I/O."""

    def test_run_requires_source(self):
        """run() should raise ValueError if no source is set."""
        preprocessor = TextPreprocessor()
        with pytest.raises(ValueError, match="Source is required"):
            preprocessor.run(steps=["normalize_unicode"])

    def test_run_file_not_found(self, tmp_path):
        """run() should raise FileNotFoundError if input nonexistent."""
        with patch(
            "newspaper_explorer.data.preprocessing.pipeline.load_source_config"
        ) as mock_config:
            mock_config.return_value = MagicMock(dataset_name="test")
            preprocessor = TextPreprocessor(source="test")
            with pytest.raises(FileNotFoundError):
                preprocessor.run(
                    steps=["normalize_unicode"],
                    input_path=tmp_path / "nonexistent.parquet",
                )

    def test_run_missing_text_column(self, tmp_path):
        """run() should raise ValueError if text column not in DataFrame."""
        # Create a parquet file without the expected text column
        parquet_path = tmp_path / "test.parquet"
        pl.DataFrame({"other_column": ["hello"]}).write_parquet(parquet_path)

        with patch(
            "newspaper_explorer.data.preprocessing.pipeline.load_source_config"
        ) as mock_config:
            mock_config.return_value = MagicMock(dataset_name="test")
            preprocessor = TextPreprocessor(source="test")
            with pytest.raises(ValueError, match="Text column"):
                preprocessor.run(
                    steps=["normalize_unicode"],
                    input_path=parquet_path,
                )

    def test_run_full_workflow_no_save(self, tmp_path):
        """run() with save=False should complete without writing files."""
        parquet_path = tmp_path / "test.parquet"
        pl.DataFrame({"text": ["Hauſe", "Waſſer", "normal"]}).write_parquet(parquet_path)

        with patch(
            "newspaper_explorer.data.preprocessing.pipeline.load_source_config"
        ) as mock_config:
            mock_config.return_value = MagicMock(dataset_name="test")
            preprocessor = TextPreprocessor(source="test")
            result = preprocessor.run(
                steps=["normalize_long_s"],
                input_path=parquet_path,
                save=False,
            )

            assert result.input_rows == 3
            assert result.output_rows == 3
            assert result.duration_seconds > 0
            assert result.sample_original is not None
            assert result.sample_processed is not None

    def test_run_with_sample(self, tmp_path):
        """run() with sample should only process first N rows."""
        parquet_path = tmp_path / "test.parquet"
        pl.DataFrame({"text": ["one", "two", "three", "four", "five"]}).write_parquet(parquet_path)

        with patch(
            "newspaper_explorer.data.preprocessing.pipeline.load_source_config"
        ) as mock_config:
            mock_config.return_value = MagicMock(dataset_name="test")
            preprocessor = TextPreprocessor(source="test")
            result = preprocessor.run(
                steps=["normalize_unicode"],
                input_path=parquet_path,
                sample=2,
                save=False,
            )

            # Input rows is the full file
            assert result.input_rows == 5
            # Output should only have 2 rows (sampled)
            assert result.output_rows == 2

    def test_run_with_save(self, tmp_path):
        """run() with save=True should write results via save_preprocessing_results."""
        parquet_path = tmp_path / "test.parquet"
        pl.DataFrame({"text": ["Hauſe", "normal"]}).write_parquet(parquet_path)

        with (
            patch(
                "newspaper_explorer.data.preprocessing.pipeline.load_source_config"
            ) as mock_config,
            patch(
                "newspaper_explorer.data.preprocessing.pipeline.save_preprocessing_results"
            ) as mock_save,
        ):
            mock_config.return_value = MagicMock(dataset_name="test")
            mock_save.return_value = {
                "output_dir": tmp_path / "output",
                "results_path": tmp_path / "output" / "results.parquet",
                "metadata_path": tmp_path / "output" / "metadata.json",
            }
            # Create the results file so stat() works
            (tmp_path / "output").mkdir()
            (tmp_path / "output" / "results.parquet").write_bytes(b"fake")

            preprocessor = TextPreprocessor(source="test")
            result = preprocessor.run(
                steps=["normalize_long_s"],
                input_path=parquet_path,
                save=True,
            )

            mock_save.assert_called_once()
            assert result.results_path == tmp_path / "output" / "results.parquet"

    def test_run_with_previous_preprocessing(self, tmp_path):
        """run() should log previous preprocessing if metadata found."""
        parquet_path = tmp_path / "test.parquet"
        pl.DataFrame({"text": ["hello"]}).write_parquet(parquet_path)

        with patch(
            "newspaper_explorer.data.preprocessing.pipeline.load_source_config"
        ) as mock_config:
            mock_config.return_value = MagicMock(dataset_name="test")
            preprocessor = TextPreprocessor(source="test")
            preprocessor.load_previous_metadata = MagicMock(
                return_value={"steps": ["normalize_unicode", "filter_empty_lines"]}
            )

            result = preprocessor.run(
                steps=["normalize_long_s"],
                input_path=parquet_path,
                save=False,
            )
            assert result.output_rows == 1

    def test_run_default_input_path(self, tmp_path):
        """run() without input_path should resolve default from config."""
        # Set up the default path structure: parsed_dir/{dataset}/textblocks.parquet
        parsed_dir = tmp_path / "parsed"
        parquet_dir = parsed_dir / "test_ds"
        parquet_dir.mkdir(parents=True)
        parquet_path = parquet_dir / "textblocks.parquet"
        pl.DataFrame({"text": ["hello world"]}).write_parquet(parquet_path)

        with patch(
            "newspaper_explorer.data.preprocessing.pipeline.load_source_config"
        ) as mock_config:
            mock_config.return_value = MagicMock(dataset_name="test_ds")
            preprocessor = TextPreprocessor(source="test")
            # Override internal config parsed_dir to point to our tmp_path
            preprocessor._config = MagicMock(parsed_dir=parsed_dir)

            result = preprocessor.run(
                steps=["normalize_unicode"],
                save=False,
            )
            assert result.input_rows == 1
            assert result.output_rows == 1
