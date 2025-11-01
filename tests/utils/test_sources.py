"""
Tests for source configuration management utilities.
"""

import json
from pathlib import Path

import pytest

from newspaper_explorer.utils.sources import (
    get_source_paths,
    list_available_sources,
    load_source_config,
)


class TestListAvailableSources:
    """Test cases for list_available_sources function."""

    def test_list_sources_with_existing_sources(self, tmp_path, monkeypatch):
        """Test listing sources when source files exist."""
        # Create mock sources directory
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        # Create multiple source files
        (sources_dir / "source1.json").write_text('{"dataset_name": "source1"}')
        (sources_dir / "source2.json").write_text('{"dataset_name": "source2"}')
        (sources_dir / "another_source.json").write_text('{"dataset_name": "another_source"}')

        # Mock the config to use tmp_path
        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        # Test
        sources = list_available_sources()

        assert isinstance(sources, list)
        assert len(sources) == 3
        assert "source1" in sources
        assert "source2" in sources
        assert "another_source" in sources

    def test_list_sources_empty_directory(self, tmp_path, monkeypatch):
        """Test listing sources when directory is empty."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        sources = list_available_sources()

        assert isinstance(sources, list)
        assert len(sources) == 0

    def test_list_sources_nonexistent_directory(self, tmp_path, monkeypatch):
        """Test listing sources when directory doesn't exist."""
        sources_dir = tmp_path / "nonexistent_sources"

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        sources = list_available_sources()

        assert isinstance(sources, list)
        assert len(sources) == 0

    def test_list_sources_natural_sorting(self, tmp_path, monkeypatch):
        """Test that sources are naturally sorted."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        # Create files in non-sorted order
        (sources_dir / "source10.json").write_text('{"dataset_name": "source10"}')
        (sources_dir / "source2.json").write_text('{"dataset_name": "source2"}')
        (sources_dir / "source1.json").write_text('{"dataset_name": "source1"}')
        (sources_dir / "source20.json").write_text('{"dataset_name": "source20"}')

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        sources = list_available_sources()

        # Natural sort should give: 1, 2, 10, 20 (not 1, 10, 2, 20)
        assert sources == ["source1", "source2", "source10", "source20"]

    def test_list_sources_ignores_non_json_files(self, tmp_path, monkeypatch):
        """Test that non-JSON files are ignored."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        (sources_dir / "valid_source.json").write_text('{"dataset_name": "valid"}')
        (sources_dir / "not_json.txt").write_text("ignored")
        (sources_dir / "also_ignored.xml").write_text("<xml/>")

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        sources = list_available_sources()

        assert len(sources) == 1
        assert sources[0] == "valid_source"


class TestLoadSourceConfig:
    """Test cases for load_source_config function."""

    def test_load_valid_source_config(self, tmp_path, monkeypatch):
        """Test loading a valid source configuration."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        # Create a valid source config
        config_data = {
            "dataset_name": "test_source",
            "data_type": "xml_ocr",
            "metadata": {
                "newspaper_title": "Test Newspaper",
                "language": "de",
                "years_available": "1900-1920",
            },
            "loading": {"pattern": "**/fulltext/*.xml", "compression": "zstd"},
            "parts": [
                {
                    "name": "test_1900",
                    "url": "https://example.com/test.tar.gz",
                    "years": "1900",
                    "md5": "abc123",
                    "size": "1 GB",
                }
            ],
        }

        (sources_dir / "test_source.json").write_text(json.dumps(config_data))

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        # Test
        loaded_config = load_source_config("test_source")

        assert loaded_config["dataset_name"] == "test_source"
        assert loaded_config["data_type"] == "xml_ocr"
        assert loaded_config["metadata"]["newspaper_title"] == "Test Newspaper"
        assert loaded_config["metadata"]["language"] == "de"
        assert loaded_config["loading"]["pattern"] == "**/fulltext/*.xml"
        assert len(loaded_config["parts"]) == 1
        assert loaded_config["parts"][0]["name"] == "test_1900"

    def test_load_nonexistent_source(self, tmp_path, monkeypatch):
        """Test loading a source that doesn't exist."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        # Create one source for the error message
        (sources_dir / "existing_source.json").write_text('{"dataset_name": "exists"}')

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        # Test
        with pytest.raises(ValueError) as excinfo:
            load_source_config("nonexistent")

        error_msg = str(excinfo.value)
        assert "Source 'nonexistent' not found" in error_msg
        assert "existing_source" in error_msg

    def test_load_config_with_unicode(self, tmp_path, monkeypatch):
        """Test loading a config with Unicode characters."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        config_data = {
            "dataset_name": "unicode_test",
            "data_type": "xml_ocr",
            "metadata": {
                "newspaper_title": "Tägliche Österreichische Zeitung",
                "language": "de",
                "description": "Historical newspaper with äöü characters",
            },
            "loading": {"pattern": "**/*.xml", "compression": "gzip"},
            "parts": [],
        }

        (sources_dir / "unicode_test.json").write_text(
            json.dumps(config_data, ensure_ascii=False), encoding="utf-8"
        )

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        loaded_config = load_source_config("unicode_test")

        assert loaded_config["metadata"]["newspaper_title"] == "Tägliche Österreichische Zeitung"
        assert "äöü" in loaded_config["metadata"]["description"]

    def test_load_config_invalid_json(self, tmp_path, monkeypatch):
        """Test loading a config file with invalid JSON."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        # Create invalid JSON file
        (sources_dir / "invalid.json").write_text("{invalid json content")

        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.sources_dir = sources_dir
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        with pytest.raises(json.JSONDecodeError):
            load_source_config("invalid")


class TestGetSourcePaths:
    """Test cases for get_source_paths function."""

    def test_get_paths_basic(self, tmp_path, monkeypatch):
        """Test getting paths for a basic source configuration."""
        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.data_dir = tmp_path
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        config_data = {
            "dataset_name": "test_dataset",
            "data_type": "xml_ocr",
        }

        paths = get_source_paths(config_data)

        assert isinstance(paths, dict)
        assert "raw_dir" in paths
        assert "text_dir" in paths
        assert "images_dir" in paths
        assert "output_file" in paths

        # Check path structure
        assert paths["raw_dir"] == tmp_path / "raw" / "test_dataset" / "xml_ocr"
        assert paths["text_dir"] == tmp_path / "raw" / "test_dataset" / "text"
        assert paths["images_dir"] == tmp_path / "raw" / "test_dataset" / "images"
        assert (
            paths["output_file"]
            == tmp_path / "raw" / "test_dataset" / "text" / "test_dataset_lines.parquet"
        )

    def test_get_paths_all_paths_are_pathlib(self, tmp_path, monkeypatch):
        """Test that all returned paths are Path objects."""
        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.data_dir = tmp_path
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        config_data = {
            "dataset_name": "test",
            "data_type": "xml_ocr",
        }

        paths = get_source_paths(config_data)

        for path_key, path_value in paths.items():
            assert isinstance(path_value, Path), f"{path_key} should be a Path object"

    def test_get_paths_different_data_types(self, tmp_path, monkeypatch):
        """Test path generation with different data types."""
        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.data_dir = tmp_path
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        # Test with different data type
        config_data = {
            "dataset_name": "test_dataset",
            "data_type": "pdf_ocr",
        }

        paths = get_source_paths(config_data)

        assert paths["raw_dir"] == tmp_path / "raw" / "test_dataset" / "pdf_ocr"
        # text_dir and images_dir should still use standard names
        assert paths["text_dir"] == tmp_path / "raw" / "test_dataset" / "text"
        assert paths["images_dir"] == tmp_path / "raw" / "test_dataset" / "images"

    def test_get_paths_different_dataset_names(self, tmp_path, monkeypatch):
        """Test path generation with different dataset names."""
        from newspaper_explorer.config.base import Config

        test_config = Config()
        test_config.data_dir = tmp_path
        monkeypatch.setattr("newspaper_explorer.utils.sources.get_config", lambda: test_config)

        config_data = {
            "dataset_name": "another_newspaper",
            "data_type": "xml_ocr",
        }

        paths = get_source_paths(config_data)

        assert paths["raw_dir"] == tmp_path / "raw" / "another_newspaper" / "xml_ocr"
        assert paths["text_dir"] == tmp_path / "raw" / "another_newspaper" / "text"
        assert paths["images_dir"] == tmp_path / "raw" / "another_newspaper" / "images"
        assert (
            paths["output_file"]
            == tmp_path / "raw" / "another_newspaper" / "text" / "another_newspaper_lines.parquet"
        )


@pytest.mark.integration
class TestSourcesIntegration:
    """Integration tests using actual source configurations."""

    def test_load_actual_der_tag_config(self):
        """Test loading the actual der_tag source configuration."""
        # This test will only pass if der_tag.json exists in the real sources dir
        try:
            config = load_source_config("der_tag")

            # Verify expected structure
            assert config["dataset_name"] == "der_tag"
            assert config["data_type"] == "xml_ocr"
            assert "metadata" in config
            assert "loading" in config
            assert "parts" in config

            # Verify metadata
            assert config["metadata"]["newspaper_title"] == "Der Tag"
            assert config["metadata"]["language"] == "de"

            # Verify loading config
            assert config["loading"]["pattern"] == "**/fulltext/*.xml"

            # Verify parts exist
            assert isinstance(config["parts"], list)
            assert len(config["parts"]) > 0

            # Verify first part structure
            first_part = config["parts"][0]
            assert "name" in first_part
            assert "url" in first_part
            assert "md5" in first_part

        except ValueError as e:
            # If der_tag doesn't exist, skip this test
            pytest.skip(f"der_tag source not available: {e}")

    def test_list_actual_sources(self):
        """Test listing actual sources in the project."""
        sources = list_available_sources()

        # Should be a list
        assert isinstance(sources, list)

        # If there are sources, they should be valid
        for source_name in sources:
            assert isinstance(source_name, str)
            assert len(source_name) > 0
            # Should not have .json extension
            assert not source_name.endswith(".json")

    def test_get_paths_for_actual_source(self):
        """Test getting paths for an actual source."""
        sources = list_available_sources()

        if not sources:
            pytest.skip("No sources available")

        # Load first available source
        config = load_source_config(sources[0])
        paths = get_source_paths(config)

        # Verify all expected keys exist
        assert "raw_dir" in paths
        assert "text_dir" in paths
        assert "images_dir" in paths
        assert "output_file" in paths

        # Verify paths contain dataset name
        dataset_name = config["dataset_name"]
        assert dataset_name in str(paths["raw_dir"])
        assert dataset_name in str(paths["text_dir"])
        assert dataset_name in str(paths["images_dir"])
        assert dataset_name in str(paths["output_file"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
