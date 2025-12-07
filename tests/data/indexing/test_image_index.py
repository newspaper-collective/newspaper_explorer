"""Tests for image_index module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from newspaper_explorer.data.indexing.image_index import ImageIndexer


class TestImageIndexer:
    """Tests for ImageIndexer class."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock config pointing to temp directory."""
        config = MagicMock()
        config.data_dir = str(tmp_path / "data")
        return config

    @pytest.fixture
    def source_dir(self, tmp_path: Path) -> Path:
        """Create source directory structure."""
        source = tmp_path / "data" / "raw" / "test_source"
        (source / "images").mkdir(parents=True)
        (source / "xml_ocr").mkdir(parents=True)
        return source

    @pytest.fixture
    def indexer(self, mock_config: MagicMock, source_dir: Path) -> ImageIndexer:
        """Create an ImageIndexer with mocked dependencies."""
        with patch(
            "newspaper_explorer.data.indexing.image_index.get_config",
            return_value=mock_config,
        ):
            return ImageIndexer("test_source")

    def test_init_sets_paths_correctly(self, indexer: ImageIndexer, tmp_path: Path) -> None:
        """Test that initialization sets up correct paths."""
        assert indexer.source_name == "test_source"
        assert indexer.source_id == "test_source"  # Falls back when no config
        assert "test_source" in str(indexer.images_dir)
        assert "test_source" in str(indexer.index_path)
        assert str(indexer.index_path).endswith("image_index.parquet")

    def test_init_always_uses_source_name(self, mock_config: MagicMock) -> None:
        """Test that source_name is always used, not ZDB source ID."""
        with patch(
            "newspaper_explorer.data.indexing.image_index.get_config",
            return_value=mock_config,
        ):
            indexer = ImageIndexer("test_source")

        # source_id should always equal source_name for consistent ID generation
        assert indexer.source_id == "test_source"

    def test_create_index_returns_empty_when_no_images_dir(self, indexer: ImageIndexer) -> None:
        """Test that create_index returns empty DataFrame when images dir missing."""
        # Remove the images directory
        indexer.images_dir.rmdir()

        result = indexer.create_index()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_create_index_returns_empty_when_no_images(
        self, indexer: ImageIndexer, source_dir: Path
    ) -> None:
        """Test that create_index returns empty DataFrame when no images found."""
        # images_dir exists but is empty
        result = indexer.create_index()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_load_index_returns_none_when_no_index_file(self, indexer: ImageIndexer) -> None:
        """Test that load_index returns None when index file doesn't exist."""
        result = indexer.load_index()
        assert result is None

    def test_load_index_returns_dataframe_when_exists(self, indexer: ImageIndexer) -> None:
        """Test that load_index returns DataFrame when index file exists."""
        # Create a sample index file
        sample_df = pl.DataFrame(
            {
                "image_path": ["1920/01/15/01/max_1.jpg"],
                "source_id": ["test_source"],
                "year": [1920],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(indexer.index_path)

        result = indexer.load_index()

        assert result is not None
        assert len(result) == 1
        assert result["year"][0] == 1920

    def test_get_stats_returns_zeros_when_no_index(self, indexer: ImageIndexer) -> None:
        """Test that get_stats returns zero values when no index exists."""
        stats = indexer.get_stats()

        assert stats["total_images"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["total_size_gb"] == 0.0
        assert stats["years"] == 0
        assert stats["min_year"] is None
        assert stats["max_year"] is None

    def test_get_stats_returns_correct_values(self, indexer: ImageIndexer) -> None:
        """Test that get_stats returns correct statistics."""
        # Create a sample index file
        sample_df = pl.DataFrame(
            {
                "image_path": [
                    "1920/01/15/01/max_1.jpg",
                    "1920/01/16/01/max_1.jpg",
                    "1921/02/01/01/max_1.jpg",
                ],
                "source_id": ["test_source", "test_source", "test_source"],
                "year": [1920, 1920, 1921],
                "file_size_bytes": [1000, 2000, 3000],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(indexer.index_path)

        stats = indexer.get_stats()

        assert stats["total_images"] == 3
        assert stats["total_size_bytes"] == 6000
        assert stats["years"] == 2
        assert stats["min_year"] == 1920
        assert stats["max_year"] == 1921

    def test_get_sample_images_returns_empty_when_no_index(self, indexer: ImageIndexer) -> None:
        """Test that get_sample_images returns empty DataFrame when no index."""
        result = indexer.get_sample_images()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_get_sample_images_respects_limit(self, indexer: ImageIndexer) -> None:
        """Test that get_sample_images respects the limit parameter."""
        # Create index with many images
        sample_df = pl.DataFrame(
            {
                "image_path": [f"1920/01/{i:02d}/01/max_1.jpg" for i in range(1, 20)],
                "source_id": ["test_source"] * 19,
                "year": [1920] * 19,
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(indexer.index_path)

        result = indexer.get_sample_images(limit=5, spread_years=False)

        assert len(result) == 5

    def test_get_sample_images_filters_by_year(self, indexer: ImageIndexer) -> None:
        """Test that get_sample_images filters by min_year and max_year."""
        sample_df = pl.DataFrame(
            {
                "image_path": [
                    "1918/01/01/01/max_1.jpg",
                    "1919/01/01/01/max_1.jpg",
                    "1920/01/01/01/max_1.jpg",
                    "1921/01/01/01/max_1.jpg",
                    "1922/01/01/01/max_1.jpg",
                ],
                "source_id": ["test_source"] * 5,
                "year": [1918, 1919, 1920, 1921, 1922],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(indexer.index_path)

        result = indexer.get_sample_images(
            limit=10, min_year=1919, max_year=1921, spread_years=False
        )

        years = result["year"].to_list()
        assert all(1919 <= y <= 1921 for y in years)
        assert 1918 not in years
        assert 1922 not in years

    def test_get_sample_images_spreads_across_years(self, indexer: ImageIndexer) -> None:
        """Test that get_sample_images spreads samples across years when enabled."""
        sample_df = pl.DataFrame(
            {
                "image_path": [
                    "1918/01/01/01/max_1.jpg",
                    "1919/01/01/01/max_1.jpg",
                    "1920/01/01/01/max_1.jpg",
                    "1921/01/01/01/max_1.jpg",
                ],
                "source_id": ["test_source"] * 4,
                "year": [1918, 1919, 1920, 1921],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(indexer.index_path)

        result = indexer.get_sample_images(limit=4, spread_years=True)

        # Should have samples from different years
        unique_years = result["year"].unique().to_list()
        assert len(unique_years) >= 2  # At least some spread


class TestBuildMetsCache:
    """Tests for _build_mets_cache method."""

    @pytest.fixture
    def indexer(self, tmp_path: Path) -> ImageIndexer:
        """Create an ImageIndexer with mocked dependencies."""
        mock_config = MagicMock()
        mock_config.data_dir = str(tmp_path / "data")

        # Create directory structure
        source_dir = tmp_path / "data" / "raw" / "test_source"
        (source_dir / "images").mkdir(parents=True)
        (source_dir / "xml_ocr").mkdir(parents=True)

        with patch(
            "newspaper_explorer.data.indexing.image_index.get_config",
            return_value=mock_config,
        ):
            return ImageIndexer("test_source")

    def test_returns_empty_dict_when_no_mets_files(self, indexer: ImageIndexer) -> None:
        """Test that empty dict is returned when no METS files found."""
        with patch(
            "newspaper_explorer.data.indexing.image_index.find_mets_files",
            return_value=[],
        ):
            result = indexer._build_mets_cache()

        assert result == {}


class TestBuildAltoDimensionCache:
    """Tests for _build_alto_dimension_cache method."""

    @pytest.fixture
    def indexer(self, tmp_path: Path) -> ImageIndexer:
        """Create an ImageIndexer with mocked dependencies."""
        mock_config = MagicMock()
        mock_config.data_dir = str(tmp_path / "data")

        # Create directory structure
        source_dir = tmp_path / "data" / "raw" / "test_source"
        (source_dir / "images").mkdir(parents=True)
        (source_dir / "xml_ocr").mkdir(parents=True)

        with patch(
            "newspaper_explorer.data.indexing.image_index.get_config",
            return_value=mock_config,
        ):
            return ImageIndexer("test_source")

    def test_returns_empty_dict_when_xml_dir_missing(self, indexer: ImageIndexer) -> None:
        """Test that empty dict is returned when xml_dir doesn't exist."""
        indexer.xml_dir.rmdir()
        result = indexer._build_alto_dimension_cache()
        assert result == {}

    def test_returns_empty_dict_when_no_alto_files(self, indexer: ImageIndexer) -> None:
        """Test that empty dict is returned when no ALTO files found."""
        result = indexer._build_alto_dimension_cache()
        assert result == {}
