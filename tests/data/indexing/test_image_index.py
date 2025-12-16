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

        assert stats.total_images == 0
        assert stats.total_size_bytes == 0
        assert stats.total_size_gb == 0.0
        assert stats.min_year is None
        assert stats.max_year is None

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

        assert stats.total_images == 3
        assert stats.total_size_bytes == 6000
        assert stats.min_year == 1920
        assert stats.max_year == 1921

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


class TestCreateIndex:
    """Additional tests for create_index method."""

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

    def test_incremental_update_skips_existing_images(self, indexer: ImageIndexer) -> None:
        """Test that incremental update only processes new images."""
        # Create existing index with complete schema
        existing_df = pl.DataFrame(
            {
                "image_path": ["1920/01/01/01/max_1.jpg"],
                "filename": ["max_1.jpg"],
                "file_size_bytes": [1000],
                "file_exists": [True],
                "source_id": ["test_source"],
                "issue_id": ["test_source_1920-01-01_001_1"],
                "page_id": [None],
                "year": [1920],
                "month": [1],
                "day": [1],
                "date": ["1920-01-01"],
                "page_number": [1],
                "edition": [None],
                "issue_number": [None],
                "width": [None],
                "height": [None],
                "alto_width": [None],
                "alto_height": [None],
                "newspaper_title": [None],
                "year_volume": [None],
                "page_count": [None],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        existing_df.write_parquet(indexer.index_path)

        # Create one existing and one new image
        img1 = indexer.images_dir / "1920" / "01" / "01" / "01" / "max_1.jpg"
        img1.parent.mkdir(parents=True, exist_ok=True)
        img1.write_bytes(b"old image")

        img2 = indexer.images_dir / "1920" / "01" / "02" / "01" / "max_1.jpg"
        img2.parent.mkdir(parents=True, exist_ok=True)
        img2.write_bytes(b"new image")

        with patch("newspaper_explorer.data.indexing.image_index.find_mets_files", return_value=[]):
            with patch(
                "newspaper_explorer.data.indexing.image_index.ImageDownloader"
            ) as mock_downloader:
                mock_downloader.return_value.extract_image_references.return_value = []
                result = indexer.create_index(force_rebuild=False)

        # Should have 2 images total (1 existing + 1 new)
        assert len(result) == 2

    def test_force_rebuild_reprocesses_all_images(self, indexer: ImageIndexer) -> None:
        """Test that force_rebuild=True reprocesses all images."""
        # Create existing index
        existing_df = pl.DataFrame(
            {
                "image_path": ["1920/01/01/01/max_1.jpg"],
                "source_id": ["test_source"],
                "year": [1920],
                "file_size_bytes": [1000],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        existing_df.write_parquet(indexer.index_path)

        # Create image
        img = indexer.images_dir / "1920" / "01" / "01" / "01" / "max_1.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"test image")

        with patch("newspaper_explorer.data.indexing.image_index.find_mets_files", return_value=[]):
            with patch(
                "newspaper_explorer.data.indexing.image_index.ImageDownloader"
            ) as mock_downloader:
                mock_downloader.return_value.extract_image_references.return_value = []
                result = indexer.create_index(force_rebuild=True)

        # Should have processed the image (existing index ignored)
        assert len(result) == 1

    def test_handles_multiple_image_formats(self, indexer: ImageIndexer) -> None:
        """Test that indexer handles jpg, jpeg, and png files."""
        # Create images with different extensions
        for ext in ["jpg", "jpeg", "png"]:
            img = indexer.images_dir / "1920" / "01" / "01" / "01" / f"max_1.{ext}"
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"test image")

        with patch("newspaper_explorer.data.indexing.image_index.find_mets_files", return_value=[]):
            with patch(
                "newspaper_explorer.data.indexing.image_index.ImageDownloader"
            ) as mock_downloader:
                mock_downloader.return_value.extract_image_references.return_value = []
                result = indexer.create_index()

        # Should find all 3 images
        assert len(result) == 3

    def test_saves_metadata_file(self, indexer: ImageIndexer) -> None:
        """Test that metadata file is saved with correct information."""
        # Create image
        img = indexer.images_dir / "1920" / "01" / "01" / "01" / "max_1.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"test image")

        with patch("newspaper_explorer.data.indexing.image_index.find_mets_files", return_value=[]):
            with patch(
                "newspaper_explorer.data.indexing.image_index.ImageDownloader"
            ) as mock_downloader:
                mock_downloader.return_value.extract_image_references.return_value = []
                indexer.create_index()

        # Check metadata file exists and has correct structure
        assert indexer.metadata_path.exists()

        import json

        with indexer.metadata_path.open() as f:
            metadata = json.load(f)

        assert metadata["source_name"] == "test_source"
        assert metadata["total_images_indexed"] == 1
        assert metadata["total_images_expected_from_mets"] == 0
        assert "index_created_at" in metadata


class TestLoadMetadata:
    """Tests for load_metadata method."""

    @pytest.fixture
    def indexer(self, tmp_path: Path) -> ImageIndexer:
        """Create an ImageIndexer with mocked dependencies."""
        mock_config = MagicMock()
        mock_config.data_dir = str(tmp_path / "data")

        source_dir = tmp_path / "data" / "raw" / "test_source"
        (source_dir / "images").mkdir(parents=True)
        (source_dir / "xml_ocr").mkdir(parents=True)

        with patch(
            "newspaper_explorer.data.indexing.image_index.get_config",
            return_value=mock_config,
        ):
            return ImageIndexer("test_source")

    def test_returns_none_when_metadata_file_missing(self, indexer: ImageIndexer) -> None:
        """Test that None is returned when metadata file doesn't exist."""
        result = indexer.load_metadata()
        assert result is None

    def test_loads_valid_metadata_file(self, indexer: ImageIndexer) -> None:
        """Test that valid metadata file is loaded correctly."""
        import json

        indexer.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source_name": "test_source",
            "total_images_indexed": 100,
            "total_images_expected_from_mets": 105,
            "index_created_at": "2025-12-17 10:00:00",
        }

        with indexer.metadata_path.open("w") as f:
            json.dump(metadata, f)

        result = indexer.load_metadata()

        assert result is not None
        assert result["source_name"] == "test_source"
        assert result["total_images_indexed"] == 100

    def test_handles_corrupted_metadata_file(self, indexer: ImageIndexer) -> None:
        """Test that corrupted metadata file is handled gracefully."""
        indexer.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.metadata_path.write_text("not valid json{{{")

        result = indexer.load_metadata()

        # Should return None on JSON decode error
        assert result is None


class TestGetStatsWithMetadata:
    """Tests for get_stats with metadata integration."""

    @pytest.fixture
    def indexer(self, tmp_path: Path) -> ImageIndexer:
        """Create an ImageIndexer with mocked dependencies."""
        mock_config = MagicMock()
        mock_config.data_dir = str(tmp_path / "data")

        source_dir = tmp_path / "data" / "raw" / "test_source"
        (source_dir / "images").mkdir(parents=True)
        (source_dir / "xml_ocr").mkdir(parents=True)

        with patch(
            "newspaper_explorer.data.indexing.image_index.get_config",
            return_value=mock_config,
        ):
            return ImageIndexer("test_source")

    def test_includes_expected_count_from_metadata(self, indexer: ImageIndexer) -> None:
        """Test that expected count is loaded from metadata file."""
        import json

        # Create index
        sample_df = pl.DataFrame(
            {
                "image_path": ["1920/01/01/01/max_1.jpg"],
                "source_id": ["test_source"],
                "year": [1920],
                "file_size_bytes": [1000],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(indexer.index_path)

        # Create metadata with expected count
        indexer.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source_name": "test_source",
            "total_images_indexed": 1,
            "total_images_expected_from_mets": 150,
        }
        with indexer.metadata_path.open("w") as f:
            json.dump(metadata, f)

        stats = indexer.get_stats()

        assert stats.total_images == 1
        assert stats.total_images_expected == 150

    def test_handles_missing_expected_count_in_metadata(self, indexer: ImageIndexer) -> None:
        """Test that missing expected count field is handled."""
        import json

        # Create index
        sample_df = pl.DataFrame(
            {
                "image_path": ["1920/01/01/01/max_1.jpg"],
                "source_id": ["test_source"],
                "year": [1920],
                "file_size_bytes": [1000],
            }
        )
        indexer.index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write_parquet(indexer.index_path)

        # Create metadata without expected count
        indexer.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"source_name": "test_source", "total_images_indexed": 1}
        with indexer.metadata_path.open("w") as f:
            json.dump(metadata, f)

        stats = indexer.get_stats()

        assert stats.total_images == 1
        assert stats.total_images_expected == 0  # Defaults to 0
