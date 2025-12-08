"""Tests for image_worker module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from newspaper_explorer.data.indexing.image_metadata_worker import extract_image_metadata_worker


def create_image_mock(width: int = 100, height: int = 200) -> MagicMock:
    """Create a properly configured Image mock."""
    mock_img = MagicMock()
    mock_img.size = (width, height)
    mock_img.__enter__ = MagicMock(return_value=mock_img)
    mock_img.__exit__ = MagicMock(return_value=False)
    return mock_img


@pytest.fixture
def mock_pil_image() -> MagicMock:
    """Create a mock for PIL.Image module."""
    mock_image_module = MagicMock()
    mock_image_module.open.return_value = create_image_mock()
    mock_image_module.DecompressionBombError = Exception
    return mock_image_module


class TestExtractImageMetadataWorker:
    """Tests for extract_image_metadata_worker function."""

    @pytest.fixture
    def images_dir(self, tmp_path: Path) -> Path:
        """Create a temporary images directory."""
        images = tmp_path / "images"
        images.mkdir()
        return images

    @pytest.fixture
    def sample_image_path(self, images_dir: Path) -> Path:
        """Create a sample image file with proper path structure."""
        # Path structure: YYYY/MM/DD/issue_number/filename.jpg
        img_path = images_dir / "1920" / "01" / "15" / "01" / "max_7.jpg"
        img_path.parent.mkdir(parents=True)
        img_path.write_bytes(b"fake image content")
        return img_path

    def test_extracts_basic_metadata_from_path(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that basic metadata is extracted from the path structure."""
        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache={},
            )

        assert result is not None
        assert result["year"] == 1920
        assert result["month"] == 1
        assert result["day"] == 15
        assert result["date"] == "1920-01-15"
        assert result["filename"] == "max_7.jpg"
        assert result["source_id"] == "test_source"
        assert result["file_exists"] is True

    def test_extracts_page_number_from_max_filename(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that page number is extracted from 'max_N.jpg' filename pattern."""
        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache={},
            )

        assert result is not None
        assert result["page_number"] == 7

    def test_uses_mets_cache_for_metadata(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that METS metadata is used when available in cache."""
        mets_cache = {
            "1920/01/15/01": {
                "issue_id": "test_source_1920-01-15_001_1",
                "newspaper_title": "Test Newspaper",
                "year_volume": "1920/15",
                "page_count": 8,
                "date": "1920-01-15",
                "issue_number": 1,
                "edition": 1,
            }
        }

        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache=mets_cache,
                alto_cache={},
            )

        assert result is not None
        assert result["issue_id"] == "test_source_1920-01-15_001_1"
        assert result["newspaper_title"] == "Test Newspaper"
        assert result["year_volume"] == "1920/15"
        assert result["page_count"] == 8
        assert result["issue_number"] == 1
        assert result["edition"] == 1

    def test_generates_page_id_with_complete_mets_data(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that page_id is generated when all required METS data is present."""
        mets_cache = {
            "1920/01/15/01": {
                "issue_id": "test_source_1920-01-15_001_1",
                "date": "1920-01-15",
                "issue_number": 1,
                "edition": 1,
            }
        }

        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache=mets_cache,
                alto_cache={},
            )

        assert result is not None
        assert result["page_id"] == "test_source_1920-01-15_001_1_007"

    def test_uses_alto_cache_for_dimensions(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that ALTO dimensions are retrieved from cache."""
        alto_cache = {
            "1920/01/15/01/007": (2000, 3000),  # 3-digit padding
        }

        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache=alto_cache,
            )

        assert result is not None
        assert result["alto_width"] == 2000
        assert result["alto_height"] == 3000

    def test_tries_different_page_paddings_for_alto_lookup(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that ALTO lookup tries different zero-padding lengths."""
        # Use 4-digit padding (should still find it)
        alto_cache = {
            "1920/01/15/01/0007": (1500, 2500),  # 4-digit padding
        }

        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache=alto_cache,
            )

        assert result is not None
        assert result["alto_width"] == 1500
        assert result["alto_height"] == 2500

    def test_returns_none_for_invalid_path_structure(self, images_dir: Path) -> None:
        """Test that None is returned for paths that don't match expected structure."""
        # Create a file with wrong path structure (too few components)
        bad_path = images_dir / "1920" / "01" / "image.jpg"
        bad_path.parent.mkdir(parents=True)
        bad_path.write_bytes(b"fake image")

        result = extract_image_metadata_worker(
            img_path=bad_path,
            images_dir=images_dir,
            source_id="test_source",
            mets_cache={},
            alto_cache={},
        )

        assert result is None

    def test_handles_non_max_filename(self, images_dir: Path, mock_pil_image: MagicMock) -> None:
        """Test that page_number is None for filenames without 'max_' pattern."""
        img_path = images_dir / "1920" / "01" / "15" / "01" / "page_007.jpg"
        img_path.parent.mkdir(parents=True)
        img_path.write_bytes(b"fake image")

        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=img_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache={},
            )

        assert result is not None
        assert result["page_number"] is None
        assert result["filename"] == "page_007.jpg"

    def test_fallback_issue_id_when_not_in_cache(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that path_key is used as issue_id when not in METS cache."""
        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache={},
            )

        assert result is not None
        assert result["issue_id"] == "1920/01/15/01"  # Falls back to path_key

    def test_includes_file_size(
        self, sample_image_path: Path, images_dir: Path, mock_pil_image: MagicMock
    ) -> None:
        """Test that file size is included in the result."""
        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_pil_image):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache={},
            )

        assert result is not None
        assert result["file_size_bytes"] is not None
        assert result["file_size_bytes"] > 0

    def test_handles_image_read_failure(self, sample_image_path: Path, images_dir: Path) -> None:
        """Test that image dimension read failures are handled gracefully."""
        mock_image_module = MagicMock()
        mock_image_module.open.side_effect = OSError("Cannot read image")
        mock_image_module.DecompressionBombError = Exception

        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_image_module):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache={},
            )

        assert result is not None
        assert result["width"] is None
        assert result["height"] is None

    def test_reads_actual_image_dimensions(self, sample_image_path: Path, images_dir: Path) -> None:
        """Test that actual image dimensions are read from the file."""
        mock_image_module = MagicMock()
        mock_image_module.open.return_value = create_image_mock(800, 1200)
        mock_image_module.DecompressionBombError = Exception

        with patch("newspaper_explorer.data.indexing.image_metadata_worker.Image", mock_image_module):
            result = extract_image_metadata_worker(
                img_path=sample_image_path,
                images_dir=images_dir,
                source_id="test_source",
                mets_cache={},
                alto_cache={},
            )

        assert result is not None
        assert result["width"] == 800
        assert result["height"] == 1200
