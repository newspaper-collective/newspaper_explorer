"""Tests for text/archive download module."""

from pathlib import Path
import tarfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from newspaper_explorer.data.download.text import ZenodoDownloader


class TestZenodoDownloader:
    """Tests for ZenodoDownloader class."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create mock config."""
        config = MagicMock()
        config.data_dir = str(tmp_path / "data")
        config.archives_dir = str(tmp_path / "archives")
        config.extracted_dir = str(tmp_path / "extracted")
        return config

    @pytest.fixture
    def mock_source_config(self) -> MagicMock:
        """Create mock source configuration."""
        config = MagicMock()
        config.dataset_name = "test_dataset"
        config.data_type = "xml_ocr"

        # Mock parts
        part1 = MagicMock()
        part1.name = "test_part_1900-1901"
        part1.url = "http://example.com/part1.tar.gz"
        part1.md5 = "abc123"
        part1.model_dump.return_value = {
            "name": "test_part_1900-1901",
            "url": "http://example.com/part1.tar.gz",
            "md5": "abc123",
        }

        part2 = MagicMock()
        part2.name = "test_part_1902-1903"
        part2.url = "http://example.com/part2.tar.gz"
        part2.md5 = None
        part2.model_dump.return_value = {
            "name": "test_part_1902-1903",
            "url": "http://example.com/part2.tar.gz",
            "md5": None,
        }

        config.parts = [part1, part2]
        config.get_year_range.return_value = (1900, 1920)

        return config

    @pytest.fixture
    def downloader(self, mock_config: MagicMock, mock_source_config: MagicMock) -> ZenodoDownloader:
        """Create ZenodoDownloader with mocked dependencies."""
        with (
            patch("newspaper_explorer.data.download.text.get_config", return_value=mock_config),
            patch(
                "newspaper_explorer.data.download.text.load_source_config",
                return_value=mock_source_config,
            ),
        ):
            return ZenodoDownloader("test_dataset")

    def test_init_creates_directories(self, downloader: ZenodoDownloader) -> None:
        """Test that initialization creates required directories."""
        assert downloader.download_dir.exists()
        assert downloader.extracted_dir.exists()

    def test_init_sets_configuration(self, downloader: ZenodoDownloader) -> None:
        """Test that configuration is set correctly."""
        assert downloader.dataset_name == "test_dataset"
        assert downloader.data_type == "xml_ocr"
        assert downloader.min_year == 1900
        assert downloader.max_year == 1920

    def test_list_available_parts(self, downloader: ZenodoDownloader) -> None:
        """Test listing available parts."""
        parts = downloader.list_available_parts()

        assert len(parts) == 2
        assert parts[0]["name"] == "test_part_1900-1901"
        assert parts[0]["url"] == "http://example.com/part1.tar.gz"
        assert parts[1]["name"] == "test_part_1902-1903"

    def test_download_part_success(self, downloader: ZenodoDownloader) -> None:
        """Test successful download of a part."""
        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content.return_value = [b"test data"]

        with (
            patch("newspaper_explorer.data.download.text.requests.get") as mock_get,
            patch("newspaper_explorer.data.download.text.verify_md5_checksum") as mock_verify,
        ):
            mock_get.return_value = mock_response
            mock_verify.return_value = True

            result = downloader.download_part("test_part_1900-1901")

        assert result.exists()
        assert result.name == "test_part_1900-1901.tar.gz"
        assert mock_get.called

    def test_download_part_skip_existing_with_valid_checksum(
        self, downloader: ZenodoDownloader
    ) -> None:
        """Test skipping download when file exists with valid checksum."""
        # Create the expected file
        download_path = (
            downloader.download_dir / "test_dataset" / "xml_ocr" / "test_part_1900-1901.tar.gz"
        )
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(b"existing data")

        with (
            patch("newspaper_explorer.data.download.text.requests.get") as mock_get,
            patch("newspaper_explorer.data.download.text.verify_md5_checksum") as mock_verify,
        ):
            mock_verify.return_value = True

            result = downloader.download_part("test_part_1900-1901")

        assert result == download_path
        assert not mock_get.called  # Should skip download

    def test_download_part_redownload_on_bad_checksum(self, downloader: ZenodoDownloader) -> None:
        """Test re-downloading when existing file has bad checksum."""
        # Create the expected file
        download_path = (
            downloader.download_dir / "test_dataset" / "xml_ocr" / "test_part_1900-1901.tar.gz"
        )
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(b"corrupted data")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content.return_value = [b"new data"]

        with (
            patch("newspaper_explorer.data.download.text.requests.get") as mock_get,
            patch("newspaper_explorer.data.download.text.verify_md5_checksum") as mock_verify,
        ):
            # First call: existing file fails checksum
            # Second call: newly downloaded file passes
            mock_verify.side_effect = [False, False]  # Both fail for this test
            mock_get.return_value = mock_response

            result = downloader.download_part("test_part_1900-1901")

        assert mock_get.called  # Should download

    def test_download_part_force_redownload(self, downloader: ZenodoDownloader) -> None:
        """Test forcing re-download even when file exists."""
        # Create the expected file
        download_path = (
            downloader.download_dir / "test_dataset" / "xml_ocr" / "test_part_1900-1901.tar.gz"
        )
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(b"existing data")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content.return_value = [b"new data"]

        with (
            patch("newspaper_explorer.data.download.text.requests.get") as mock_get,
            patch("newspaper_explorer.data.download.text.verify_md5_checksum") as mock_verify,
        ):
            mock_get.return_value = mock_response
            mock_verify.return_value = True

            result = downloader.download_part("test_part_1900-1901", force_redownload=True)

        assert mock_get.called  # Should download despite existing file

    def test_download_part_invalid_name(self, downloader: ZenodoDownloader) -> None:
        """Test downloading with invalid part name."""
        with pytest.raises(ValueError, match="Part 'invalid_part' not found"):
            downloader.download_part("invalid_part")

    def test_download_part_http_error(self, downloader: ZenodoDownloader) -> None:
        """Test handling HTTP errors during download."""
        with patch("newspaper_explorer.data.download.text.requests.get") as mock_get:
            mock_get.side_effect = requests.HTTPError("404 Not Found")

            with pytest.raises(requests.HTTPError):
                downloader.download_part("test_part_1900-1901")

    def test_download_part_without_checksum(self, downloader: ZenodoDownloader) -> None:
        """Test downloading part without MD5 checksum."""
        # Create the expected file
        download_path = (
            downloader.download_dir / "test_dataset" / "xml_ocr" / "test_part_1902-1903.tar.gz"
        )
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(b"existing data")

        with patch("newspaper_explorer.data.download.text.requests.get") as mock_get:
            result = downloader.download_part("test_part_1902-1903")

        assert result == download_path
        assert not mock_get.called  # Should skip (no checksum to verify)

    def test_get_archive_path(self, downloader: ZenodoDownloader) -> None:
        """Test getting archive path for a part."""
        path = downloader._get_archive_path("test_part_1900-1901")

        assert path.name == "test_part_1900-1901.tar.gz"
        assert "test_dataset" in str(path)
        assert "xml_ocr" in str(path)

    def test_extract_archive(self, downloader: ZenodoDownloader, tmp_path: Path) -> None:
        """Test extracting tar.gz archive."""
        # Create a test tar.gz file
        archive_path = tmp_path / "test.tar.gz"
        test_dir = tmp_path / "test_content"
        test_dir.mkdir()
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(test_file, arcname="test.txt")

        extract_path = tmp_path / "extracted"

        downloader._extract_archive(archive_path, extract_path)

        assert (extract_path / "test.txt").exists()
        assert (extract_path / "test.txt").read_text() == "test content"

    def test_find_year_source_path_with_dertagcopy(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test finding year source path with dertagcopy prefix."""
        temp_path = tmp_path / "temp"
        dertagcopy_path = temp_path / "dertagcopy"
        dertagcopy_path.mkdir(parents=True)

        result = downloader._find_year_source_path(temp_path, "test_part")

        assert result == dertagcopy_path

    def test_find_year_source_path_direct_years(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test finding year source path with direct year directories."""
        temp_path = tmp_path / "temp"
        temp_path.mkdir()

        # Create year directories directly
        (temp_path / "1900").mkdir()
        (temp_path / "1901").mkdir()

        # Also create dertagcopy (should be ignored)
        (temp_path / "dertagcopy").mkdir()

        result = downloader._find_year_source_path(temp_path, "test_part")

        assert result == temp_path  # Should prefer direct years

    def test_find_year_source_path_no_structure(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test finding year source path when no expected structure exists."""
        temp_path = tmp_path / "temp"
        temp_path.mkdir()

        result = downloader._find_year_source_path(temp_path, "test_part")

        assert result is None

    def test_extract_part_file_not_found(self, downloader: ZenodoDownloader) -> None:
        """Test extracting when archive doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            downloader.extract_part("test_part_1900-1901")

    def test_extract_part_with_fix_errors(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test extraction with error fixing enabled."""
        # Create a simple archive
        archive_path = downloader._get_archive_path("test_part_1900-1901")
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="test.txt")
            info.size = 4
            tar.addfile(info, fileobj=None)

        with (
            patch.object(downloader, "_organize_extracted_years") as mock_organize,
            patch("newspaper_explorer.data.download.text.DataFixer") as mock_fixer_class,
        ):
            mock_organize.return_value = tmp_path / "result"
            mock_fixer = MagicMock()
            mock_fixer_class.return_value = mock_fixer

            downloader.extract_part("test_part_1900-1901", fix_errors=True)

        assert mock_fixer.apply_fixes.called

    def test_extract_part_without_fix_errors(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test extraction without error fixing."""
        # Create a simple archive
        archive_path = downloader._get_archive_path("test_part_1900-1901")
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="test.txt")
            info.size = 4
            tar.addfile(info, fileobj=None)

        with (
            patch.object(downloader, "_organize_extracted_years") as mock_organize,
            patch("newspaper_explorer.data.download.text.DataFixer") as mock_fixer_class,
        ):
            mock_organize.return_value = tmp_path / "result"

            downloader.extract_part("test_part_1900-1901", fix_errors=False)

        assert not mock_fixer_class.called

    def test_download_part_timeout(self, downloader: ZenodoDownloader) -> None:
        """Test handling timeout during download."""
        with patch("newspaper_explorer.data.download.text.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("Connection timeout")

            with pytest.raises(requests.Timeout):
                downloader.download_part("test_part_1900-1901")

    def test_init_with_custom_data_dir(
        self, mock_config: MagicMock, mock_source_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test initialization with custom data directory."""
        custom_dir = tmp_path / "custom"

        with (
            patch("newspaper_explorer.data.download.text.get_config", return_value=mock_config),
            patch(
                "newspaper_explorer.data.download.text.load_source_config",
                return_value=mock_source_config,
            ),
        ):
            downloader = ZenodoDownloader("test_dataset", data_dir=custom_dir)

        assert downloader.data_dir == custom_dir

    def test_move_year_directories(self, downloader: ZenodoDownloader, tmp_path: Path) -> None:
        """Test moving year directories."""
        source = tmp_path / "source"
        source.mkdir()

        # Create valid year directories
        (source / "1900").mkdir()
        (source / "1901").mkdir()
        (source / "invalid").mkdir()  # Should be skipped
        (source / "2000").mkdir()  # Outside range, should be skipped

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        years = downloader._move_year_directories(source, raw_dir)

        assert "1900" in years
        assert "1901" in years
        assert "invalid" not in years
        assert "2000" not in years
        assert (raw_dir / "1900").exists()
        assert (raw_dir / "1901").exists()

    def test_move_year_directories_merge(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test merging when destination year directory exists."""
        source = tmp_path / "source"
        source.mkdir()

        # Create year directory with month
        year_dir = source / "1900"
        year_dir.mkdir()
        (year_dir / "01").mkdir()
        (year_dir / "01" / "file.txt").write_text("test")

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "1900").mkdir()  # Pre-existing

        with patch.object(downloader, "_merge_directory") as mock_merge:
            years = downloader._move_year_directories(source, raw_dir)

        assert "1900" in years
        assert mock_merge.called

    def test_merge_directory(self, downloader: ZenodoDownloader, tmp_path: Path) -> None:
        """Test merging directory contents."""
        source = tmp_path / "source"
        source.mkdir()

        # Create month directory with day
        month_dir = source / "01"
        month_dir.mkdir()
        day_dir = month_dir / "15"
        day_dir.mkdir()
        (day_dir / "file.txt").write_text("test")

        dest = tmp_path / "dest"
        dest.mkdir()

        downloader._merge_directory(source, dest)

        assert (dest / "01" / "15" / "file.txt").exists()

    def test_merge_directory_existing_items(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test merging when items already exist in destination."""
        source = tmp_path / "source"
        source.mkdir()

        month_dir = source / "01"
        month_dir.mkdir()
        (month_dir / "new_day").mkdir()

        dest = tmp_path / "dest"
        dest.mkdir()
        dest_month = dest / "01"
        dest_month.mkdir()
        (dest_month / "existing_day").mkdir()

        downloader._merge_directory(source, dest)

        # Both should exist
        assert (dest / "01" / "new_day").exists()
        assert (dest / "01" / "existing_day").exists()

    def test_cleanup_empty_parent_dirs(self, downloader: ZenodoDownloader, tmp_path: Path) -> None:
        """Test cleanup of empty directories."""
        # Create nested empty directories
        empty_path = tmp_path / "extracted" / "dataset" / "xml_ocr"
        empty_path.mkdir(parents=True)

        # Set extracted_dir to the base
        downloader.extracted_dir = tmp_path / "extracted"

        downloader._cleanup_empty_parent_dirs(empty_path)

        # All empty directories should be removed
        assert not (tmp_path / "extracted").exists()

    def test_cleanup_empty_parent_dirs_with_content(
        self, downloader: ZenodoDownloader, tmp_path: Path
    ) -> None:
        """Test cleanup stops when directory has content."""
        base = tmp_path / "extracted" / "dataset"
        empty = base / "xml_ocr"
        empty.mkdir(parents=True)

        # Add a file to base
        (base / "keep.txt").write_text("content")

        downloader.extracted_dir = tmp_path / "extracted"

        downloader._cleanup_empty_parent_dirs(empty)

        # Empty child removed, but parent with content kept
        assert not empty.exists()
        assert base.exists()
        assert (base / "keep.txt").exists()

    def test_get_extraction_status(self, downloader: ZenodoDownloader, tmp_path: Path) -> None:
        """Test getting extraction status."""
        # Create download directory
        download_dir = downloader.download_dir / "test_dataset" / "xml_ocr"
        download_dir.mkdir(parents=True)

        # Create downloaded archive
        (download_dir / "test_part_1900-1901.tar.gz").write_bytes(b"archive")

        # Mock the parts to have year ranges
        downloader.config.parts[0].years = "1900-1901"
        downloader.config.parts[1].years = "1902-1903"

        # Create some extracted years matching first part
        with patch("newspaper_explorer.data.download.text.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.data_dir = tmp_path / "data"
            mock_get_config.return_value = mock_config

            raw_dir = tmp_path / "data" / "raw" / "test_dataset" / "xml_ocr"
            raw_dir.mkdir(parents=True)
            (raw_dir / "1900").mkdir()
            (raw_dir / "1901").mkdir()

            status = downloader.get_extraction_status()

        # Should have status for both parts
        assert "test_part_1900-1901" in status
        assert "test_part_1902-1903" in status

        # First part should show as downloaded and extracted
        part1_status = status["test_part_1900-1901"]
        assert part1_status["downloaded"] is True
        assert part1_status["extracted"] is True
        assert "1900" in part1_status["extracted_years"]
        assert "1901" in part1_status["extracted_years"]

        # Second part not downloaded
        part2_status = status["test_part_1902-1903"]
        assert part2_status["downloaded"] is False
