"""Tests for image download module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from lxml import etree
import pytest
import requests

from newspaper_explorer.data.download.images import ImageDownloader
from newspaper_explorer.models.data.images import ImageReference


class TestImageDownloader:
    """Tests for ImageDownloader class."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create mock config."""
        config = MagicMock()
        config.data_dir = str(tmp_path / "data")
        return config

    @pytest.fixture
    def mock_source_config(self) -> MagicMock:
        """Create mock source configuration."""
        config = MagicMock()
        config.dataset_name = "test_source"
        config.data_type = "xml_ocr"
        return config

    @pytest.fixture
    def downloader(
        self, mock_config: MagicMock, mock_source_config: MagicMock, tmp_path: Path
    ) -> ImageDownloader:
        """Create ImageDownloader with mocked dependencies."""
        # Create directory structure
        data_dir = tmp_path / "data"
        xml_dir = data_dir / "raw" / "test_source" / "xml_ocr"
        xml_dir.mkdir(parents=True)

        with (
            patch("newspaper_explorer.data.download.images.get_config", return_value=mock_config),
            patch(
                "newspaper_explorer.data.download.images.load_source_config",
                return_value=mock_source_config,
            ),
        ):
            return ImageDownloader("test_source", max_workers=2, max_retries=2, timeout=5)

    def test_init_sets_paths_correctly(self, downloader: ImageDownloader, tmp_path: Path) -> None:
        """Test that initialization sets up correct paths."""
        assert downloader.source_name == "test_source"
        assert downloader.dataset_name == "test_source"
        assert downloader.data_type == "xml_ocr"
        assert "xml_ocr" in str(downloader.xml_dir)
        assert "images" in str(downloader.images_dir)

    def test_init_sets_parameters(self, downloader: ImageDownloader) -> None:
        """Test that initialization sets download parameters."""
        assert downloader.max_workers == 2
        assert downloader.max_retries == 2
        assert downloader.timeout == 5
        assert downloader.validate is True
        assert downloader.min_image_size == 1024

    def test_init_with_custom_validation_settings(
        self, mock_config: MagicMock, mock_source_config: MagicMock
    ) -> None:
        """Test initialization with custom validation settings."""
        with patch("newspaper_explorer.data.download.images.get_config", return_value=mock_config):
            with patch(
                "newspaper_explorer.data.download.images.load_source_config",
                return_value=mock_source_config,
            ):
                downloader = ImageDownloader("test_source", validate=False, min_image_size=2048)

        assert downloader.validate is False
        assert downloader.min_image_size == 2048

    def test_extract_image_references_from_mets(self, tmp_path: Path) -> None:
        """Test extracting image references from METS XML."""
        # Create a minimal METS file with MAX fileGrp
        mets_content = """<?xml version="1.0" encoding="UTF-8"?>
        <mets:mets xmlns:mets="http://www.loc.gov/METS/"
                   xmlns:xlink="http://www.w3.org/1999/xlink">
            <mets:fileSec>
                <mets:fileGrp USE="MAX">
                    <mets:file ID="max_001">
                        <mets:FLocat xlink:href="http://example.com/image001.jpg"/>
                    </mets:file>
                    <mets:file ID="max_002">
                        <mets:FLocat xlink:href="http://example.com/image002.png"/>
                    </mets:file>
                </mets:fileGrp>
            </mets:fileSec>
        </mets:mets>
        """

        mets_file = tmp_path / "test.xml"
        mets_file.write_text(mets_content)

        with patch("newspaper_explorer.data.download.images.get_config"):
            with patch("newspaper_explorer.data.download.images.load_source_config"):
                downloader = ImageDownloader("test_source")
                references = downloader.extract_image_references(mets_file)

        assert len(references) == 2
        assert references[0].file_id == "max_001"
        assert references[0].url == "http://example.com/image001.jpg"
        assert references[0].extension == ".jpg"
        assert references[1].file_id == "max_002"
        assert references[1].extension == ".png"

    def test_extract_image_references_no_max_group(self, tmp_path: Path) -> None:
        """Test handling METS without MAX fileGrp."""
        mets_content = """<?xml version="1.0" encoding="UTF-8"?>
        <mets:mets xmlns:mets="http://www.loc.gov/METS/">
            <mets:fileSec>
                <mets:fileGrp USE="MIN">
                    <mets:file ID="min_001">
                        <mets:FLocat/>
                    </mets:file>
                </mets:fileGrp>
            </mets:fileSec>
        </mets:mets>
        """

        mets_file = tmp_path / "test.xml"
        mets_file.write_text(mets_content)

        with patch("newspaper_explorer.data.download.images.get_config"):
            with patch("newspaper_explorer.data.download.images.load_source_config"):
                downloader = ImageDownloader("test_source")
                references = downloader.extract_image_references(mets_file)

        assert len(references) == 0

    def test_extract_image_references_malformed_xml(self, tmp_path: Path) -> None:
        """Test handling malformed XML."""
        mets_file = tmp_path / "bad.xml"
        mets_file.write_text("not valid xml {{{")

        with patch("newspaper_explorer.data.download.images.get_config"):
            with patch("newspaper_explorer.data.download.images.load_source_config"):
                downloader = ImageDownloader("test_source")
                references = downloader.extract_image_references(mets_file)

        assert len(references) == 0

    def test_extract_image_references_missing_file(self, tmp_path: Path) -> None:
        """Test handling missing METS file."""
        mets_file = tmp_path / "nonexistent.xml"

        with patch("newspaper_explorer.data.download.images.get_config"):
            with patch("newspaper_explorer.data.download.images.load_source_config"):
                downloader = ImageDownloader("test_source")
                references = downloader.extract_image_references(mets_file)

        assert len(references) == 0

    def test_get_image_path_maintains_directory_structure(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test that image path mirrors XML directory structure."""
        # Set up XML directory structure
        xml_subdir = downloader.xml_dir / "1920" / "01" / "15"
        xml_subdir.mkdir(parents=True, exist_ok=True)
        mets_file = xml_subdir / "test.xml"
        mets_file.touch()

        image_ref = ImageReference(file_id="max_001", url="http://example.com", extension=".jpg")

        result_path = downloader._get_image_path(mets_file, image_ref)

        # Should mirror the structure under images/
        assert "images" in str(result_path)
        assert "1920" in str(result_path)
        assert "01" in str(result_path)
        assert "15" in str(result_path)
        assert result_path.name == "max_001.jpg"

    def test_get_image_path_creates_directories(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test that target directories are created."""
        xml_subdir = downloader.xml_dir / "1920" / "01" / "15"
        xml_subdir.mkdir(parents=True)
        mets_file = xml_subdir / "test.xml"
        mets_file.touch()

        image_ref = ImageReference(file_id="max_001", url="http://example.com", extension=".jpg")

        result_path = downloader._get_image_path(mets_file, image_ref)

        # Directory should exist
        assert result_path.parent.exists()

    def test_download_single_image_success(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test successful image download."""
        save_path = tmp_path / "test.jpg"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake image data"

        with patch("newspaper_explorer.data.download.images.requests.get") as mock_get:
            with patch(
                "newspaper_explorer.data.download.images.validate_image_file"
            ) as mock_validate:
                mock_get.return_value = mock_response
                mock_validate.return_value = Mock(is_valid=True)

                result = downloader._download_single_image(
                    "http://example.com/image.jpg", save_path, "img_001"
                )

        assert result["success"] is True
        assert result["skipped"] is False
        assert save_path.exists()
        assert save_path.read_bytes() == b"fake image data"

    def test_download_single_image_skip_existing_valid(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test skipping download for existing valid image."""
        save_path = tmp_path / "test.jpg"
        save_path.write_bytes(b"existing image")

        with patch("newspaper_explorer.data.download.images.validate_image_file") as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)

            result = downloader._download_single_image(
                "http://example.com/image.jpg", save_path, "img_001"
            )

        assert result["success"] is True
        assert result["skipped"] is True
        assert result["validated"] is True

    def test_download_single_image_redownload_invalid(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test re-downloading when existing file is invalid."""
        save_path = tmp_path / "test.jpg"
        save_path.write_bytes(b"corrupted")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"new image data"

        with patch("newspaper_explorer.data.download.images.requests.get") as mock_get:
            with patch(
                "newspaper_explorer.data.download.images.validate_image_file"
            ) as mock_validate:
                # First call: existing file is invalid
                # Second call: newly downloaded file is valid
                mock_validate.side_effect = [
                    Mock(is_valid=False, error="Too small"),
                    Mock(is_valid=True),
                ]
                mock_get.return_value = mock_response

                result = downloader._download_single_image(
                    "http://example.com/image.jpg", save_path, "img_001"
                )

        assert result["success"] is True
        assert save_path.read_bytes() == b"new image data"

    def test_download_single_image_retry_on_failure(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test retry logic on download failure."""
        save_path = tmp_path / "test.jpg"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"image data"

        with patch("newspaper_explorer.data.download.images.requests.get") as mock_get:
            with patch(
                "newspaper_explorer.data.download.images.validate_image_file"
            ) as mock_validate:
                # First attempt fails, second succeeds
                mock_get.side_effect = [
                    requests.RequestException("Network error"),
                    mock_response,
                ]
                mock_validate.return_value = Mock(is_valid=True)

                result = downloader._download_single_image(
                    "http://example.com/image.jpg", save_path, "img_001"
                )

        assert result["success"] is True
        assert mock_get.call_count == 2

    def test_download_single_image_fails_after_retries(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test failure after exhausting retries."""
        save_path = tmp_path / "test.jpg"

        with patch("newspaper_explorer.data.download.images.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")

            result = downloader._download_single_image(
                "http://example.com/image.jpg", save_path, "img_001"
            )

        assert result["success"] is False
        assert "error" in result
        assert not save_path.exists()

    def test_download_single_image_without_validation(
        self, mock_config: MagicMock, mock_source_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test download without validation skips existing files."""
        with patch("newspaper_explorer.data.download.images.get_config", return_value=mock_config):
            with patch(
                "newspaper_explorer.data.download.images.load_source_config",
                return_value=mock_source_config,
            ):
                downloader = ImageDownloader("test_source", validate=False)

        save_path = tmp_path / "test.jpg"
        save_path.write_bytes(b"existing")

        result = downloader._download_single_image(
            "http://example.com/image.jpg", save_path, "img_001"
        )

        assert result["success"] is True
        assert result["skipped"] is True
        assert result["validated"] is False

    def test_download_single_image_http_error(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test handling HTTP errors."""
        save_path = tmp_path / "test.jpg"

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("newspaper_explorer.data.download.images.requests.get") as mock_get:
            mock_get.return_value = mock_response

            result = downloader._download_single_image(
                "http://example.com/image.jpg", save_path, "img_001"
            )

        assert result["success"] is False
        assert not save_path.exists()

    def test_download_single_image_timeout(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test handling timeout errors."""
        save_path = tmp_path / "test.jpg"

        with patch("newspaper_explorer.data.download.images.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("Request timed out")

            result = downloader._download_single_image(
                "http://example.com/image.jpg", save_path, "img_001"
            )

        assert result["success"] is False
        assert "error" in result

    def test_extract_image_references_default_extension(self, tmp_path: Path) -> None:
        """Test default extension when URL has no extension."""
        mets_content = """<?xml version="1.0" encoding="UTF-8"?>
        <mets:mets xmlns:mets="http://www.loc.gov/METS/"
                   xmlns:xlink="http://www.w3.org/1999/xlink">
            <mets:fileSec>
                <mets:fileGrp USE="MAX">
                    <mets:file ID="max_001">
                        <mets:FLocat xlink:href="http://example.com/image"/>
                    </mets:file>
                </mets:fileGrp>
            </mets:fileSec>
        </mets:mets>
        """

        mets_file = tmp_path / "test.xml"
        mets_file.write_text(mets_content)

        with patch("newspaper_explorer.data.download.images.get_config"):
            with patch("newspaper_explorer.data.download.images.load_source_config"):
                downloader = ImageDownloader("test_source")
                references = downloader.extract_image_references(mets_file)

        assert len(references) == 1
        assert references[0].extension == ".jpg"  # Default extension

    def test_get_image_path_fallback_for_path_outside_xml_dir(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test fallback when METS file is outside xml_dir."""
        # METS file outside the expected xml_dir
        mets_file = tmp_path / "external" / "mets.xml"
        mets_file.parent.mkdir(parents=True)

        image_ref = ImageReference(
            file_id="img001", url="http://example.com/image.jpg", extension=".jpg"
        )

        result_path = downloader._get_image_path(mets_file, image_ref)

        # Should use images_dir directly as fallback
        assert result_path == downloader.images_dir / "img001.jpg"

    def test_download_images_no_mets_files(self, downloader: ImageDownloader) -> None:
        """Test download_images when no METS files exist."""
        with patch("newspaper_explorer.data.download.images.find_mets_files", return_value=[]):
            stats = downloader.download_images()

        assert stats["total"] == 0
        assert stats["downloaded"] == 0
        assert stats["skipped"] == 0
        assert stats["failed"] == 0

    def test_download_images_with_mets_files(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test download_images processes METS files."""
        # Create minimal METS file
        mets_content = """<?xml version="1.0" encoding="UTF-8"?>
        <mets:mets xmlns:mets="http://www.loc.gov/METS/"
                   xmlns:xlink="http://www.w3.org/1999/xlink">
            <mets:fileSec>
                <mets:fileGrp USE="MAX">
                    <mets:file ID="max_001">
                        <mets:FLocat xlink:href="http://example.com/img1.jpg"/>
                    </mets:file>
                </mets:fileGrp>
            </mets:fileSec>
        </mets:mets>"""

        mets_file = tmp_path / "test.xml"
        mets_file.write_text(mets_content)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake image"

        with (
            patch(
                "newspaper_explorer.data.download.images.find_mets_files",
                return_value=[mets_file],
            ),
            patch("newspaper_explorer.data.download.images.requests.get") as mock_get,
            patch("newspaper_explorer.data.download.images.validate_image_file") as mock_validate,
        ):
            mock_get.return_value = mock_response
            mock_validate.return_value = MagicMock(is_valid=True)

            stats = downloader.download_images()

        assert stats["total"] == 1
        assert stats["downloaded"] == 1

    def test_get_download_status_no_mets_files(self, downloader: ImageDownloader) -> None:
        """Test get_download_status when no METS files exist."""
        with patch("newspaper_explorer.data.download.images.find_mets_files", return_value=[]):
            status = downloader.get_download_status()

        assert status["mets_files"] == 0
        assert status["total_images_expected"] == 0
        assert status["images_downloaded"] == 0
        assert status["coverage_pct"] == 0.0
        assert status["images_dir_exists"] is False

    def test_get_download_status_with_downloaded_images(
        self, downloader: ImageDownloader, tmp_path: Path
    ) -> None:
        """Test get_download_status with downloaded images."""
        # Create METS file with one image reference
        mets_content = """<?xml version="1.0" encoding="UTF-8"?>
        <mets:mets xmlns:mets="http://www.loc.gov/METS/"
                   xmlns:xlink="http://www.w3.org/1999/xlink">
            <mets:fileSec>
                <mets:fileGrp USE="MAX">
                    <mets:file ID="max_001">
                        <mets:FLocat xlink:href="http://example.com/img1.jpg"/>
                    </mets:file>
                </mets:fileGrp>
            </mets:fileSec>
        </mets:mets>"""

        mets_file = tmp_path / "test.xml"
        mets_file.write_text(mets_content)

        # Create one downloaded image
        downloader.images_dir.mkdir(parents=True, exist_ok=True)
        (downloader.images_dir / "img1.jpg").write_bytes(b"fake")

        with patch(
            "newspaper_explorer.data.download.images.find_mets_files",
            return_value=[mets_file],
        ):
            status = downloader.get_download_status()

        assert status["mets_files"] == 1
        assert status["total_images_expected"] == 1
        assert status["images_downloaded"] == 1
        assert status["coverage_pct"] == 100.0
