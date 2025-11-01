"""
Tests for image downloading functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from newspaper_explorer.data.images import ImageDownloader, ImageReference


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("newspaper_explorer.data.images.get_config") as mock:
        config = Mock()
        config.data_dir = Path("/fake/data")
        mock.return_value = config
        yield config


@pytest.fixture
def mock_source_config():
    """Mock source configuration."""
    return {
        "dataset_name": "test_source",
        "data_type": "xml_ocr",
        "metadata": {"newspaper_title": "Test Newspaper"},
    }


def test_image_downloader_initialization(mock_config, mock_source_config):
    """Test ImageDownloader initialization."""
    with patch("newspaper_explorer.data.images.load_source_config") as mock_load:
        mock_load.return_value = mock_source_config

        downloader = ImageDownloader(source_name="test_source", max_workers=4, max_retries=2)

        assert downloader.source_name == "test_source"
        assert downloader.max_workers == 4
        assert downloader.max_retries == 2
        assert downloader.dataset_name == "test_source"
        assert downloader.xml_dir == Path("/fake/data/raw/test_source/xml_ocr")
        assert downloader.images_dir == Path("/fake/data/raw/test_source/images")


def test_image_reference():
    """Test ImageReference dataclass."""
    ref = ImageReference(
        file_id="FILE_0001_MASTER", url="https://example.com/image.jpg", extension=".jpg"
    )

    assert ref.file_id == "FILE_0001_MASTER"
    assert ref.url == "https://example.com/image.jpg"
    assert ref.extension == ".jpg"


def test_find_mets_files(mock_config, mock_source_config, tmp_path):
    """Test finding METS files."""
    with patch("newspaper_explorer.data.images.load_source_config") as mock_load:
        mock_load.return_value = mock_source_config

        # Create test directory structure
        xml_dir = tmp_path / "xml_ocr"
        xml_dir.mkdir()

        # Create mock METS files
        (xml_dir / "issue1.xml").touch()
        (xml_dir / "issue2.xml").touch()

        # Create fulltext directory (should be excluded)
        fulltext_dir = xml_dir / "fulltext"
        fulltext_dir.mkdir()
        (fulltext_dir / "page1.xml").touch()

        # Patch config to use tmp_path
        mock_config.return_value.data_dir = tmp_path

        downloader = ImageDownloader(source_name="test_source")
        downloader.xml_dir = xml_dir

        mets_files = downloader.find_mets_files()

        # Should find 2 METS files (excluding fulltext)
        assert len(mets_files) == 2
        assert all("fulltext" not in str(f) for f in mets_files)


def test_get_image_path(mock_config, mock_source_config, tmp_path):
    """Test image path calculation."""
    with patch("newspaper_explorer.data.images.load_source_config") as mock_load:
        mock_load.return_value = mock_source_config

        # Setup paths
        xml_dir = tmp_path / "xml_ocr"
        images_dir = tmp_path / "images"
        mets_file = xml_dir / "1901" / "01" / "08" / "issue.xml"

        mock_config.return_value.data_dir = tmp_path

        downloader = ImageDownloader(source_name="test_source")
        downloader.xml_dir = xml_dir
        downloader.images_dir = images_dir

        img_ref = ImageReference(
            file_id="FILE_0001_MASTER",
            url="https://example.com/image.jpg",
            extension=".jpg",
        )

        image_path = downloader._get_image_path(mets_file, img_ref)

        # Should mirror directory structure
        expected = images_dir / "1901" / "01" / "08" / "FILE_0001_MASTER.jpg"
        assert image_path == expected


def test_extract_image_references(mock_config, mock_source_config, tmp_path):
    """Test extracting image references from METS XML."""
    with patch("newspaper_explorer.data.images.load_source_config") as mock_load:
        mock_load.return_value = mock_source_config
        mock_config.return_value.data_dir = tmp_path

        # Create mock METS XML
        mets_xml = """<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/"
           xmlns:xlink="http://www.w3.org/1999/xlink">
    <mets:fileSec>
        <mets:fileGrp USE="MAX">
            <mets:file ID="FILE_0001_MASTER" MIMETYPE="image/jpeg">
                <mets:FLocat xlink:href="https://example.com/image1.jpg"/>
            </mets:file>
            <mets:file ID="FILE_0002_MASTER" MIMETYPE="image/jpeg">
                <mets:FLocat xlink:href="https://example.com/image2.jpg"/>
            </mets:file>
        </mets:fileGrp>
    </mets:fileSec>
</mets:mets>"""

        mets_file = tmp_path / "test.xml"
        mets_file.write_text(mets_xml, encoding="utf-8")

        downloader = ImageDownloader(source_name="test_source")
        references = downloader.extract_image_references(mets_file)

        assert len(references) == 2
        assert references[0].file_id == "FILE_0001_MASTER"
        assert references[0].url == "https://example.com/image1.jpg"
        assert references[0].extension == ".jpg"
        assert references[1].file_id == "FILE_0002_MASTER"


def test_extract_image_references_no_max_group(mock_config, mock_source_config, tmp_path):
    """Test handling METS without MAX fileGrp."""
    with patch("newspaper_explorer.data.images.load_source_config") as mock_load:
        mock_load.return_value = mock_source_config
        mock_config.return_value.data_dir = tmp_path

        # Create METS without MAX group
        mets_xml = """<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/">
    <mets:fileSec>
        <mets:fileGrp USE="DEFAULT">
        </mets:fileGrp>
    </mets:fileSec>
</mets:mets>"""

        mets_file = tmp_path / "test.xml"
        mets_file.write_text(mets_xml, encoding="utf-8")

        downloader = ImageDownloader(source_name="test_source")
        references = downloader.extract_image_references(mets_file)

        assert len(references) == 0


@pytest.mark.integration
def test_download_single_image(mock_config, mock_source_config, tmp_path):
    """Test downloading a single image."""
    with patch("newspaper_explorer.data.images.load_source_config") as mock_load:
        mock_load.return_value = mock_source_config
        mock_config.return_value.data_dir = tmp_path

        downloader = ImageDownloader(source_name="test_source")

        # Mock requests.get
        with patch("newspaper_explorer.data.images.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.content = b"fake image data"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            save_path = tmp_path / "test_image.jpg"
            result = downloader._download_single_image(
                url="https://example.com/image.jpg",
                save_path=save_path,
                img_id="FILE_0001",
            )

            assert result["success"] is True
            assert result["skipped"] is False
            assert save_path.exists()
            assert save_path.read_bytes() == b"fake image data"


def test_download_single_image_already_exists(mock_config, mock_source_config, tmp_path):
    """Test skipping already downloaded images."""
    with patch("newspaper_explorer.data.images.load_source_config") as mock_load:
        mock_load.return_value = mock_source_config
        mock_config.return_value.data_dir = tmp_path

        downloader = ImageDownloader(source_name="test_source")

        # Create existing file
        save_path = tmp_path / "test_image.jpg"
        save_path.write_bytes(b"existing data")

        result = downloader._download_single_image(
            url="https://example.com/image.jpg",
            save_path=save_path,
            img_id="FILE_0001",
        )

        assert result["success"] is True
        assert result["skipped"] is True
        # Content unchanged
        assert save_path.read_bytes() == b"existing data"
