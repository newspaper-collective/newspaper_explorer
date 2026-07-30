"""
Pytest fixtures for data tests.

Provides reusable mock configurations, XML samples, and DataFrames for testing
download, parsing, preprocessing, and data processing functions.
"""

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

# =============================================================================
# Mock Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_config(tmp_path: Path) -> Mock:
    """
    Mock application configuration with tmp_path for isolation.

    Returns a Mock config object with data directories set to tmp_path.
    """
    config = Mock()
    config.data_dir = tmp_path
    config.archives_dir = tmp_path / "archives"
    config.extracted_dir = tmp_path / "extracted"
    config.parsed_dir = tmp_path / "parsed"
    config.preprocessed_dir = tmp_path / "preprocessed"
    config.sources_dir = tmp_path / "sources"
    config.results_dir = tmp_path / "results"
    return config


# =============================================================================
# Mock Source Configuration Fixtures
# =============================================================================


def create_mock_part(name: str, url: str, years: str, md5: str, size: str) -> Mock:
    """
    Create a mock part object with proper string name attribute.

    Args:
        name: Part name (e.g., 'dertag_1900-1902')
        url: Download URL
        years: Year range (e.g., '1900-1902')
        md5: MD5 checksum
        size: Human-readable size

    Returns:
        Mock object mimicking a SourcePart model
    """
    part = Mock()
    part.name = name
    part.url = url
    part.years = years
    part.md5 = md5
    part.size = size
    part.model_dump.return_value = {
        "name": name,
        "url": url,
        "years": years,
        "md5": md5,
        "size": size,
    }
    return part


@pytest.fixture
def mock_source_config_der_tag() -> Mock:
    """
    Mock source configuration for 'der_tag' dataset.

    Returns a Mock mimicking SourceConfig for the Der Tag newspaper.
    """
    mock = Mock()
    mock.dataset_name = "der_tag"
    mock.data_type = "xml_ocr"
    mock.get_year_range.return_value = (1900, 1920)
    mock.metadata = Mock()
    mock.metadata.newspaper_title = "Der Tag"
    mock.metadata.language = "de"
    mock.parts = [
        create_mock_part(
            name="dertag_1900-1902",
            url="https://zenodo.org/test/dertag_1900-1902.tar.gz",
            years="1900-1902",
            md5="abc123",
            size="1.4 GB",
        ),
        create_mock_part(
            name="dertag_1903-1905",
            url="https://zenodo.org/test/dertag_1903-1905.tar.gz",
            years="1903-1905",
            md5="def456",
            size="375.8 MB",
        ),
    ]
    return mock


@pytest.fixture
def mock_source_config_generic() -> Mock:
    """
    Mock source configuration for a generic test source.

    Returns a Mock mimicking SourceConfig for testing purposes.
    """
    mock = Mock()
    mock.dataset_name = "test_source"
    mock.data_type = "xml_ocr"
    mock.get_year_range.return_value = (1900, 1920)
    mock.metadata = Mock()
    mock.metadata.newspaper_title = "Test Newspaper"
    mock.metadata.language = "de"
    mock.parts = []
    return mock


# =============================================================================
# XML/METS Fixtures
# =============================================================================


@pytest.fixture
def sample_mets_xml() -> str:
    """Sample METS XML with MAX fileGrp for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
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


@pytest.fixture
def sample_mets_xml_no_max() -> str:
    """Sample METS XML without MAX fileGrp for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<mets:mets xmlns:mets="http://www.loc.gov/METS/">
    <mets:fileSec>
        <mets:fileGrp USE="DEFAULT">
        </mets:fileGrp>
    </mets:fileSec>
</mets:mets>"""


@pytest.fixture
def sample_alto_xml() -> str:
    """Sample ALTO XML for testing text extraction."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
    <Layout>
        <Page WIDTH="2000" HEIGHT="3000">
            <PrintSpace>
                <TextBlock ID="TB1" HPOS="100" VPOS="100" WIDTH="500" HEIGHT="200">
                    <TextLine ID="TL1" HPOS="100" VPOS="100" WIDTH="500" HEIGHT="50">
                        <String CONTENT="Hello" HPOS="100" VPOS="100" WIDTH="100" HEIGHT="50"/>
                        <String CONTENT="World" HPOS="220" VPOS="100" WIDTH="100" HEIGHT="50"/>
                    </TextLine>
                </TextBlock>
            </PrintSpace>
        </Page>
    </Layout>
</alto>"""


# =============================================================================
# Directory Structure Fixtures
# =============================================================================


@pytest.fixture
def xml_dir_structure(tmp_path: Path) -> Path:
    """
    Create a mock XML directory structure for testing.

    Creates:
        tmp_path/xml_ocr/
            1920/03/03/
                issue.xml
                fulltext/
                    page1.xml
                    page2.xml

    Returns the xml_dir path.
    """
    xml_dir = tmp_path / "xml_ocr"
    issue_dir = xml_dir / "1920" / "03" / "03"
    issue_dir.mkdir(parents=True)

    # Create METS file
    (issue_dir / "issue.xml").touch()

    # Create fulltext directory with ALTO files
    fulltext_dir = issue_dir / "fulltext"
    fulltext_dir.mkdir()
    (fulltext_dir / "page1.xml").touch()
    (fulltext_dir / "page2.xml").touch()

    return xml_dir


@pytest.fixture
def images_dir_structure(tmp_path: Path) -> Path:
    """
    Create a mock images directory structure for testing.

    Returns the images_dir path.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return images_dir


# =============================================================================
# Basic Text DataFrame Fixtures
# =============================================================================


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Basic DataFrame for testing."""
    return pl.DataFrame({"text": ["Hello World", "Test Text", "Another Line"]})


@pytest.fixture
def german_df() -> pl.DataFrame:
    """DataFrame with German text including umlauts and special characters."""
    return pl.DataFrame(
        {
            "text": [
                "Münchner Straße",
                "Die größte Überraschung",
                "Äpfel und Öl",
                "Fließendes Wasser",
            ]
        }
    )


@pytest.fixture
def empty_df() -> pl.DataFrame:
    """DataFrame with empty strings."""
    return pl.DataFrame({"text": ["", "Test", ""]})


@pytest.fixture
def null_df() -> pl.DataFrame:
    """DataFrame with null values."""
    return pl.DataFrame({"text": ["Test", None, "Another"]})


# =============================================================================
# Whitespace Fixtures
# =============================================================================


@pytest.fixture
def whitespace_df() -> pl.DataFrame:
    """DataFrame with various whitespace patterns."""
    return pl.DataFrame(
        {
            "text": [
                "Hello    World",
                "Multiple   spaces   here",
                "Tabs\there",
                "Newlines\nand\nmore",
                "Mixed \t \n whitespace",
                "  leading and trailing  ",
            ]
        }
    )


# =============================================================================
# Punctuation and Numbers Fixtures
# =============================================================================


@pytest.fixture
def punctuation_df() -> pl.DataFrame:
    """DataFrame with various punctuation marks."""
    return pl.DataFrame(
        {
            "text": [
                "Hello, World!",
                "Question? Answer.",
                "Semi;colon:here",
                "Quotes \"and\" 'more'",
                "Brackets (and) [more] {braces}",
                "Dashes - and — more",
            ]
        }
    )


@pytest.fixture
def numbers_df() -> pl.DataFrame:
    """DataFrame with various number formats."""
    return pl.DataFrame(
        {
            "text": [
                "Year 1920",
                "Price 123.45",
                "Page 42",
                "Mixed text123numbers",
                "Subscript₁₂₃",
                "Roman numerals: XII",
            ]
        }
    )


# =============================================================================
# Language-Specific Fixtures
# =============================================================================


@pytest.fixture
def stopwords_df() -> pl.DataFrame:
    """DataFrame with German and English stopwords."""
    return pl.DataFrame(
        {
            "text": [
                "Der Mann und die Frau",
                "The man and the woman",
                "Das ist ein Test",
                "This is a test",
            ]
        }
    )


@pytest.fixture
def french_df() -> pl.DataFrame:
    """DataFrame with French text including accents."""
    return pl.DataFrame({"text": ["café", "naïve", "résumé", "crème brûlée"]})


# =============================================================================
# OCR and Artifact Fixtures
# =============================================================================


@pytest.fixture
def ocr_artifacts_df() -> pl.DataFrame:
    """DataFrame with OCR artifacts and invalid characters."""
    return pl.DataFrame(
        {
            "text": [
                "Normal text here",
                "Text with ™ symbols ©",
                "Control\x00chars\x01here",
                "Unicode garbage: \u200b\u200c\u200d",
                "Mixed: Valid äöü and invalid §€£",
            ]
        }
    )


@pytest.fixture
def hyphenated_df() -> pl.DataFrame:
    """DataFrame with hyphenated words (line breaks in OCR)."""
    return pl.DataFrame(
        {
            "text": [
                "Über-\nraschung",
                "Wissen-\nschaft",
                "Normal text",
                "Ent-\nwicklung",
            ]
        }
    )


# =============================================================================
# Historical Text Fixtures
# =============================================================================


@pytest.fixture
def historical_german_df() -> pl.DataFrame:
    """DataFrame with historical German text (long s, old spellings)."""
    return pl.DataFrame(
        {
            "text": [
                "Das iſt ein Teſt",  # long s  # noqa: RUF001
                "Thür und Thor",  # Old spellings
                "Weib und Kind",
                "ſehr ſchön",  # Multiple long s # noqa: RUF001
            ]
        }
    )


@pytest.fixture
def mixed_case_df() -> pl.DataFrame:
    """DataFrame with mixed case text."""
    return pl.DataFrame(
        {
            "text": [
                "Hello World",
                "UPPERCASE TEXT",
                "lowercase text",
                "MiXeD CaSe TeXt",
            ]
        }
    )
