"""
Tests for ALTO XML data loading functionality.
"""

from datetime import datetime
from pathlib import Path

import pytest

from newspaper_explorer.data.loading.loader import DataLoader
from newspaper_explorer.data.parser.alto import ALTOParser, TextLine


class TestTextLine:
    """Test TextLine dataclass"""

    def test_text_line_creation(self):
        """Test creating a TextLine object"""
        line = TextLine(
            line_id="test_line_1",
            text="Sample text",
            text_block_id="block_1",
            filename="test.xml",
            date=datetime(1901, 1, 1),
            x=100,
            y=200,
            width=500,
            height=20,
            newspaper_id="3074409X",
        )

        assert line.line_id == "test_line_1"
        assert line.text == "Sample text"
        assert line.year == 1901  # Property access
        assert line.month == 1  # Property access
        assert line.day == 1  # Property access

    def test_text_line_to_dict(self):
        """Test converting TextLine to dictionary"""
        line = TextLine(
            line_id="test_line_1",
            text="Sample text",
            text_block_id="block_1",
            filename="test.xml",
        )

        d = line.to_dict()
        assert isinstance(d, dict)
        assert d["line_id"] == "test_line_1"
        assert d["text"] == "Sample text"


class TestALTOParser:
    """Test ALTO XML parser"""

    def test_parse_filename(self):
        """Test filename parsing"""
        parser = ALTOParser()

        # Standard format
        (
            newspaper_id,
            date,
            issue_number,
            daily_issue_number,
            page_number,
        ) = parser._parse_filename("3074409X_1901-02-15_000_415_H_2_009.xml")

        assert newspaper_id == "3074409X"
        assert date == datetime(1901, 2, 15)
        assert issue_number == 415
        assert daily_issue_number == 2
        assert page_number == 9

    def test_parse_filename_invalid(self):
        """Test filename parsing with invalid filename"""
        parser = ALTOParser()

        result = parser._parse_filename("invalid.xml")

        assert all(v is None for v in result)


class TestDataLoader:
    """Test DataLoader functionality"""

    def test_dataloader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader(max_workers=2)
        assert loader.max_workers == 2

    def test_dataloader_default_workers(self):
        """Test DataLoader with default workers"""
        loader = DataLoader()
        assert loader.max_workers >= 1


# Integration test (requires actual data)
@pytest.mark.integration
def test_parse_real_file():
    """Test parsing a real ALTO XML file"""
    # This test requires actual data files
    # Skip if data directory doesn't exist
    data_dir = Path("data/raw/der_tag/xml_ocr")
    if not data_dir.exists():
        pytest.skip("Data directory not found")

    # Find ALTO files (in fulltext directories)
    xml_files = list(data_dir.glob("**/fulltext/*.xml"))
    if not xml_files:
        pytest.skip("No ALTO XML files found")

    parser = ALTOParser()
    lines = parser.parse_file(xml_files[0])

    assert len(lines) > 0
    assert all(isinstance(line, TextLine) for line in lines)
    assert all(line.text for line in lines)


@pytest.mark.integration
def test_dataloader_load_small_batch():
    """Test loading a small batch of real ALTO files"""
    data_dir = Path("data/raw/der_tag/xml_ocr")
    if not data_dir.exists():
        pytest.skip("Data directory not found")

    # Find ALTO files
    xml_files = list(data_dir.glob("**/fulltext/*.xml"))
    if len(xml_files) < 3:
        pytest.skip("Not enough ALTO XML files found")

    loader = DataLoader(max_workers=2)

    # Load just 3 files for testing
    df = loader.load_directory(data_dir, pattern="**/fulltext/*.xml", max_files=3, auto_save=False)

    # Verify DataFrame structure
    assert len(df) > 0
    assert "text" in df.columns
    assert "line_id" in df.columns
    assert "text_block_id" in df.columns
    assert "filename" in df.columns
    assert "date" in df.columns
    assert "newspaper_id" in df.columns

    # Verify we have data from 3 files
    unique_files = df["filename"].n_unique()
    assert unique_files == 3

    # Verify text content exists
    assert df["text"].null_count() == 0
    assert all(len(text) > 0 for text in df["text"].to_list()[:10])


@pytest.mark.integration
def test_dataloader_with_mets_enrichment():
    """Test that METS metadata is properly enriched"""
    data_dir = Path("data/raw/der_tag/xml_ocr")
    if not data_dir.exists():
        pytest.skip("Data directory not found")

    # Find ALTO files
    xml_files = list(data_dir.glob("**/fulltext/*.xml"))
    if not xml_files:
        pytest.skip("No ALTO XML files found")

    loader = DataLoader(max_workers=1)

    # Load just 1 file
    df = loader.load_directory(data_dir, pattern="**/fulltext/*.xml", max_files=1, auto_save=False)

    assert len(df) > 0

    # Check if METS metadata columns exist and have values
    if "newspaper_title" in df.columns:
        # METS file was found and parsed
        assert df["newspaper_title"].null_count() < len(df)  # At least some values
        print(f"METS enrichment working: title = {df['newspaper_title'][0]}")
    else:
        # METS file not found - this is okay for the test
        print("No METS metadata found (this is okay)")


@pytest.mark.integration
def test_dataloader_save_parquet():
    """Test saving loaded data to parquet file"""
    import tempfile

    data_dir = Path("data/raw/der_tag/xml_ocr")
    if not data_dir.exists():
        pytest.skip("Data directory not found")

    xml_files = list(data_dir.glob("**/fulltext/*.xml"))
    if not xml_files:
        pytest.skip("No ALTO XML files found")

    loader = DataLoader(max_workers=1)

    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Load and save
        df = loader.load_directory(
            data_dir,
            pattern="**/fulltext/*.xml",
            max_files=2,
            output_parquet=tmp_path,
            auto_save=False,
        )

        # Verify file was created
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0

        # Load it back and verify
        df_loaded = loader.load_parquet(tmp_path)
        assert len(df_loaded) == len(df)
        assert df_loaded.columns == df.columns

    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()
