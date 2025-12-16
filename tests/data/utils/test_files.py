"""Tests for file utilities."""

from pathlib import Path

import pytest

from newspaper_explorer.data.utils.files import find_mets_files, find_xml_files


class TestFindXMLFiles:
    """Tests for find_xml_files function."""

    def test_find_xml_files_in_directory(self, tmp_path: Path) -> None:
        """Test finding XML files in a directory."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        # Create some XML files
        (xml_dir / "file1.xml").write_text("<root/>")
        (xml_dir / "file2.xml").write_text("<root/>")
        (xml_dir / "file3.txt").write_text("not xml")

        files = find_xml_files(xml_dir)

        assert len(files) == 2
        assert all(f.suffix == ".xml" for f in files)

    def test_find_xml_files_recursive(self, tmp_path: Path) -> None:
        """Test finding XML files recursively."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        # Create nested structure
        (xml_dir / "file1.xml").write_text("<root/>")

        sub_dir = xml_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file2.xml").write_text("<root/>")

        files = find_xml_files(xml_dir)

        assert len(files) == 2

    def test_find_xml_files_with_custom_pattern(self, tmp_path: Path) -> None:
        """Test finding XML files with custom pattern."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        fulltext_dir = xml_dir / "fulltext"
        fulltext_dir.mkdir()

        (xml_dir / "mets.xml").write_text("<mets/>")
        (fulltext_dir / "alto.xml").write_text("<alto/>")

        # Find only files in fulltext
        files = find_xml_files(xml_dir, "**/fulltext/*.xml")

        assert len(files) == 1
        assert "fulltext" in str(files[0])

    def test_find_xml_files_returns_sorted(self, tmp_path: Path) -> None:
        """Test that files are returned in natural sorted order."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        # Create files in non-alphabetical order
        (xml_dir / "file10.xml").write_text("<root/>")
        (xml_dir / "file2.xml").write_text("<root/>")
        (xml_dir / "file1.xml").write_text("<root/>")

        files = find_xml_files(xml_dir)

        # Should be naturally sorted: file1, file2, file10
        assert files[0].name == "file1.xml"
        assert files[1].name == "file2.xml"
        assert files[2].name == "file10.xml"

    def test_find_xml_files_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test finding XML files in non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"

        files = find_xml_files(nonexistent)

        assert files == []

    def test_find_xml_files_empty_directory(self, tmp_path: Path) -> None:
        """Test finding XML files in empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        files = find_xml_files(empty_dir)

        assert files == []


class TestFindMETSFiles:
    """Tests for find_mets_files function."""

    def test_find_mets_files_excludes_fulltext(self, tmp_path: Path) -> None:
        """Test that METS files in fulltext directories are excluded."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        # Create METS file in root
        (xml_dir / "mets.xml").write_text("<mets/>")

        # Create ALTO file in fulltext (should be excluded)
        fulltext_dir = xml_dir / "fulltext"
        fulltext_dir.mkdir()
        (fulltext_dir / "alto.xml").write_text("<alto/>")

        mets_files = find_mets_files(xml_dir)

        assert len(mets_files) == 1
        assert "fulltext" not in str(mets_files[0])

    def test_find_mets_files_realistic_structure(self, tmp_path: Path) -> None:
        """Test finding METS files in realistic directory structure."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        # Create realistic structure: year/month/day/issue/mets.xml
        issue_dir = xml_dir / "1920" / "03" / "15" / "issue_001_ed_1"
        issue_dir.mkdir(parents=True)

        (issue_dir / "1920031501_mets.xml").write_text("<mets/>")

        # Create fulltext directory with ALTO files
        fulltext_dir = issue_dir / "fulltext"
        fulltext_dir.mkdir()
        (fulltext_dir / "page_001.xml").write_text("<alto/>")
        (fulltext_dir / "page_002.xml").write_text("<alto/>")

        mets_files = find_mets_files(xml_dir)

        assert len(mets_files) == 1
        assert mets_files[0].name == "1920031501_mets.xml"

    def test_find_mets_files_multiple_issues(self, tmp_path: Path) -> None:
        """Test finding METS files across multiple issues."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        # Create multiple issue directories
        for issue in range(1, 4):
            issue_dir = xml_dir / f"issue_{issue:03d}"
            issue_dir.mkdir()
            (issue_dir / f"mets_{issue}.xml").write_text("<mets/>")

        mets_files = find_mets_files(xml_dir)

        assert len(mets_files) == 3

    def test_find_mets_files_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test finding METS files in non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"

        mets_files = find_mets_files(nonexistent)

        assert mets_files == []

    def test_find_mets_files_returns_sorted(self, tmp_path: Path) -> None:
        """Test that METS files are returned in natural sorted order."""
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()

        # Create files in non-chronological order
        (xml_dir / "mets_10.xml").write_text("<mets/>")
        (xml_dir / "mets_2.xml").write_text("<mets/>")
        (xml_dir / "mets_1.xml").write_text("<mets/>")

        mets_files = find_mets_files(xml_dir)

        # Should be naturally sorted
        assert mets_files[0].name == "mets_1.xml"
        assert mets_files[1].name == "mets_2.xml"
        assert mets_files[2].name == "mets_10.xml"

    def test_find_mets_files_with_fixtures(self) -> None:
        """Test finding METS files in fixtures directory."""
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "mets"

        if not fixtures_dir.exists():
            pytest.skip("Fixtures directory not found")

        mets_files = find_mets_files(fixtures_dir)

        # Should find fixture METS files
        assert len(mets_files) > 0
        assert all(f.suffix == ".xml" for f in mets_files)
