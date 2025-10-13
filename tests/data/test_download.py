"""
Tests for download functionality.
"""

import pytest

from newspaper_explorer.data.download import ZenodoDownloader


class TestZenodoDownloader:
    """Test cases for ZenodoDownloader class."""

    def test_init(self, tmp_path):
        """Test ZenodoDownloader initialization."""
        downloader = ZenodoDownloader(data_dir=tmp_path)
        assert downloader.data_dir == tmp_path
        assert downloader.download_dir.exists()
        assert downloader.extracted_dir.exists()

    def test_list_available_parts(self):
        """Test listing available dataset parts."""
        downloader = ZenodoDownloader()
        parts = downloader.list_available_parts()

        assert isinstance(parts, list)
        assert len(parts) > 0

        # Check first part has required fields
        first_part = parts[0]
        assert "name" in first_part
        assert "url" in first_part
        assert "years" in first_part
        assert "md5" in first_part

    def test_get_extraction_status(self):
        """Test getting extraction status."""
        downloader = ZenodoDownloader()
        status = downloader.get_extraction_status()

        assert isinstance(status, dict)
        assert len(status) > 0

        # Check status structure
        for part_name, info in status.items():
            assert "years" in info
            assert "size" in info
            assert "md5" in info
            assert "downloaded" in info
            assert "extracted" in info
            assert isinstance(info["downloaded"], bool)
            assert isinstance(info["extracted"], bool)

    def test_print_status_summary(self, capsys):
        """Test printing status summary."""
        downloader = ZenodoDownloader()
        downloader.print_status_summary()

        captured = capsys.readouterr()
        assert "DATASET STATUS SUMMARY" in captured.out
        assert "Part Name" in captured.out
        assert "Downloaded" in captured.out
        assert "Extracted" in captured.out


@pytest.mark.integration
class TestDownloadIntegration:
    """Integration tests for download functionality (require internet)."""

    def test_download_part_no_extract(self, tmp_path):
        """Test downloading a single part without extraction."""
        downloader = ZenodoDownloader(data_dir=tmp_path)

        # This would require mocking or actual download
        # For now, just test that the method exists and has correct signature
        assert hasattr(downloader, "download_part")
        assert callable(downloader.download_part)

    def test_parallel_download(self, tmp_path):
        """Test parallel download functionality."""
        downloader = ZenodoDownloader(data_dir=tmp_path)

        # Test that the method exists
        assert hasattr(downloader, "download_parts_parallel")
        assert callable(downloader.download_parts_parallel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
