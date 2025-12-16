"""Tests for checksum utilities."""

from pathlib import Path

from newspaper_explorer.data.utils.checksums import calculate_md5_checksum, verify_md5_checksum


class TestCalculateMD5Checksum:
    """Tests for calculate_md5_checksum function."""

    def test_calculate_md5_for_small_file(self, tmp_path: Path) -> None:
        """Test MD5 calculation for a small file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = calculate_md5_checksum(test_file)

        # MD5 of "Hello, World!" is known
        assert checksum == "65a8e27d8879283831b664bd8b7f0ad4"

    def test_calculate_md5_for_empty_file(self, tmp_path: Path) -> None:
        """Test MD5 calculation for an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        checksum = calculate_md5_checksum(test_file)

        # MD5 of empty file is known
        assert checksum == "d41d8cd98f00b204e9800998ecf8427e"

    def test_calculate_md5_for_large_file(self, tmp_path: Path) -> None:
        """Test MD5 calculation for a larger file (tests chunked reading)."""
        test_file = tmp_path / "large.bin"

        # Create a file larger than 8KB to test chunked reading
        data = b"x" * 10240  # 10KB
        test_file.write_bytes(data)

        checksum = calculate_md5_checksum(test_file)

        # Should successfully calculate MD5
        assert len(checksum) == 32
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_calculate_md5_for_binary_file(self, tmp_path: Path) -> None:
        """Test MD5 calculation for a binary file."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(bytes(range(256)))

        checksum = calculate_md5_checksum(test_file)

        # Should handle binary data correctly
        assert len(checksum) == 32

    def test_same_content_gives_same_md5(self, tmp_path: Path) -> None:
        """Test that same content produces same MD5."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        content = "Test content for MD5"
        file1.write_text(content)
        file2.write_text(content)

        checksum1 = calculate_md5_checksum(file1)
        checksum2 = calculate_md5_checksum(file2)

        assert checksum1 == checksum2

    def test_different_content_gives_different_md5(self, tmp_path: Path) -> None:
        """Test that different content produces different MD5."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Content A")
        file2.write_text("Content B")

        checksum1 = calculate_md5_checksum(file1)
        checksum2 = calculate_md5_checksum(file2)

        assert checksum1 != checksum2


class TestVerifyMD5Checksum:
    """Tests for verify_md5_checksum function."""

    def test_verify_correct_checksum(self, tmp_path: Path) -> None:
        """Test verification with correct checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        expected_md5 = "65a8e27d8879283831b664bd8b7f0ad4"

        result = verify_md5_checksum(test_file, expected_md5)

        assert result is True

    def test_verify_incorrect_checksum(self, tmp_path: Path) -> None:
        """Test verification with incorrect checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        wrong_md5 = "0000000000000000000000000000000"

        result = verify_md5_checksum(test_file, wrong_md5)

        assert result is False

    def test_verify_empty_file(self, tmp_path: Path) -> None:
        """Test verification of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        expected_md5 = "d41d8cd98f00b204e9800998ecf8427e"

        result = verify_md5_checksum(test_file, expected_md5)

        assert result is True

    def test_verify_case_sensitive(self, tmp_path: Path) -> None:
        """Test that checksum verification is case-sensitive."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        # MD5 in uppercase will not match lowercase output
        expected_md5_upper = "65A8E27D8879283831B664BD8B7F0AD4"

        result = verify_md5_checksum(test_file, expected_md5_upper)

        # Current implementation is case-sensitive
        assert result is False
