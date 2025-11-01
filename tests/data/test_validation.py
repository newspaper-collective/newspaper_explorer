"""
Tests for data validation utilities, including image validation.
"""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from newspaper_explorer.data.utils.validation import (
    ImageValidationResult,
    check_image_size,
    validate_image_file,
)


class TestImageValidation:
    """Tests for image validation functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def valid_image(self, temp_dir):
        """Create a valid test image."""
        img_path = temp_dir / "valid.jpg"
        # Create a larger 500x500 red image to ensure it's above 1KB
        img = Image.new("RGB", (500, 500), color="red")
        img.save(img_path, "JPEG", quality=95)
        return img_path

    @pytest.fixture
    def small_image(self, temp_dir):
        """Create a very small (but valid) test image."""
        img_path = temp_dir / "small.jpg"
        # Create a 1x1 pixel image
        img = Image.new("RGB", (1, 1), color="blue")
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def empty_file(self, temp_dir):
        """Create an empty file."""
        file_path = temp_dir / "empty.jpg"
        file_path.touch()
        return file_path

    @pytest.fixture
    def corrupt_file(self, temp_dir):
        """Create a corrupted image file (larger than 1KB but not valid image)."""
        file_path = temp_dir / "corrupt.jpg"
        with open(file_path, "wb") as f:
            # Write enough bytes to pass size check but not be a valid image
            f.write(b"This is not a valid image file" * 100)
        return file_path

    def test_validate_valid_image(self, valid_image):
        """Test validation of a valid image."""
        result = validate_image_file(valid_image)

        assert isinstance(result, ImageValidationResult)
        assert result.is_valid is True
        assert result.file_path == valid_image
        assert result.file_size > 1024  # Should be larger than 1KB
        assert result.width == 500
        assert result.height == 500
        assert result.format == "JPEG"
        assert result.error is None

    def test_validate_nonexistent_file(self, temp_dir):
        """Test validation of a file that doesn't exist."""
        nonexistent = temp_dir / "nonexistent.jpg"
        result = validate_image_file(nonexistent)

        assert result.is_valid is False
        assert result.error == "File does not exist"

    def test_validate_empty_file(self, empty_file):
        """Test validation of an empty file."""
        result = validate_image_file(empty_file)

        assert result.is_valid is False
        assert result.file_size == 0
        assert "too small" in result.error.lower()

    def test_validate_corrupt_file(self, corrupt_file):
        """Test validation of a corrupted image file."""
        result = validate_image_file(corrupt_file)

        assert result.is_valid is False
        assert "failed to validate" in result.error.lower()

    def test_validate_small_image_with_custom_threshold(self, small_image):
        """Test validation with custom size threshold."""
        # Small image should fail with default threshold
        result = validate_image_file(small_image, min_size_bytes=1024)
        assert result.is_valid is False

        # But should pass with lower threshold
        result = validate_image_file(small_image, min_size_bytes=10)
        assert result.is_valid is True

    def test_check_image_size_valid(self, valid_image):
        """Test quick size check for valid image."""
        assert check_image_size(valid_image, min_size_bytes=1024) is True

    def test_check_image_size_too_small(self, small_image):
        """Test quick size check for small image."""
        assert check_image_size(small_image, min_size_bytes=1024) is False

    def test_check_image_size_nonexistent(self, temp_dir):
        """Test quick size check for nonexistent file."""
        nonexistent = temp_dir / "nonexistent.jpg"
        assert check_image_size(nonexistent) is False

    def test_validation_result_to_dict(self, valid_image):
        """Test conversion of validation result to dictionary."""
        result = validate_image_file(valid_image)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "is_valid" in result_dict
        assert "file_path" in result_dict
        assert "file_size" in result_dict
        assert "width" in result_dict
        assert "height" in result_dict
        assert "format" in result_dict
        assert "error" in result_dict
        assert result_dict["is_valid"] is True
