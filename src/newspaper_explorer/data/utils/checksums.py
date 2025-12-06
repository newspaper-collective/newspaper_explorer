"""
Checksum utilities for file integrity verification.

General-purpose utilities for calculating and verifying file checksums.
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_md5(filepath: Path) -> str:
    """
    Calculate MD5 checksum of a file.

    Args:
        filepath: Path to the file

    Returns:
        MD5 checksum as hex string

    Note:
        MD5 is used here for file integrity verification (checksums), not cryptographic security.
        This is acceptable for detecting file corruption during download.

    Example:
        >>> checksum = calculate_md5(Path("archive.tar.gz"))
        >>> print(checksum)
        'd41d8cd98f00b204e9800998ecf8427e'
    """
    md5_hash = hashlib.md5(usedforsecurity=False)  # Explicitly mark non-security use
    with filepath.open("rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """
    Verify file checksum matches expected value.

    Args:
        filepath: Path to the file to verify
        expected_md5: Expected MD5 checksum

    Returns:
        True if checksum matches, False otherwise

    Example:
        >>> if not verify_checksum(Path("archive.tar.gz"), "abc123..."):
        ...     print("File corrupted!")
    """
    logger.info("Verifying checksum...")
    actual_md5 = calculate_md5(filepath)

    if actual_md5 == expected_md5:
        logger.info(f"Checksum verified: {actual_md5}")
        return True

    logger.warning("Checksum mismatch!")
    logger.warning(f"  Expected: {expected_md5}")
    logger.warning(f"  Got:      {actual_md5}")

    return False
