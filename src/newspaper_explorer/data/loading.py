"""
DataLoader class to extract text from raw newspaper data files.

"""

from pathlib import Path

class DataLoader:
    """
    Class to handle loading and extracting text from raw newspaper data files.
    """

    def __init__(self):
        pass

    def extract_text(self, file_path: Path) -> str:
        """
        Extract text from a given raw data file.

        Args:
            file_path (Path): Path to the raw data file.

        Returns:
            str: Extracted text from the raw data file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()