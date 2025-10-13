"""
Module to clean and normalise data

TODO
- join words broken by linewrap
- lowercase?
- remove special characters?
- remove/replace multiple spaces
- remove/replace hyphens?
- remove/replace newlines
- remove/replace tabs
- remove/replace non-UTF8 characters
- normalise spelling, historical variations

"""



class DataCleaner:
    """
    Class to handle data cleaning and normalisation.
    """

    def __init__(self):
        pass

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalise the input text.

        Args:
            text (str): The raw text to be cleaned.

        Returns:
            str: The cleaned and normalised text.
        """
        # Implement cleaning and normalisation logic here
        return text

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text (e.g., lowercasing, removing special characters).

        Args:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        # Implement normalization logic here
        return text

    def clean_data(self, raw_data: str) -> str:
        """
        Public method to clean and normalise raw data.

        Args:
            raw_data (str): The raw data string.

        Returns:
            str: The cleaned and normalised data.
        """
        cleaned = self._clean_text(raw_data)
        normalized = self._normalize_text(cleaned)
        return normalized