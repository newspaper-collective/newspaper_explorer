"""
Text preprocessing pipeline for newspaper data.

Provides a simple pipeline function that chains together preprocessing steps
of the modular preprocessing modules.

For direct use of preprocessing functions, import from individual modules:
    from newspaper_explorer.data.preprocessing.historical import normalize_historical
    from newspaper_explorer.data.preprocessing.cleaning import lowercase
    from newspaper_explorer.data.preprocessing.filtering import remove_stopwords
    from newspaper_explorer.data.preprocessing.linguistic import lemmatize_spacy
"""

import logging
from typing import List

import polars as pl

# Import all preprocessing functions
from newspaper_explorer.data.preprocessing.normalization import (
    simple,
    transnormer,
    dta_cab,
)
from newspaper_explorer.data.preprocessing.cleaning import (
    normalize_whitespace,
    lowercase,
    remove_diacritics,
)
from newspaper_explorer.data.preprocessing.filtering import (
    remove_punctuation,
    remove_numbers,
    remove_stopwords,
    filter_by_length,
    clean_ocr_artifacts,
)
from newspaper_explorer.data.preprocessing.linguistic import (
    dehyphenate,
    lemmatize_spacy,
    lemmatize_germalemma,
)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Simple preprocessing pipeline for German newspaper text.

    This class only exists for backward compatibility with existing code.
    It provides a pipeline() method to chain preprocessing steps.

    For new code, consider using preprocessing functions directly from their modules.
    """

    def __init__(self, text_column: str = "text"):
        """
        Initialize preprocessor.

        Args:
            text_column: Name of the column containing text to process
        """
        self.text_column = text_column
        logger.debug(f"TextPreprocessor initialized for column: {text_column}")

    def pipeline(
        self,
        df: pl.DataFrame,
        steps: List[str],
        output_column: str = "text_processed",
    ) -> pl.DataFrame:
        """
        Apply a pipeline of preprocessing steps.

        Available steps:
        - normalize: Normalize historical German characters
        - remove-diacritics: Remove diacritics
        - normalize-whitespace: Normalize whitespace
        - normalize-transnormer: Neural normalization with Transnormer
        - normalize-dtacab: API normalization with DTA-CAB
        - lowercase: Convert to lowercase
        - remove-punctuation: Remove punctuation
        - remove-numbers: Remove numbers
        - remove-stopwords: Remove German stopwords
        - dehyphenate: Remove line-break hyphens
        - lemmatize-spacy: Lemmatize with spaCy (fast)
        - lemmatize: Lemmatize with GermaLemma (slow)
        - filter-length: Filter out too short/long texts
        - clean-ocr: Remove OCR artifacts and invalid characters

        Args:
            df: Input DataFrame
            steps: List of step names to apply in order
            output_column: Name for final output column

        Returns:
            DataFrame with processed text
        """
        logger.info(f"Starting preprocessing pipeline with {len(steps)} steps")
        logger.info(f"Steps: {', '.join(steps)}")

        current_column = self.text_column

        for i, step in enumerate(steps, 1):
            logger.info(f"Step {i}/{len(steps)}: {step}")

            if i == len(steps):
                out_col = output_column
            else:
                out_col = f"_tmp_{i}"

            if step == "normalize":
                df = simple(df, text_column=current_column, output_column=out_col)
            elif step == "remove-diacritics":
                df = remove_diacritics(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "normalize-whitespace":
                df = normalize_whitespace(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "normalize-transnormer":
                df = transnormer(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "normalize-dtacab":
                df = dta_cab(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "lowercase":
                df = lowercase(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "remove-punctuation":
                df = remove_punctuation(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "remove-numbers":
                df = remove_numbers(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "remove-stopwords":
                df = remove_stopwords(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "dehyphenate":
                df = dehyphenate(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "lemmatize-spacy":
                df = lemmatize_spacy(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "lemmatize":
                df = lemmatize_germalemma(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            elif step == "filter-length":
                df = filter_by_length(df, text_column=current_column, input_column=current_column)
                # Note: filter-length doesn't change the column, so keep current_column
                current_column = current_column
            elif step == "clean-ocr":
                df = clean_ocr_artifacts(
                    df,
                    text_column=current_column,
                    input_column=current_column,
                    output_column=out_col,
                )
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")

            current_column = out_col

        temp_cols = [col for col in df.columns if col.startswith("_tmp_")]
        if temp_cols:
            df = df.drop(temp_cols)

        logger.info("Preprocessing pipeline complete")
        return df
