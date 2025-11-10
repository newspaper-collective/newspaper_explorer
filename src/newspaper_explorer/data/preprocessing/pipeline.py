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
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

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
    filter_by_word_count,
    clean_ocr_artifacts,
)
from newspaper_explorer.data.preprocessing.linguistic import (
    dehyphenate,
    lemmatize_spacy,
    lemmatize_germalemma,
)
from newspaper_explorer.data.utils.metadata import PreprocessingMetadata

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Chains together multiple text preprocessing steps in sequence.

    This class provides a convenient pipeline() method that applies preprocessing
    steps (like normalization, cleaning, lemmatization) one after another to a text
    column in a Polars DataFrame.

    Example:
        >>> preprocessor = TextPreprocessor(text_column="text")
        >>> df = preprocessor.pipeline(
        ...     df,
        ...     steps=["normalize", "lowercase", "remove-stopwords"],
        ...     output_column="text_processed"
        ... )

    Note:
        All original DataFrame columns (including IDs and metadata) are preserved
        in the output, so you can still trace processed text back to its source.

        For more control, you can import and use preprocessing functions directly:
        from newspaper_explorer.data.preprocessing.cleaning import lowercase
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
        batch_size: int = 32,
        num_beams: int = 4,
        num_gpus: int = 1,
        use_cache: bool = True,
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
        - filter-length: Filter out too short/long texts (by character count)
        - filter-word-count: Filter out too few/many words (by word count)
        - clean-ocr: Remove OCR artifacts and invalid characters

        Args:
            df: Input DataFrame (must contain text_column)
            steps: List of step names to apply in order
            output_column: Name for final output column
            batch_size: Batch size for transnormer inference (default: 32)
            num_beams: Number of beams for transnormer generation (default: 4)
            num_gpus: Number of GPUs for parallel transnormer processing (default: 1)

        Returns:
            DataFrame with processed text and all original columns preserved.
            Foreign keys (source_id, issue_id, page_id, text_block_id, line_id)
            are automatically preserved for traceability.

        Note:
            All columns from the input DataFrame are preserved in the output,
            ensuring that foreign keys and metadata remain available after
            preprocessing for linking back to source data.
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
                    batch_size=batch_size,
                    num_beams=num_beams,
                    num_gpus=num_gpus,
                    use_cache=use_cache,
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
            elif step == "filter-word-count":
                df = filter_by_word_count(
                    df, text_column=current_column, input_column=current_column
                )
                # Note: filter-word-count doesn't change the column, so keep current_column
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

    def load_previous_metadata(self, input_path: Union[Path, str]) -> Optional[Dict[str, Any]]:
        """
        Load preprocessing metadata from input file if it exists.

        Args:
            input_path: Path to input parquet file

        Returns:
            Dictionary with previous preprocessing info, or None if not found
        """
        from newspaper_explorer.data.utils.metadata import (
            find_metadata_for_parquet,
            load_metadata,
            PreprocessingMetadata,
        )

        input_path = Path(input_path)
        metadata_path = find_metadata_for_parquet(input_path)

        if metadata_path:
            try:
                metadata = load_metadata(metadata_path)
                if isinstance(metadata, PreprocessingMetadata):
                    logger.info(
                        f"Found previous preprocessing: {metadata.preprocessing_id} "
                        f"({len(metadata.steps)} steps)"
                    )
                    return metadata.to_dict()
            except Exception as e:
                logger.debug(f"Could not load preprocessing metadata: {e}")

        return None

    def create_metadata(
        self,
        source: str,
        steps: List[str],
        parameters: Dict[str, Any],
        input_df: pl.DataFrame,
        output_df: pl.DataFrame,
        duration_seconds: float,
        previous_preprocessing: Optional[Dict[str, Any]] = None,
    ) -> PreprocessingMetadata:
        """
        Create preprocessing metadata for saving alongside results.

        Args:
            source: Source dataset name
            steps: List of preprocessing steps applied
            parameters: Processing parameters
            input_df: Input DataFrame (for statistics)
            output_df: Output DataFrame (for statistics)
            duration_seconds: Processing time
            previous_preprocessing: Previous preprocessing metadata if input was preprocessed

        Returns:
            PreprocessingMetadata object ready to save
        """
        from newspaper_explorer.data.utils.metadata import (
            PreprocessingMetadata,
            extract_input_stats,
            extract_output_stats,
        )

        metadata = PreprocessingMetadata(
            source=source,
            steps=steps,
            parameters=parameters,
            input_data=extract_input_stats(input_df),
            output_data=extract_output_stats(output_df),
            duration_seconds=duration_seconds,
            previous_preprocessing=previous_preprocessing,
            status="completed",
            completed_at=datetime.now().isoformat(),
            preprocessing_id=None,  # Will be auto-generated
            error_message=None,  # Not needed for completed status
        )

        return metadata
