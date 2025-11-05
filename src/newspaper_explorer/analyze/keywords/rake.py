"""
RAKE keyword extraction for newspaper texts.

Implements RAKE (Rapid Automatic Keyword Extraction) to extract multi-word
keyphrases from newspaper articles. RAKE identifies candidate keywords based
on word co-occurrence and word frequency, making it excellent for extracting
meaningful phrases rather than just single words.

Unlike TF-IDF (which finds statistically distinctive words) or LDA (which finds
topic-based terms), RAKE extracts semantically meaningful keyphrases based on
linguistic patterns.

Example:
    >>> from newspaper_explorer.analyze.keywords.rake import RAKEExtractor
    >>> extractor = RAKEExtractor(source_name="der_tag")
    >>> keyphrases = extractor.extract_keyphrases(top_k=10)
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import polars as pl
from rake_nltk import Rake
from tqdm import tqdm

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)


class RAKEExtractor:
    """
    Extract keyphrases from newspaper texts using RAKE.

    RAKE (Rapid Automatic Keyword Extraction) identifies candidate keyphrases
    by analyzing word co-occurrence patterns and word frequencies. It's
    particularly good at extracting multi-word expressions like "erste weltkrieg"
    or "sozialdemokratische partei".

    Key advantages:
    - Extracts meaningful multi-word phrases
    - Language-independent (works with any language)
    - Fast and efficient
    - No training required

    Methodology:
    1. Splits text on delimiters (punctuation, stopwords)
    2. Identifies candidate keyphrases (sequences of content words)
    3. Scores phrases based on word frequency and co-occurrence
    4. Returns top-ranked phrases
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        use_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        min_phrase_length: int = 1,
        max_phrase_length: int = 3,
    ):
        """
        Initialize RAKE extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            use_stopwords: Whether to use German stopwords
            custom_stopwords: Additional stopwords to exclude
            min_phrase_length: Minimum words per keyphrase
            max_phrase_length: Maximum words per keyphrase
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()
        self.min_phrase_length = min_phrase_length
        self.max_phrase_length = max_phrase_length

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to textblocks for better phrase extraction
            source_dir = self.config.data_dir / "raw" / source_name / "text"
            textblocks_file = source_dir / f"{source_name}_textblocks.parquet"
            lines_file = source_dir / f"{source_name}_lines.parquet"

            if textblocks_file.exists():
                self.input_file = textblocks_file
                logger.info("Using textblocks.parquet (aggregated text blocks)")
            elif lines_file.exists():
                self.input_file = lines_file
                logger.info("Using lines.parquet (line-level data)")
            else:
                self.input_file = textblocks_file

        # Setup stopwords for RAKE
        stopwords = self._get_stopwords(use_stopwords, custom_stopwords)

        # Initialize RAKE with German stopwords
        self.rake = Rake(
            stopwords=stopwords,
            min_length=min_phrase_length,
            max_length=max_phrase_length,
        )

        logger.info(f"Initialized RAKE extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Phrase length: {min_phrase_length}-{max_phrase_length} words")
        logger.info(f"Using {len(stopwords)} stopwords")

    def _get_stopwords(
        self, use_stopwords: bool, custom_stopwords: Optional[List[str]]
    ) -> List[str]:
        """Get German stopwords list."""
        stopwords = []

        if use_stopwords:
            try:
                from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS

                stopwords = list(DE_STOP_WORDS)
                logger.info(f"Loaded {len(stopwords)} German stopwords from SpaCy")
            except ImportError:
                logger.warning(
                    "SpaCy not installed, using basic German stopwords. "
                    "Install with: pip install -e '.[nlp]' for better stopword list"
                )
                # Basic German stopwords
                stopwords = [
                    "der",
                    "die",
                    "das",
                    "und",
                    "in",
                    "zu",
                    "den",
                    "ist",
                    "von",
                    "mit",
                    "auf",
                    "für",
                    "als",
                    "an",
                    "im",
                    "dem",
                    "ein",
                    "eine",
                    "nicht",
                    "auch",
                    "sich",
                    "wird",
                    "oder",
                    "aus",
                    "werden",
                    "bei",
                    "nach",
                    "um",
                    "am",
                    "des",
                    "durch",
                    "einem",
                    "einer",
                    "bis",
                    "sind",
                    "war",
                    "nur",
                    "noch",
                    "kann",
                    "hat",
                    "wir",
                    "sie",
                ]

        # Add custom stopwords
        if custom_stopwords:
            stopwords.extend(custom_stopwords)
            logger.info(f"Added {len(custom_stopwords)} custom stopwords")

        return stopwords

    def extract_keyphrases(
        self,
        top_k: int = 10,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Extract keyphrases from documents.

        Args:
            top_k: Number of top keyphrases per document
            limit: Limit number of documents to process
            group_by: Columns to group by (aggregates text)

        Returns:
            DataFrame with columns: doc_id, keyphrases, scores
        """
        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.input_file}\n"
                f"Run parsing first: newspaper-explorer data parse --source {self.source_name}"
            )

        logger.info(f"Loading data from {self.input_file}")
        df = pl.read_parquet(self.input_file)

        if limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} rows")

        # Group if requested
        if group_by:
            logger.info(f"Grouping by: {', '.join(group_by)}")
            df = df.group_by(group_by).agg(pl.col(self.text_column).str.concat(" "))

        # Extract document IDs and texts
        texts = df[self.text_column].to_list()

        # Create document IDs
        if "text_block_id" in df.columns:
            doc_ids = df["text_block_id"].to_list()
        elif "filename" in df.columns:
            doc_ids = df["filename"].to_list()
        elif group_by:
            # For grouped data, create ID from group columns
            doc_ids = [
                "_".join([str(row[col]) for col in group_by])
                for row in df.select(group_by).iter_rows(named=True)
            ]
        else:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]

        # Extract keyphrases
        logger.info(f"Extracting keyphrases from {len(texts)} documents...")
        results = []

        for doc_id, text in tqdm(
            zip(doc_ids, texts), total=len(texts), desc="Extracting keyphrases"
        ):
            if not text or not isinstance(text, str):
                results.append(
                    {
                        "doc_id": doc_id,
                        "keyphrases": [],
                        "scores": [],
                    }
                )
                continue

            # Extract keyphrases
            self.rake.extract_keywords_from_text(text)
            ranked_phrases = self.rake.get_ranked_phrases_with_scores()

            # Take top k
            top_phrases = ranked_phrases[:top_k]

            if top_phrases:
                scores, phrases = zip(*top_phrases)
                results.append(
                    {
                        "doc_id": doc_id,
                        "keyphrases": list(phrases),
                        "scores": [float(s) for s in scores],
                    }
                )
            else:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keyphrases": [],
                        "scores": [],
                    }
                )

        results_df = pl.DataFrame(results)
        logger.info(f"Extracted keyphrases for {len(results_df)} documents")

        # Add grouping columns if they exist
        if group_by:
            # Merge back grouping columns
            group_data = df.select(group_by)
            group_data = group_data.with_columns(pl.Series("doc_id", doc_ids))
            results_df = results_df.join(group_data, on="doc_id", how="left")

        return results_df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "rake_keyphrases",
    ) -> Path:
        """
        Save results to parquet file.

        Args:
            results_df: Results DataFrame
            output_name: Output filename (without extension)

        Returns:
            Path to saved file
        """
        output_dir = self.config.results_dir / self.source_name / "keywords"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{output_name}.parquet"
        results_df.write_parquet(output_file)

        logger.info(f"Saved results to {output_file}")
        return output_file


def extract_keyphrases_simple(
    source_name: str,
    top_k: int = 10,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Simple one-shot function to extract keyphrases using RAKE.

    Args:
        source_name: Source name (e.g., "der_tag")
        top_k: Keyphrases per document
        limit: Limit documents

    Returns:
        DataFrame with keyphrases

    Example:
        >>> keyphrases_df = extract_keyphrases_simple("der_tag", top_k=10)
        >>> print(keyphrases_df)
    """
    extractor = RAKEExtractor(source_name=source_name)
    return extractor.extract_keyphrases(top_k=top_k, limit=limit)
