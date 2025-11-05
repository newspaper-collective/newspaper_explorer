"""
KeyBERT keyword extraction for newspaper texts.

Implements KeyBERT to extract keywords using BERT embeddings and cosine similarity.
KeyBERT identifies keywords by finding words/phrases most similar to the document
embedding, making it excellent for semantically meaningful keyword extraction.

Unlike statistical methods (TF-IDF, RAKE, YAKE), KeyBERT uses deep learning
embeddings to capture semantic similarity, making it particularly good for:
- Semantic keyword extraction
- Finding conceptually related terms
- Multi-lingual keyword extraction
- Handling synonyms and related concepts

Example:
    >>> from newspaper_explorer.analyze.keywords.keybert import KeyBERTExtractor
    >>> extractor = KeyBERTExtractor(source_name="der_tag")
    >>> keywords = extractor.extract_keywords(top_k=10)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl

# Configure sentence-transformers cache directory to use project models directory
# This keeps model downloads organized and prevents cluttering user home directory
_MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / "models" / "sentence_transformers"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_MODELS_DIR)
from keybert import KeyBERT
from tqdm import tqdm

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)


class KeyBERTExtractor:
    """
    Extract keywords from newspaper texts using KeyBERT.

    KeyBERT uses BERT embeddings to find keywords that are most similar to
    the document. This semantic approach captures meaning beyond word frequency:

    Process:
    1. Generate document embedding using BERT
    2. Generate embeddings for candidate keywords/phrases
    3. Calculate cosine similarity between document and keywords
    4. Return keywords with highest similarity

    Key advantages:
    - Semantically meaningful keywords
    - Captures concepts beyond exact word matches
    - Works well with multi-word expressions
    - Supports multiple languages

    Note: Requires more computational resources than statistical methods.
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        keyphrase_ngram_range: Tuple[int, int] = (1, 2),
        use_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        diversity: float = 0.5,
    ):
        """
        Initialize KeyBERT extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            model_name: Sentence transformer model to use
                - "paraphrase-multilingual-MiniLM-L12-v2" (default, good for German)
                - "distiluse-base-multilingual-cased-v2" (larger, more accurate)
                - "all-MiniLM-L6-v2" (English only, faster)
            keyphrase_ngram_range: N-gram range for candidate phrases (min, max)
            use_stopwords: Whether to filter stopwords from candidates
            custom_stopwords: Additional stopwords to exclude
            diversity: Diversity of keywords (0=similar, 1=diverse) using MMR
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.diversity = diversity

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to textblocks for better keyword extraction
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

        # Setup stopwords
        stopwords_list = self._get_stopwords(use_stopwords, custom_stopwords)
        self.stopwords = set(stopwords_list) if stopwords_list else None

        # Initialize KeyBERT
        logger.info(f"Loading KeyBERT with model: {model_name}")
        try:
            self.model = KeyBERT(model=model_name)
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to default model")
            self.model = KeyBERT()

        logger.info(f"Initialized KeyBERT extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"N-gram range: {keyphrase_ngram_range}")
        logger.info(f"Diversity: {diversity}")
        if self.stopwords:
            logger.info(f"Using {len(self.stopwords)} stopwords")

    def _get_stopwords(
        self, use_stopwords: bool, custom_stopwords: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Get German stopwords list."""
        if not use_stopwords:
            return None

        stopwords = []

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

        return stopwords if stopwords else None

    def extract_keywords(
        self,
        top_k: int = 10,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
        use_mmr: bool = True,
    ) -> pl.DataFrame:
        """
        Extract keywords from documents using KeyBERT.

        Args:
            top_k: Number of top keywords per document
            limit: Limit number of documents to process
            group_by: Columns to group by (aggregates text)
            use_mmr: Use Maximal Marginal Relevance for diversity

        Returns:
            DataFrame with columns: doc_id, keywords, scores

        Note:
            Scores are cosine similarity (0-1, higher = more relevant)
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

        # Extract keywords
        logger.info(f"Extracting keywords from {len(texts)} documents...")
        results = []

        for doc_id, text in tqdm(zip(doc_ids, texts), total=len(texts), desc="Extracting keywords"):
            if not text or not isinstance(text, str) or len(text.strip()) < 20:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )
                continue

            try:
                # Extract keywords with KeyBERT
                if use_mmr and self.diversity > 0:
                    # Use MMR for diverse keywords
                    extracted = self.model.extract_keywords(
                        text,
                        keyphrase_ngram_range=self.keyphrase_ngram_range,
                        stop_words=self.stopwords,
                        top_n=top_k,
                        use_mmr=True,
                        diversity=self.diversity,
                    )
                else:
                    # Use cosine similarity only
                    extracted = self.model.extract_keywords(
                        text,
                        keyphrase_ngram_range=self.keyphrase_ngram_range,
                        stop_words=self.stopwords,
                        top_n=top_k,
                    )

                if extracted:
                    keywords = [kw for kw, score in extracted]
                    scores = [float(score) for kw, score in extracted]

                    results.append(
                        {
                            "doc_id": doc_id,
                            "keywords": keywords,
                            "scores": scores,
                        }
                    )
                else:
                    results.append(
                        {
                            "doc_id": doc_id,
                            "keywords": [],
                            "scores": [],
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to extract keywords for {doc_id}: {e}")
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )

        results_df = pl.DataFrame(results)
        logger.info(f"Extracted keywords for {len(results_df)} documents")

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
        output_name: str = "keybert_keywords",
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


def extract_keywords_simple(
    source_name: str,
    top_k: int = 10,
    limit: Optional[int] = None,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> pl.DataFrame:
    """
    Simple one-shot function to extract keywords using KeyBERT.

    Args:
        source_name: Source name (e.g., "der_tag")
        top_k: Keywords per document
        limit: Limit documents
        model_name: BERT model to use

    Returns:
        DataFrame with keywords

    Example:
        >>> keywords_df = extract_keywords_simple("der_tag", top_k=10)
        >>> print(keywords_df)
    """
    extractor = KeyBERTExtractor(
        source_name=source_name,
        model_name=model_name,
    )
    return extractor.extract_keywords(top_k=top_k, limit=limit)
