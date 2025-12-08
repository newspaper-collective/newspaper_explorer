"""
FASTopic-based topic modeling for newspaper texts.

Implements FASTopic (Fast, Adaptive, Stable, and Transferable Topic Model)
to discover semantic topics using optimal transport between document, topic,
and word embeddings from pretrained Transformers.

FASTopic advantages over BERTopic:
- Faster training and inference
- More stable topic assignments
- Better transferability across domains
- Adaptive topic discovery

Example:
    >>> from newspaper_explorer.analyze.topics.fastopic import FASTopicExtractor
    >>> extractor = FASTopicExtractor(source_name="der_tag")
    >>>
    >>> # Extract topics for documents
    >>> doc_topics = extractor.extract_topics(
    ...     group_by=["date"],
    ...     n_topics=50,
    ...     limit=1000
    ... )
    >>>
    >>> # Save with metadata
    >>> output_path = extractor.save_results(doc_topics, output_name="fastopic_topics")
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

from newspaper_explorer.config import external_tools  # noqa: F401 (sets SENTENCE_TRANSFORMERS_HOME)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.metadata import save_metadata
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.models.data.metadata import AnalysisMetadata

logger = logging.getLogger(__name__)


class FASTopicExtractor:
    """
    Extract topics from newspaper texts using FASTopic.

    FASTopic uses optimal transport between document, topic, and word embeddings
    from pretrained Transformers to model topics efficiently and stably.

    Key features:
    - Fast training and inference (faster than BERTopic)
    - Stable topic assignments across runs
    - Adaptive topic discovery
    - Multilingual support via sentence transformers
    - No clustering required (optimal transport approach)

    Workflow:
        1. Load and aggregate documents (by date, page, issue, etc.)
        2. Preprocess texts (tokenization, stopword removal)
        3. Train FASTopic model with optimal transport
        4. Extract topic terms and document-topic distributions
        5. Assign topics to documents
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        doc_embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        vocab_size: int = 10000,
        min_text_length: int = 100,
        use_stopwords: bool = True,
    ):
        """
        Initialize FASTopic extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            doc_embed_model: SentenceTransformer model for document embeddings
            vocab_size: Maximum vocabulary size for preprocessing
            min_text_length: Minimum text length to consider (chars)
            use_stopwords: Use German stopwords for preprocessing (default: True)
        """
        self.source_name = source_name
        self.text_column = text_column
        self.min_text_length = min_text_length
        self.vocab_size = vocab_size
        self.use_stopwords = use_stopwords
        self.doc_embed_model_name = doc_embed_model

        # Get config
        config = get_config()

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Try textblocks first, fall back to lines
            textblocks_file = (
                config.data_dir / "processed" / source_name / "text" / "textblocks.parquet"
            )
            lines_file = (
                config.data_dir / "raw" / source_name / "text" / f"{source_name}_lines.parquet"
            )

            if textblocks_file.exists():
                self.input_file = textblocks_file
                logger.info(f"Using textblocks: {self.input_file}")
            elif lines_file.exists():
                self.input_file = lines_file
                logger.info(f"Using lines: {self.input_file}")
            else:
                raise FileNotFoundError(
                    f"No input file found. Tried:\n"
                    f"  - {textblocks_file}\n"
                    f"  - {lines_file}\n"
                    f"Run 'newspaper-explorer data parse' or 'newspaper-explorer data aggregate' first"
                )

        # Results directory
        self.results_dir = config.results_dir / source_name / "topics"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stopwords
        if use_stopwords:
            try:
                import spacy

                nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
                self.stopwords = list(nlp.Defaults.stop_words)
            except (ImportError, OSError):
                logger.warning(
                    "spaCy German model not found. Install with: python -m spacy download de_core_news_sm"
                )
                self.stopwords = None
        else:
            self.stopwords = None

        # Store config
        self.config = config

        # Store metadata
        self._last_input_df = None
        self._last_params = None
        self._last_extraction_time = None
        self._last_model = None

        logger.info(f"Initialized FASTopic extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")

    def extract_topics(
        self,
        group_by: Optional[List[str]] = None,
        n_topics: int = 50,
        top_k: int = 10,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract topics from documents using FASTopic.

        Args:
            group_by: Columns to group by (e.g., ["date"], ["page_id"])
            n_topics: Number of topics to discover
            top_k: Number of topic terms to return per document
            limit: Limit number of input rows (for testing)

        Returns:
            DataFrame with doc_id, topic_terms, scores, topics, topic_probs,
            and foreign keys
        """
        start_time = time.time()

        logger.info("Loading input data...")
        df = pl.read_parquet(self.input_file)

        if limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} rows")

        # Extract foreign keys from IDs first (for grouping)
        # If textblocks, extract page_id, issue_id etc. from text_block_id
        if "text_block_id" in df.columns and not all(
            col in df.columns for col in ["page_id", "issue_id"]
        ):
            logger.info("Extracting foreign keys from text_block_id...")

            # Extract keys for all rows
            foreign_keys_list = []
            for text_block_id in df["text_block_id"]:
                fks = extract_foreign_keys(text_block_id)
                foreign_keys_list.append(fks)

            # Add as columns
            df = df.with_columns(
                [
                    pl.Series("source_id", [fk.source_id for fk in foreign_keys_list]),
                    pl.Series("issue_id", [fk.issue_id for fk in foreign_keys_list]),
                    pl.Series("page_id", [fk.page_id for fk in foreign_keys_list]),
                ]
            )

        # Aggregate by grouping
        if group_by:
            logger.info(f"Aggregating by: {group_by}")
            df = df.with_columns(
                [pl.concat_str([pl.col(c) for c in group_by], separator="_").alias("doc_id")]
            )
            df_grouped = df.group_by(group_by + ["doc_id"]).agg(
                [pl.col(self.text_column).str.join(" ").alias(self.text_column)]
            )
        else:
            if "page_id" in df.columns:
                df = df.with_columns([pl.col("page_id").alias("doc_id")])
            elif "text_block_id" in df.columns:
                df = df.with_columns([pl.col("text_block_id").alias("doc_id")])
            else:
                df = df.with_columns(
                    [pl.lit("doc_").add(pl.int_range(pl.len()).cast(str)).alias("doc_id")]
                )
            df_grouped = df.select(["doc_id", self.text_column])

        logger.info(f"Processing {len(df_grouped)} documents")

        # Filter by text length
        df_filtered = df_grouped.filter(
            pl.col(self.text_column).str.len_chars() >= self.min_text_length
        )
        logger.info(
            f"After filtering: {len(df_filtered)} documents (min_length={self.min_text_length})"
        )

        if len(df_filtered) == 0:
            logger.error("No documents left after filtering!")
            return pl.DataFrame()

        # Extract texts
        docs = df_filtered[self.text_column].to_list()
        doc_ids = df_filtered["doc_id"].to_list()

        # Import FASTopic and preprocessing tools
        try:
            from fastopic import FASTopic
            from topmost import Preprocess
        except ImportError:
            logger.error(
                "FASTopic not installed. Install with: pip install fastopic\n"
                "For other languages, also install spacy language models: python -m spacy download de_core_news_sm"
            )
            raise

        # Setup tokenizer for German
        if self.use_stopwords and self.stopwords:
            try:
                import spacy

                nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
                tokenizer = lambda x: [token.text for token in nlp(x) if not token.is_stop]
                logger.info("Using German tokenizer with stopwords")
            except (ImportError, OSError):
                logger.warning("German spaCy model not available, using default tokenizer")
                tokenizer = None
        else:
            tokenizer = None

        # Initialize preprocessing
        logger.info(f"Preprocessing with vocab_size={self.vocab_size}")
        preprocess = Preprocess(vocab_size=self.vocab_size, tokenizer=tokenizer)

        # Initialize FASTopic model
        logger.info(f"Initializing FASTopic with {n_topics} topics")
        logger.info(f"Document embedding model: {self.doc_embed_model_name}")

        model = FASTopic(
            n_topics,
            preprocess,
            doc_embed_model=self.doc_embed_model_name,
            verbose=True,
        )

        # Train model and get topic assignments
        logger.info("Training FASTopic model...")
        top_words, train_theta = model.fit_transform(docs)

        logger.info("FASTopic training complete!")
        logger.info(f"Discovered {n_topics} topics")

        # Build results
        logger.info("Building results DataFrame...")
        results = []

        for idx, (doc_id, theta) in enumerate(zip(doc_ids, train_theta)):
            # Get top topics for this document (highest probability topics)
            top_topic_indices = np.argsort(theta)[::-1][:3]  # Top 3 topics
            top_topic_probs = theta[top_topic_indices]

            # Get topic terms across all top topics
            all_terms = []
            all_scores = []

            for topic_idx, prob in zip(top_topic_indices, top_topic_probs):
                topic_words = model.get_topic(topic_idx)
                if topic_words:
                    # Get words and scores from this topic
                    for word, score in topic_words[:top_k]:
                        all_terms.append(word)
                        all_scores.append(float(prob * score))  # Weight by topic probability

            # Deduplicate and sort by score
            term_scores = list(zip(all_terms, all_scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)

            seen = set()
            unique_terms = []
            unique_scores = []
            for term, score in term_scores:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)
                    unique_scores.append(score)
                    if len(unique_terms) >= top_k:
                        break

            result = {
                "doc_id": doc_id,
                "topic_terms": unique_terms,
                "scores": unique_scores,
                "topics": top_topic_indices.tolist(),
                "topic_probs": top_topic_probs.tolist(),
            }
            results.append(result)

        logger.info(f"Assigned topics to {len(results)} documents")

        # Create DataFrame
        if not results:
            logger.warning("No results - no valid documents")
            return pl.DataFrame()

        results_df = pl.DataFrame(results)

        # Add foreign keys
        logger.info("Extracting foreign keys...")
        foreign_keys_list = []
        for doc_id in results_df["doc_id"]:
            fks = extract_foreign_keys(doc_id)
            foreign_keys_list.append(fks)

        results_df = results_df.with_columns(
            [
                pl.Series("source_id", [fk.source_id for fk in foreign_keys_list]),
                pl.Series("issue_id", [fk.issue_id for fk in foreign_keys_list]),
                pl.Series("page_id", [fk.page_id for fk in foreign_keys_list]),
                pl.Series("text_block_id", [fk.text_block_id for fk in foreign_keys_list]),
            ]
        )

        # Store metadata
        self._last_extraction_time = time.time() - start_time
        self._last_input_df = df_grouped
        self._last_model = model
        self._last_params = {
            "group_by": group_by,
            "n_topics": n_topics,
            "top_k": top_k,
            "limit": limit,
            "vocab_size": self.vocab_size,
            "doc_embed_model": self.doc_embed_model_name,
            "min_text_length": self.min_text_length,
            "use_stopwords": self.use_stopwords,
            "total_documents": len(df_grouped),
        }

        logger.info(
            f"[OK] Extracted topics for {len(results_df)} documents in {self._last_extraction_time:.2f}s"
        )

        return results_df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "fastopic_topics",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save topic modeling results with metadata.

        Args:
            results_df: Results DataFrame from extract_topics()
            output_name: Output filename without extension
            top_k: Optional override for top_k in metadata

        Returns:
            Path to saved parquet file
        """
        if self._last_input_df is None or self._last_params is None:
            raise ValueError("No extraction results to save. Run extract_topics() first.")

        # Create metadata
        params = self._last_params.copy()
        if top_k is not None:
            params["top_k"] = top_k

        # Extract input/output statistics
        input_data = {"num_documents": len(self._last_input_df)}
        output_data = extract_output_stats(results_df)

        # Add topic-specific statistics
        if "topics" in results_df.columns:
            all_topics = []
            for topic_list in results_df["topics"]:
                if isinstance(topic_list, list) and topic_list:
                    all_topics.extend(topic_list)
            output_data["unique_topics_assigned"] = len(set(all_topics))
            output_data["total_topic_assignments"] = len(all_topics)

        metadata = AnalysisMetadata(
            analysis_type="topics",
            method_type="fastopic",
            model_name=f"fastopic_{params.get('n_topics', 50)}topics",
            source=self.source_name,
            parameters=params,
            input_data=input_data,
            output_data=output_data,
            granularity="textblock",  # FASTopic runs on textblock level
            duration_seconds=self._last_extraction_time,
            status="completed",
        )

        # Save using unified helper
        results_base = self.config.results_dir / self.source_name
        paths = save_analysis_results(
            results_df=results_df,
            metadata=metadata,
            results_base_dir=results_base,
            results_filename="topics.parquet",
        )

        logger.info(f"[OK] Saved results to: {paths['results_path']}")
        logger.info(f"[OK] Metadata saved to: {paths['metadata_path']}")

        return paths["results_path"]
