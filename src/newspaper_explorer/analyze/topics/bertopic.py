"""
BERTopic-based topic modeling for newspaper texts.

Implements BERTopic to discover semantic topics using transformer embeddings
and extract representative terms from newspaper articles. Unlike LDA which uses
bag-of-words, BERTopic leverages BERT embeddings for semantic understanding.

This is adapted from the optimized standalone script with improvements:
- Integration with unified metadata system
- Foreign key support for linking results
- Configurable chunking and aggregation strategies

Example:
    >>> from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor
    >>> extractor = BERTopicExtractor(source_name="der_tag")
    >>>
    >>> # Extract topics for documents
    >>> doc_topics = extractor.extract_topics(
    ...     group_by=["date"],
    ...     min_topic_size=5,
    ...     limit=1000
    ... )
    >>>
    >>> # Save with metadata
    >>> output_path = extractor.save_results(doc_topics, output_name="bertopic_topics")
"""

import logging
import os
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.models import AnalysisMetadata
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.metadata import save_metadata
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.data.utils.text import chunk_text, split_text_into_sentences

# Set up model cache directory
_MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / "models" / "sentence_transformers"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_MODELS_DIR)

logger = logging.getLogger(__name__)


def _chunk_document_worker(args: Tuple[str, str, int, int]) -> Tuple[str, List[str]]:
    """
    Worker function for parallel document chunking.

    Args:
        args: Tuple of (doc_id, text, chunk_sentences, min_text_length)

    Returns:
        Tuple of (doc_id, list of chunks)
    """
    doc_id, text, chunk_sentences, min_text_length = args

    if len(text) < min_text_length:
        return (doc_id, [])

    try:
        # Split into sentences using spaCy
        sentences = split_text_into_sentences(text, model="de_core_news_sm")
    except (ImportError, OSError):
        # Fallback to simple chunking
        chunk_size = chunk_sentences * 100
        chunks = chunk_text(
            text,
            max_length=chunk_size,
            split_symbols=[" ", ",", ".", ";", ":", "!", "?"],
            split_margin=50,
        )
        return (doc_id, [c for c in chunks if len(c) >= min_text_length])

    # Group sentences into chunks
    chunks = []
    for i in range(0, len(sentences), chunk_sentences):
        chunk = " ".join(sentences[i : i + chunk_sentences])
        if len(chunk) >= min_text_length:
            chunks.append(chunk)

    return (doc_id, chunks)


class BERTopicExtractor:
    """
    Extract topics from newspaper texts using BERTopic.

    BERTopic uses transformer embeddings (BERT) for semantic similarity,
    then clusters documents and extracts representative words using c-TF-IDF.

    Key features:
    - Semantic topic discovery (not just word co-occurrence like LDA)
    - Automatic topic number detection (no need to specify k)
    - Multilingual support via paraphrase-multilingual models
    - Document chunking for better topic granularity

    Workflow:
        1. Load and aggregate documents (by date, page, issue, etc.)
        2. Chunk documents for better topic detection
        3. Generate embeddings and cluster
        4. Extract topic terms using c-TF-IDF
        5. Assign topics back to original documents
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        min_text_length: int = 100,
        chunk_sentences: int = 5,
        use_stopwords: bool = True,
        num_gpus: int = 1,
    ):
        """
        Initialize BERTopic extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            embedding_model: SentenceTransformer model name
            min_text_length: Minimum text length to consider (chars)
            chunk_sentences: Number of sentences per chunk for topic detection
            use_stopwords: Use German stopwords for vectorizer (default: True)
            num_gpus: Number of GPUs to use (1 = single GPU, >1 = multi-GPU, default: 1)
        """
        self.source_name = source_name
        self.text_column = text_column
        self.min_text_length = min_text_length
        self.chunk_sentences = chunk_sentences
        self.num_gpus = num_gpus
        self.config = get_config()

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to textblocks for better coherence
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

        # Get stopwords
        stopwords = None
        if use_stopwords:
            try:
                from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS

                stopwords = list(DE_STOP_WORDS)
                logger.info(f"Using {len(stopwords)} German stopwords")
            except ImportError:
                logger.warning("spaCy not installed, proceeding without stopwords")
                stopwords = None

        # Initialize models
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vectorizer_model = CountVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 2),
            min_df=2,
        )

        logger.info(f"Initialized BERTopic extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for better topic detection.

        Uses spaCy to split into sentences, then groups N sentences per chunk.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        try:
            # Split into sentences using spaCy
            sentences = split_text_into_sentences(text, model="de_core_news_sm")
        except (ImportError, OSError) as e:
            logger.warning(f"Could not use spaCy for sentence splitting: {e}")
            logger.warning("Falling back to simple chunking")
            # Fallback to simple chunking if spaCy not available
            chunk_size = self.chunk_sentences * 100
            chunks = chunk_text(
                text,
                max_length=chunk_size,
                split_symbols=[" ", ",", ".", ";", ":", "!", "?"],
                split_margin=50,
            )
            return [c for c in chunks if len(c) >= self.min_text_length]

        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), self.chunk_sentences):
            chunk = " ".join(sentences[i : i + self.chunk_sentences])
            if len(chunk) >= self.min_text_length:
                chunks.append(chunk)

        return chunks

    def extract_topics(
        self,
        group_by: Optional[List[str]] = None,
        n_topics: Optional[int] = None,
        min_cluster_size: int = 10,
        top_k: int = 5,
        limit: Optional[int] = None,
        batch_size: int = 32,
        umap_sample_pages: Optional[int] = None,
        filter_pages: Optional[List[int]] = None,
        years: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
    ) -> pl.DataFrame:
        """
        Extract topics from documents using global BERTopic.

        **Global Topics Approach**:
        1. Loads all documents and chunks them into sentences
        2. Embeds all chunks in batches (supports multi-GPU)
        3. Fits BERTopic ONCE on entire corpus to discover global topics
        4. Assigns each document to discovered global topics

        This approach enables:
        - Temporal analysis (same topic space across all documents)
        - Fast processing (single fit instead of per-document fits)
        - Better topic quality (learns from entire corpus)
        - Comparable topics across documents

        **Sampling Strategies for Large Corpora**:

        For massive datasets (>10M lines), use sampling to reduce memory:

        1. **Page filtering**: `filter_pages=[1, 2, 3]` - Only process first 3 pages
        2. **Year filtering**: `years=[1900, 1901]` - Only process specific years
        3. **Random sampling**: `sample_size=100000` - Random sample of documents
        4. **UMAP sampling**: `umap_sample_pages=3` - Fit UMAP on subset, transform all
        5. **Combined**: Use multiple strategies together

        **Recommended for 60M+ lines**:
        - Process year-by-year: Call this function per year
        - Use `filter_pages=[1, 2, 3]` for front pages only
        - Use `umap_sample_pages=3` for memory efficiency
        - Use `sample_size` for initial exploration

        Args:
            group_by: Columns to group by (e.g., ["date"], ["page_id"])
            n_topics: Target number of topics (None = automatic discovery)
            min_cluster_size: Minimum chunks per topic cluster
            top_k: Number of topic terms to return per document
            limit: Limit number of input rows (for testing)
            batch_size: Batch size for embedding generation
            umap_sample_pages: If set, fit UMAP on first N pages of each issue
                              (e.g., 3 = first 3 pages). Reduces memory usage
                              by 10-100x while maintaining topic quality.
                              Transforms all embeddings after fit.
            filter_pages: Only process specific page numbers (e.g., [1, 2, 3] for first 3 pages)
                         Filters BEFORE grouping for maximum memory savings.
            years: Only process specific years (e.g., [1900, 1901])
                  Filters BEFORE grouping for maximum memory savings.
            sample_size: Random sample N documents after grouping (for exploration)
            random_seed: Seed for random sampling (default: 42)

        Returns:
            DataFrame with doc_id, topic_terms, scores, topics, topic_probs,
            and foreign keys
        """
        start_time = time.time()

        logger.info("Loading input data...")
        df = pl.read_parquet(self.input_file)

        # Apply filters BEFORE any processing for maximum memory savings
        original_count = len(df)

        # Filter by years
        if years is not None:
            if "year" in df.columns:
                df = df.filter(pl.col("year").is_in(years))
                logger.info(
                    f"Filtered to years {years}: {len(df):,} rows (from {original_count:,})"
                )
            elif "date" in df.columns:
                # Extract year from date column
                df = df.with_columns(pl.col("date").dt.year().alias("year"))
                df = df.filter(pl.col("year").is_in(years))
                logger.info(
                    f"Filtered to years {years}: {len(df):,} rows (from {original_count:,})"
                )
            else:
                logger.warning("No 'year' or 'date' column found - cannot filter by years")

        # Filter by page numbers
        if filter_pages is not None:
            if "page_number" in df.columns:
                df = df.filter(pl.col("page_number").is_in(filter_pages))
                logger.info(
                    f"Filtered to pages {filter_pages}: {len(df):,} rows "
                    f"({len(df) / original_count * 100:.1f}% of original)"
                )
            else:
                logger.warning("No 'page_number' column found - cannot filter by pages")

        if limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} rows")

        # Aggregate by grouping
        if group_by:
            logger.info(f"Aggregating by: {group_by}")
            df = df.with_columns(
                [pl.concat_str([pl.col(c) for c in group_by], separator="_").alias("doc_id")]
            )
            df_grouped = df.group_by(group_by + ["doc_id"]).agg(
                [pl.col(self.text_column).str.join(" ").alias("text")]
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

        # Apply random sampling after grouping (for exploration)
        if sample_size is not None and sample_size < len(df_grouped):
            logger.info(
                f"Random sampling {sample_size:,} documents from {len(df_grouped):,} "
                f"(seed={random_seed})"
            )
            df_grouped = df_grouped.sample(n=sample_size, seed=random_seed)

        logger.info(f"Processing {len(df_grouped)} documents")

        # Phase 1: Chunk all documents in parallel
        logger.info("Phase 1: Chunking all documents in parallel...")

        # Prepare arguments for parallel processing
        chunk_args = [
            (row["doc_id"], row[self.text_column] or "", self.chunk_sentences, self.min_text_length)
            for row in df_grouped.iter_rows(named=True)
        ]

        # Use multiprocessing for parallel chunking
        n_workers = max(1, cpu_count() - 1)  # Leave one core free
        logger.info(f"Using {n_workers} workers for parallel chunking")

        doc_chunks = []  # List of (doc_id, chunk_text)
        doc_metadata = {}  # doc_id -> doc info

        with Pool(processes=n_workers) as pool:
            # Process in chunks with progress bar
            for doc_id, chunks in tqdm(
                pool.imap(_chunk_document_worker, chunk_args, chunksize=100),
                total=len(chunk_args),
                desc="Chunking documents",
            ):
                if len(chunks) == 0:
                    continue

                doc_metadata[doc_id] = {"num_chunks": len(chunks)}
                for chunk in chunks:
                    doc_chunks.append((doc_id, chunk))

        if len(doc_chunks) == 0:
            logger.warning("No chunks generated - all documents too short or invalid")
            return pl.DataFrame()

        logger.info(f"Generated {len(doc_chunks)} chunks from {len(doc_metadata)} documents")

        # Phase 2: Batch embed all chunks
        logger.info("Phase 2: Embedding all chunks...")
        chunk_texts = [chunk for _, chunk in doc_chunks]

        # Multi-GPU encoding if num_gpus > 1
        if self.num_gpus > 1:
            logger.info(f"Using multi-GPU encoding with {self.num_gpus} GPUs")
            # Start multi-process pool for parallel GPU encoding
            target_devices = [f"cuda:{i}" for i in range(self.num_gpus)]
            pool = self.embedding_model.start_multi_process_pool(target_devices=target_devices)
            embeddings = self.embedding_model.encode_multi_process(
                chunk_texts,
                pool=pool,
                batch_size=batch_size,
            )
            self.embedding_model.stop_multi_process_pool(pool)
        else:
            # Single GPU encoding
            embeddings = self.embedding_model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Phase 3: Fit global BERTopic model
        logger.info("Phase 3: Fitting global BERTopic model...")

        # Prepare sampling strategy if requested
        sample_indices = None
        if umap_sample_pages is not None:
            logger.info(f"Using smart sampling: first {umap_sample_pages} pages of each issue")

            # Build mapping: chunk_idx -> doc_id
            chunk_to_doc = [doc_id for doc_id, _ in doc_chunks]

            # Get page_number for each doc_id from original data
            doc_to_page = {}
            if "page_number" in df.columns:
                # Extract page numbers per doc_id
                for row in df.select(["doc_id", "page_number"]).unique().iter_rows(named=True):
                    doc_id = row["doc_id"]
                    page_num = row.get("page_number")
                    if page_num is not None and doc_id not in doc_to_page:
                        doc_to_page[doc_id] = page_num

                # Filter chunks by page number
                sample_indices = [
                    idx
                    for idx, doc_id in enumerate(chunk_to_doc)
                    if doc_to_page.get(doc_id, 999) <= umap_sample_pages
                ]

                logger.info(
                    f"Sampled {len(sample_indices):,} chunks from first {umap_sample_pages} pages "
                    f"({len(sample_indices) / len(embeddings) * 100:.1f}% of {len(embeddings):,} total chunks)"
                )
                logger.info(
                    f"Memory reduction: {len(embeddings) / max(len(sample_indices), 1):.1f}x"
                )
            else:
                logger.warning("No page_number column found - falling back to random sampling")
                sample_size = min(500_000, len(embeddings))
                sample_indices = np.random.choice(
                    len(embeddings), sample_size, replace=False
                ).tolist()
                logger.info(f"Random sampled {len(sample_indices):,} chunks")

        # Fit UMAP on sample if requested, otherwise on all embeddings
        if sample_indices is not None and len(sample_indices) < len(embeddings):
            sample_texts = [chunk_texts[i] for i in sample_indices]
            sample_embeddings = embeddings[sample_indices]

            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=self.vectorizer_model,
                min_topic_size=min_cluster_size,
                nr_topics=n_topics,
                calculate_probabilities=False,
                verbose=True,
            )

            logger.info(f"Fitting UMAP on {len(sample_embeddings):,} sampled embeddings...")
            topics_sample, _ = topic_model.fit_transform(sample_texts, sample_embeddings)

            logger.info(f"Transforming all {len(embeddings):,} embeddings with fitted model...")
            topics, probs = topic_model.transform(chunk_texts, embeddings)
        else:
            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=self.vectorizer_model,
                min_topic_size=min_cluster_size,
                nr_topics=n_topics,
                calculate_probabilities=False,
                verbose=True,
            )

            topics, probs = topic_model.fit_transform(chunk_texts, embeddings)

        logger.info("Global topic model fitted successfully")

        # Get topic info
        topic_info = topic_model.get_topic_info()
        logger.info(f"Discovered {len(topic_info[topic_info['Topic'] != -1])} topics")

        # Phase 4: Assign topics to documents
        logger.info("Phase 4: Assigning topics to documents...")
        results = []

        for doc_id in tqdm(doc_metadata.keys(), desc="Assigning topics"):
            # Find all chunks for this document
            doc_chunk_indices = [i for i, (did, _) in enumerate(doc_chunks) if did == doc_id]
            doc_topics = [topics[i] for i in doc_chunk_indices]

            # Count topic frequencies (excluding -1 outliers)
            topic_counts = {}
            for t in doc_topics:
                if t != -1:
                    topic_counts[t] = topic_counts.get(t, 0) + 1

            if not topic_counts:
                # No valid topics for this document
                continue

            # Sort by frequency
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            top_topics = sorted_topics[:3]  # Top 3 topics

            # Calculate probabilities
            total = sum(count for _, count in top_topics)
            topic_ids = [tid for tid, _ in top_topics]
            topic_probs = [count / total for _, count in top_topics]

            # Get topic terms
            all_terms = []
            all_scores = []

            for topic_id, prob in zip(topic_ids, topic_probs):
                words_scores = topic_model.get_topic(topic_id)
                if words_scores:
                    for word, score in words_scores[:top_k]:
                        all_terms.append(word)
                        all_scores.append(float(prob * score))

            # Deduplicate and sort
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

            results.append(
                {
                    "doc_id": doc_id,
                    "topic_terms": unique_terms,
                    "scores": unique_scores,
                    "topics": topic_ids,
                    "topic_probs": topic_probs,
                }
            )

        logger.info(f"Assigned topics to {len(results)} documents")

        # Create DataFrame
        if not results:
            logger.warning("No results - no documents had valid topics")
            return pl.DataFrame()

        df_results = pl.DataFrame(results)

        # Add foreign keys
        logger.info("Extracting foreign keys...")
        foreign_keys_list = []
        for doc_id in df_results["doc_id"]:
            fks = extract_foreign_keys(doc_id)
            foreign_keys_list.append(fks)

        df_results = df_results.with_columns(
            [
                pl.Series("source_id", [fk.get("source_id") for fk in foreign_keys_list]),
                pl.Series("issue_id", [fk.get("issue_id") for fk in foreign_keys_list]),
                pl.Series("page_id", [fk.get("page_id") for fk in foreign_keys_list]),
                pl.Series("text_block_id", [fk.get("text_block_id") for fk in foreign_keys_list]),
            ]
        )

        # Store for save_results
        self._last_extraction_time = time.time() - start_time
        self._last_input_df = df_grouped
        self._last_topic_model = topic_model
        self._last_params = {
            "group_by": group_by,
            "n_topics": n_topics,
            "min_cluster_size": min_cluster_size,
            "top_k": top_k,
            "limit": limit,
            "batch_size": batch_size,
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "min_text_length": self.min_text_length,
            "chunk_sentences": self.chunk_sentences,
            "total_chunks": len(doc_chunks),
            "total_documents": len(doc_metadata),
        }

        logger.info(f"Extracted topics for {len(df_results)} documents")

        return df_results

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "bertopic_topics",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save results to parquet file with metadata.

        Args:
            results_df: Results DataFrame with topic assignments
            output_name: Output filename (without extension, default: "bertopic_topics")
            top_k: Number of terms (for metadata, optional)

        Returns:
            Path to saved parquet file
        """
        # Create metadata
        params = getattr(self, "_last_params", {})
        if top_k is not None:
            params["top_k"] = top_k

        # Extract input/output statistics
        input_data = {}
        if hasattr(self, "_last_input_df"):
            input_data["num_documents"] = len(getattr(self, "_last_input_df", []))

        output_data = extract_output_stats(results_df)

        # Add topic-specific statistics
        if "topics" in results_df.columns:
            all_topics = []
            for topic_list in results_df["topics"]:
                # topic_list is a Python list, not a Polars Series
                if isinstance(topic_list, list) and topic_list and topic_list != [-1]:
                    all_topics.extend(topic_list)
            output_data["unique_topics_assigned"] = len(set(all_topics))
            output_data["total_topic_assignments"] = len(all_topics)

        metadata = AnalysisMetadata(
            analysis_type="topics",
            method_type="bertopic",
            model_name=f"bertopic_{params.get('embedding_model', 'unknown')}d",
            source=self.source_name,
            parameters=params,
            input_data=input_data,
            output_data=output_data,
            duration_seconds=getattr(self, "_last_extraction_time", None),
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

        logger.info(f"Saved results to {paths['results_path']}")
        logger.info(f"Saved metadata to {paths['metadata_path']}")

        return paths["results_path"]
