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

import contextlib
from datetime import datetime
import logging
import multiprocessing as mp
from multiprocessing import Barrier, Process, Queue
from multiprocessing.queues import Queue as QueueType
from multiprocessing.synchronize import Barrier as BarrierType
import os
from pathlib import Path
import time
from typing import Optional

from keybert import KeyBERT
import polars as pl
from sentence_transformers import SentenceTransformer
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
import torch
from tqdm import tqdm

from newspaper_explorer.config import external_tools  # noqa: F401 (sets SENTENCE_TRANSFORMERS_HOME)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.metadata import save_metadata
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.models.data.metadata import AnalysisMetadata

logger = logging.getLogger(__name__)


def _extract_keywords_on_gpu(
    device: str,
    texts: list[str],
    doc_ids: list[str],
    model_name: str,
    keyphrase_ngram_range: tuple[int, int],
    stopwords: Optional[list[str]],
    top_k: int,
    diversity: float,
    *,
    use_mmr: bool,
    max_seq_length: int,
    batch_size: int = 32,
    result_queue: Optional[QueueType] = None,
    barrier: Optional[BarrierType] = None,
    gpu_id: Optional[int] = None,
) -> Optional[list[dict]]:
    """
    Worker function to extract keywords on a specific GPU device.

    Can be used in two modes:
    1. Direct call (single-GPU): result_queue=None, barrier=None - returns results
    2. Spawned process (multi-GPU): result_queue and barrier provided - sends to queue

    Args:
        device: Device string (e.g., "cuda:0", "cuda", "cpu")
        texts: List of texts to process
        doc_ids: Document IDs for tracking
        model_name: Sentence transformer model name
        keyphrase_ngram_range: N-gram range for keywords
        stopwords: List of stopwords to exclude
        top_k: Number of keywords per document
        diversity: MMR diversity parameter
        use_mmr: Whether to use MMR
        max_seq_length: Max sequence length for tokenization
        batch_size: Batch size for embedding generation (default: 32)
        result_queue: Optional multiprocessing queue for results (multi-GPU mode)
        barrier: Optional multiprocessing barrier for synchronization (multi-GPU mode)
        gpu_id: Optional GPU ID for progress bar labeling (multi-GPU mode)

    Returns:
        List of result dictionaries if result_queue is None, else None
    """
    # In multi-GPU mode, suppress output during model loading
    if gpu_id is not None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Load models
    sentence_model = SentenceTransformer(model_name, device=device)
    sentence_model.max_seq_length = max_seq_length

    # Note: torch.compile is applied per-GPU worker in multi-GPU mode if needed
    # For now, we skip compilation in worker processes to avoid overhead

    model = KeyBERT(model=sentence_model)  # type: ignore[arg-type]

    # Wait for all GPUs to finish loading (multi-GPU mode only)
    if barrier is not None:
        barrier.wait()

    # Prepare stopwords
    stopwords_param = stopwords if stopwords is not None else []

    # Process texts in batches for better GPU utilization
    results = []

    # Configure progress bar
    if gpu_id is not None:
        pbar_desc = f"GPU {gpu_id}"
        pbar_position = gpu_id
    else:
        pbar_desc = "Extracting keywords"
        pbar_position = 0

    # Process in batches - KeyBERT supports batch processing!
    for batch_start in tqdm(
        range(0, len(texts), batch_size),
        desc=pbar_desc,
        position=pbar_position,
        leave=True,
    ):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_ids = doc_ids[batch_start:batch_end]

        # Filter out invalid texts
        valid_batch = []
        valid_ids = []
        invalid_indices = []
        for idx, (doc_id, text) in enumerate(zip(batch_ids, batch_texts)):
            if text and isinstance(text, str) and len(text.strip()) >= 20:
                valid_batch.append(text)
                valid_ids.append(doc_id)
            else:
                # Track invalid texts to add empty results later
                invalid_indices.append((idx, doc_id))

        if not valid_batch:
            # All texts in batch were invalid
            for _, doc_id in invalid_indices:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )
            continue

        try:
            # Extract keywords for entire batch at once (true batch processing!)
            if use_mmr and diversity > 0:
                batch_extracted = model.extract_keywords(
                    valid_batch,  # Pass list of documents for batch processing
                    keyphrase_ngram_range=keyphrase_ngram_range,
                    stop_words=stopwords_param,
                    top_n=top_k,
                    use_mmr=True,
                    diversity=diversity,
                )
            else:
                batch_extracted = model.extract_keywords(
                    valid_batch,  # Pass list of documents for batch processing
                    keyphrase_ngram_range=keyphrase_ngram_range,
                    stop_words=stopwords_param,
                    top_n=top_k,
                )

            # Process batch results
            for doc_id, extracted in zip(valid_ids, batch_extracted):
                if extracted and isinstance(extracted, list):
                    keywords = [kw for kw, score in extracted]
                    scores = [score for kw, score in extracted]
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

            # Add empty results for invalid texts
            for _, doc_id in invalid_indices:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )

        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"Error processing batch: {e}")
            # Add empty results for failed batch
            for doc_id in valid_ids:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )
            for _, doc_id in invalid_indices:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )

    # Return or send to queue
    if result_queue is not None:
        result_queue.put(results)
        return None

    return results


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
        keyphrase_ngram_range: tuple[int, int] = (1, 2),
        *,
        use_stopwords: bool = True,
        custom_stopwords: Optional[list[str]] = None,
        diversity: float = 0.5,
        batch_size: int = 32,
        max_seq_length: int = 512,
        device: Optional[str] = None,
        use_chunking: bool = True,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        use_multi_gpu: Optional[bool] = None,
        compile_model: bool = False,
    ) -> None:
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
            batch_size: Batch size for processing documents (default: 32)
            max_seq_length: Maximum sequence length for BERT (default: 512)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_chunking: Split long texts into chunks to avoid truncation (default: True)
            chunk_size: Token size for each chunk (default: 400, leaves room for special tokens)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
            use_multi_gpu: Use multiple GPUs if available (None=auto-detect, True=force, False=disable)
            compile_model: Use torch.compile for model optimization (default: False, requires PyTorch 2.0+)
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()
        self.model_name = model_name
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.diversity = diversity
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.compile_model = compile_model
        self.use_multi_gpu = use_multi_gpu

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

        # Setup stopwords - KeyBERT needs List[str], not set
        stopwords_list = self._get_stopwords(use_stopwords, custom_stopwords)
        self.stopwords = stopwords_list  # Keep as List[str] | None, not set

        # Detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Log configuration
        logger.info(f"Loading KeyBERT with model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Max sequence length: {max_seq_length}")
        if use_chunking:
            logger.info(f"Chunking enabled: chunk_size={chunk_size}, overlap={chunk_overlap}")

        # Initialize sentence model for chunking
        self.sentence_model: Optional[SentenceTransformer]
        if use_chunking:
            self.sentence_model = SentenceTransformer(model_name, device=self.device)
            self.sentence_model.max_seq_length = max_seq_length

            # Apply torch.compile if requested (PyTorch 2.0+)
            if (
                compile_model
                and hasattr(torch, "compile")
                and hasattr(self.sentence_model, "__getitem__")
            ):
                try:
                    logger.info("Compiling model with torch.compile for optimization...")
                    # Access the transformer model and compile it
                    transformer_model = self.sentence_model[0]
                    if hasattr(transformer_model, "auto_model"):
                        compiled = torch.compile(  # type: ignore
                            transformer_model.auto_model, mode="reduce-overhead", fullgraph=False
                        )
                        transformer_model.auto_model = compiled
                        logger.info("Model compilation successful")
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    logger.warning(
                        f"Model compilation failed: {e}. Continuing without compilation."
                    )
            elif compile_model:
                logger.warning(
                    "torch.compile not available (requires PyTorch 2.0+). Continuing without compilation."
                )
        else:
            self.sentence_model = None

        # Check for multi-GPU (auto-enable if multiple GPUs available)
        if self.device == "cuda":
            gpu_count = torch.cuda.device_count()
            logger.info(f"Primary GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPUs: {gpu_count}")

            # Auto-detect multi-GPU if use_multi_gpu is None
            if use_multi_gpu is None:
                use_multi_gpu = gpu_count > 1
                if use_multi_gpu:
                    logger.info(f"Multi-GPU auto-enabled: will use {gpu_count} GPUs")

            if gpu_count > 1 and use_multi_gpu:
                logger.info(f"Multi-GPU enabled: will use {gpu_count} GPUs")
                self.num_gpus = gpu_count
            elif gpu_count > 1 and not use_multi_gpu:
                logger.info("Multi-GPU available but disabled (use --use-multi-gpu to enable)")
                self.num_gpus = 1
            else:
                self.num_gpus = 1
        else:
            self.num_gpus = 1

        logger.info(f"Initialized KeyBERT extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"N-gram range: {keyphrase_ngram_range}")
        logger.info(f"Diversity: {diversity}")
        if self.stopwords:
            logger.info(f"Using {len(self.stopwords)} stopwords")

    def _get_stopwords(
        self, *, use_stopwords: bool, custom_stopwords: Optional[list[str]]
    ) -> Optional[list[str]]:
        """Get German stopwords list."""
        if not use_stopwords:
            return None

        # Convert set[LiteralString] to list[str] explicitly
        stopwords: list[str] = list(DE_STOP_WORDS)
        logger.info(f"Loaded {len(stopwords)} German stopwords from SpaCy")

        # Add custom stopwords
        if custom_stopwords:
            stopwords.extend(custom_stopwords)
            logger.info(f"Added {len(custom_stopwords)} custom stopwords")

        return stopwords

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split long text into overlapping chunks to avoid truncation.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        if not self.use_chunking or not self.sentence_model:
            return [text]

        # Tokenize text
        tokens = self.sentence_model.tokenizer.tokenize(text)

        # If text fits in one chunk, return as-is
        if len(tokens) <= self.chunk_size:
            return [text]

        # Create overlapping chunks
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # Convert tokens back to text
            chunk_text = self.sentence_model.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)

            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap

        logger.debug(f"Split text into {len(chunks)} chunks (original: {len(tokens)} tokens)")
        return chunks

    def _merge_keywords(
        self, keyword_lists: list[list[tuple[str, float]]], top_k: int
    ) -> list[tuple[str, float]]:
        """
        Merge keywords from multiple chunks.

        For each keyword, takes the maximum score across chunks and returns top_k.

        Args:
            keyword_lists: List of keyword lists from different chunks
            top_k: Number of top keywords to return

        Returns:
            Merged and deduplicated list of (keyword, score) tuples
        """
        # Aggregate scores for each keyword (take maximum)
        keyword_scores: dict[str, float] = {}
        for kw_list in keyword_lists:
            for keyword, score in kw_list:
                if keyword in keyword_scores:
                    keyword_scores[keyword] = max(keyword_scores[keyword], score)
                else:
                    keyword_scores[keyword] = score

        # Sort by score and return top_k
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]

    def extract_keywords(
        self,
        top_k: int = 10,
        limit: Optional[int] = None,
        group_by: Optional[list[str]] = None,
        *,
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
            DataFrame with columns: doc_id, source_id, issue_id, page_id, text_block_id, keywords, scores

        Note:
            Scores are cosine similarity (0-1, higher = more relevant)
        """
        start_time = time.time()

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

        # Auto-aggregate by page if no grouping specified and using textblocks
        if group_by is None and "page_id" in df.columns:
            logger.info("Auto-aggregating text blocks by page for keyword extraction")
            group_by = ["page_id"]

        # Group if requested - Fix deprecation warning
        if group_by:
            logger.info(f"Grouping by: {', '.join(group_by)}")
            df = df.group_by(group_by).agg(
                pl.col(self.text_column).str.join(" ")  # Changed from str.concat
            )

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

        # Multi-GPU processing
        if self.num_gpus > 1:
            logger.info(f"Distributing {len(texts)} documents across {self.num_gpus} GPUs")

            # Set start method to 'spawn' for CUDA compatibility
            with contextlib.suppress(RuntimeError):
                mp.set_start_method("spawn", force=True)

            # Create barrier for synchronization
            barrier = Barrier(self.num_gpus)

            # Split documents across GPUs
            docs_per_gpu = (len(texts) + self.num_gpus - 1) // self.num_gpus
            processes: list[Process] = []
            result_queue: Queue[list[dict[str, object]]] = Queue()

            for gpu_id in range(self.num_gpus):
                start_idx = gpu_id * docs_per_gpu
                end_idx = min(start_idx + docs_per_gpu, len(texts))

                if start_idx >= len(texts):
                    break

                gpu_texts = texts[start_idx:end_idx]
                gpu_doc_ids = doc_ids[start_idx:end_idx]

                logger.info(f"GPU {gpu_id}: will process {len(gpu_texts)} documents")

                p = Process(
                    target=_extract_keywords_on_gpu,
                    kwargs={
                        "device": f"cuda:{gpu_id}",
                        "texts": gpu_texts,
                        "doc_ids": gpu_doc_ids,
                        "model_name": self.model_name,
                        "keyphrase_ngram_range": self.keyphrase_ngram_range,
                        "stopwords": self.stopwords,
                        "top_k": top_k,
                        "diversity": self.diversity,
                        "use_mmr": use_mmr,
                        "max_seq_length": self.max_seq_length,
                        "batch_size": self.batch_size,
                        "result_queue": result_queue,
                        "barrier": barrier,
                        "gpu_id": gpu_id,
                    },
                )
                p.start()
                processes.append(p)

            # Collect results from all GPUs
            logger.info("All GPUs are loading models and will start processing together...")
            logger.info("Waiting for GPU processes to complete...")
            results = []
            for _ in range(len(processes)):
                results.extend(result_queue.get())

            # Wait for all processes to finish
            for p in processes:
                p.join()

            logger.info("Multi-GPU processing complete")

        else:
            # Single GPU/CPU processing
            single_gpu_results = _extract_keywords_on_gpu(
                device=self.device,
                texts=texts,
                doc_ids=doc_ids,
                model_name=self.model_name,
                keyphrase_ngram_range=self.keyphrase_ngram_range,
                stopwords=self.stopwords,
                top_k=top_k,
                diversity=self.diversity,
                use_mmr=use_mmr,
                max_seq_length=self.max_seq_length,
                batch_size=self.batch_size,
                result_queue=None,  # Single-GPU mode
                barrier=None,
                gpu_id=None,
            )

            # Type safety: single-GPU mode must return results
            if single_gpu_results is None:
                logger.error("Unexpected None result from single-GPU processing")
                raise RuntimeError("Single-GPU processing returned no results")

            results = single_gpu_results

        # Convert to DataFrame
        results_df = pl.DataFrame(results)

        # Extract foreign keys from doc_ids
        logger.info("Extracting foreign keys from doc_ids...")
        doc_ids_list = results_df["doc_id"].to_list()
        foreign_keys = [extract_foreign_keys(doc_id) for doc_id in doc_ids_list]

        # Add foreign key columns
        results_df = results_df.with_columns(
            [
                pl.Series("source_id", [fk.source_id for fk in foreign_keys]),
                pl.Series("issue_id", [fk.issue_id for fk in foreign_keys]),
                pl.Series("page_id", [fk.page_id for fk in foreign_keys]),
                pl.Series(
                    "text_block_id",
                    [fk.text_block_id or "" for fk in foreign_keys],
                ),
            ]
        )

        # Output structure: doc_id, source_id, issue_id, page_id, text_block_id,
        # keywords, scores
        # Note: KeyBERT doesn't support grouping columns yet

        # Store timing info for save_results
        self._last_extraction_time = time.time() - start_time
        self._last_input_df = df

        logger.info(f"Extracted keywords for {len(results_df)} documents")

        return results_df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "keybert_keywords",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save results to parquet file with metadata.

        Args:
            results_df: Results DataFrame
            output_name: Output filename (without extension, default: "keywords")
            top_k: Number of keywords (for metadata, optional)

        Returns:
            Path to saved file
        """
        logger.info("Creating metadata...")

        # Calculate statistics
        total_keywords = sum(len(kw_list) for kw_list in results_df["keywords"].to_list())
        avg_keywords = total_keywords / len(results_df) if len(results_df) > 0 else 0
        avg_score = (
            sum(
                sum(scores) / len(scores) if scores else 0
                for scores in results_df["scores"].to_list()
            )
            / len(results_df)
            if len(results_df) > 0
            else 0
        )

        # Create output statistics
        output_stats = extract_output_stats(results_df)
        output_stats.update(
            {
                "total_documents": len(results_df),
                "total_keywords": total_keywords,
                "avg_keywords_per_doc": round(avg_keywords, 2),
                "avg_score": round(avg_score, 4),
            }
        )

        # Get duration if available from last extraction
        duration = getattr(self, "_last_extraction_time", None)

        # Get input stats if available
        input_stats = {}
        if hasattr(self, "_last_input_df"):
            input_stats = extract_input_stats(self._last_input_df)

        # Create metadata with properly formatted timestamps
        completed_at = datetime.now().isoformat()

        metadata = AnalysisMetadata(
            analysis_id=None,  # Will be auto-generated
            analysis_type="keywords",
            method_type="keybert",
            model_name=self.model_name.replace("/", "_").replace("-", "_"),
            model_version="unknown",  # sentence-transformers version could be extracted if needed
            source=self.source_name,
            parameters={
                "model_name": self.model_name,
                "keyphrase_ngram_range": list(self.keyphrase_ngram_range),
                "diversity": self.diversity,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
                "use_chunking": self.use_chunking,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "use_multi_gpu": self.use_multi_gpu if self.use_multi_gpu is not None else False,
                "num_gpus": self.num_gpus if self.use_multi_gpu else 1,
                "compile_model": self.compile_model,
                "device": self.device,
                "top_k": top_k,
                "text_column": self.text_column,
                "use_stopwords": self.stopwords is not None,
            },
            input_data=input_stats,
            output_data=output_stats,
            granularity="textblock",  # KeyBERT keyword extraction runs on textblock level
            status="completed",
            duration_seconds=duration,
            completed_at=completed_at,
            error_message=None,
        )

        # Save using unified helper
        results_base = self.config.results_dir / self.source_name
        paths = save_analysis_results(
            results_df=results_df,
            metadata=metadata,
            results_base_dir=results_base,
            results_filename="keywords.parquet",
        )

        logger.info(f"Saved results to {paths['results_path']}")
        logger.info(f"Metadata saved to: {paths['metadata_path']}")
        logger.info(f"Analysis ID: {metadata.analysis_id}")

        return paths["results_path"]


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
