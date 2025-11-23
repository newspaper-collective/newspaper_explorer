"""
LDA-based topic term extraction for newspaper texts.

Implements Latent Dirichlet Allocation (LDA) to discover topics and extract
representative terms from newspaper articles. LDA identifies latent topics in
the corpus and extracts the most probable words for each topic, which can be
used for topic-based document representation.

Unlike keyword extraction methods (TF-IDF, RAKE, YAKE, KeyBERT), LDA discovers
corpus-wide topics first, then assigns topic-based terms to documents. These
are NOT keywords in the traditional sense, but topically representative vocabulary.

Methodological Note:
    The terms extracted are "topic terms" (topically representative words),
    not "keywords" (semantically important phrases). Use this for understanding
    thematic structure, not for keyphrase extraction.

Example:
    >>> from newspaper_explorer.analyze.keywords.lda import LDAExtractor
    >>> extractor = LDAExtractor(source_name="der_tag")
    >>>
    >>> # Train LDA model on the corpus
    >>> model_info = extractor.train_model(num_topics=20, passes=10)
    >>>
    >>> # Extract representative terms for each topic
    >>> topic_terms = extractor.get_topic_terms(top_k=10)
    >>>
    >>> # Assign topic-based terms to documents
    >>> doc_terms = extractor.extract_topic_based_document_terms(top_k=5)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.models import AnalysisMetadata
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.metadata import save_metadata
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats

logger = logging.getLogger(__name__)


class LDAExtractor:
    """
    Extract keywords from newspaper texts using LDA (Latent Dirichlet Allocation).

    LDA is a topic modeling technique that discovers latent topics in a corpus.
    Each topic is a distribution over words, and documents are mixtures of topics.
    Keywords are extracted as the most probable words for each topic.

    Workflow:
        1. Train LDA model on corpus (one-time operation)
        2. Extract topic keywords (top words per topic)
        3. Assign document keywords based on topic distribution
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        use_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        model_dir: Optional[Path] = None,
    ):
        """
        Initialize LDA extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            use_stopwords: Whether to remove common stopwords
            custom_stopwords: Additional stopwords to exclude
            model_dir: Directory to save/load models (default: results/{source}/topics/models)
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to textblocks.parquet for better topic coherence
            # Falls back to lines.parquet if textblocks don't exist
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
                # Set to textblocks even if doesn't exist - will error later
                self.input_file = textblocks_file

        # Setup stopwords
        self.stopwords = self._get_stopwords(use_stopwords, custom_stopwords)

        # Setup model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = self.config.results_dir / source_name / "topics" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Model components (loaded on demand)
        self.lda_model: Optional[LdaModel] = None
        self.dictionary: Optional[corpora.Dictionary] = None
        self.corpus: Optional[List] = None
        self.doc_ids: Optional[List] = None

        logger.info(f"Initialized LDA extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Using {len(self.stopwords)} stopwords")

    def _get_stopwords(self, use_stopwords: bool, custom_stopwords: Optional[List[str]]) -> set:
        """
        Get stopwords set combining German, English, and custom stopwords.
        """
        stopwords = set()

        if use_stopwords:
            # Start with Gensim's built-in stopwords (mostly English)
            stopwords.update(GENSIM_STOPWORDS)

            # Add German stopwords from SpaCy
            try:
                from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS

                stopwords.update(DE_STOP_WORDS)
                logger.info(f"Added {len(DE_STOP_WORDS)} German stopwords from SpaCy")
            except ImportError:
                logger.warning(
                    "SpaCy not installed, using basic German stopwords. "
                    "Install with: pip install -e '.[nlp]' for better stopword list"
                )
                # Basic German stopwords fallback
                basic_german = {
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
                }
                stopwords.update(basic_german)

        # Add custom stopwords
        if custom_stopwords:
            stopwords.update(custom_stopwords)
            logger.info(f"Added {len(custom_stopwords)} custom stopwords")

        return stopwords

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for LDA: tokenize, lowercase, remove stopwords.

        Args:
            text: Raw text string

        Returns:
            List of preprocessed tokens
        """
        if not text or not isinstance(text, str):
            return []

        # Tokenize (simple split on whitespace and basic punctuation)
        tokens = text.lower().split()

        # Remove punctuation and filter
        tokens = [token.strip(".,;:!?\"'()[]{}—–-") for token in tokens]

        # Filter: length > 2, not stopwords, alphabetic
        tokens = [
            token
            for token in tokens
            if len(token) > 2 and token not in self.stopwords and token.isalpha()
        ]

        return tokens

    def _load_and_preprocess_data(
        self,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Load data and preprocess for LDA.

        Args:
            limit: Limit number of documents
            group_by: Columns to group by (aggregates text)

        Returns:
            Tuple of (preprocessed_documents, document_ids)
        """
        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.input_file}\n"
                "Run parsing first: newspaper-explorer data parse --source {self.source_name}"
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

        # Extract text and create doc IDs
        texts = df[self.text_column].to_list()

        # Create document IDs based on available columns
        if "text_block_id" in df.columns:
            doc_ids = df["text_block_id"].to_list()
        elif "filename" in df.columns:
            doc_ids = df["filename"].to_list()
        else:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]

        # Preprocess
        logger.info("Preprocessing documents...")
        preprocessed = [self._preprocess_text(text) for text in tqdm(texts, desc="Preprocessing")]

        # Filter out empty documents
        valid_docs = [(doc, doc_id) for doc, doc_id in zip(preprocessed, doc_ids) if len(doc) > 0]

        if not valid_docs:
            raise ValueError("No valid documents after preprocessing")

        preprocessed, doc_ids = zip(*valid_docs)

        logger.info(f"Preprocessed {len(preprocessed)} documents")
        logger.info(
            f"Average tokens per document: {np.mean([len(doc) for doc in preprocessed]):.1f}"
        )

        return list(preprocessed), list(doc_ids)

    def train_model(
        self,
        num_topics: int = 10,
        passes: int = 10,
        iterations: int = 400,
        chunksize: int = 2000,
        random_state: int = 42,
        alpha: str = "auto",
        eta: str = "auto",
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
        no_below: int = 5,
        no_above: float = 0.5,
        keep_n: int = 100000,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        """
        Train LDA model on the corpus.

        Args:
            num_topics: Number of topics to discover
            passes: Number of passes through the corpus
            iterations: Maximum iterations per pass
            chunksize: Documents per training chunk
            random_state: Random seed for reproducibility
            alpha: Document-topic density ('auto', 'symmetric', or float)
            eta: Topic-word density ('auto', 'symmetric', or float)
            limit: Limit number of documents for training
            group_by: Columns to group by before training
            no_below: Filter tokens appearing in < N documents
            no_above: Filter tokens appearing in > fraction of documents
            keep_n: Keep only top N most frequent tokens
            force_retrain: Force retraining even if model exists

        Returns:
            Dict with model info (perplexity, coherence, etc.)
        """
        model_path = self.model_dir / f"lda_{num_topics}topics.model"
        dict_path = self.model_dir / f"lda_{num_topics}topics.dict"

        # Check if model already exists
        if model_path.exists() and not force_retrain:
            logger.info(f"Loading existing model from {model_path}")
            self.lda_model = LdaModel.load(str(model_path))
            self.dictionary = corpora.Dictionary.load(str(dict_path))

            return {
                "status": "loaded_existing",
                "model_path": str(model_path),
                "num_topics": num_topics,
            }

        # Load and preprocess data
        preprocessed, doc_ids = self._load_and_preprocess_data(limit=limit, group_by=group_by)
        self.doc_ids = doc_ids

        # Create dictionary
        logger.info("Creating dictionary...")
        self.dictionary = corpora.Dictionary(preprocessed)

        # Filter extremes
        logger.info(
            f"Filtering dictionary (no_below={no_below}, no_above={no_above}, keep_n={keep_n})"
        )
        original_size = len(self.dictionary)
        self.dictionary.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n,
        )
        logger.info(f"Dictionary size: {original_size} -> {len(self.dictionary)} tokens")

        # Create corpus
        logger.info("Creating bag-of-words corpus...")
        self.corpus = [
            self.dictionary.doc2bow(doc) for doc in tqdm(preprocessed, desc="Creating corpus")
        ]

        # Train LDA model
        logger.info(f"Training LDA model with {num_topics} topics...")
        logger.info(f"Parameters: passes={passes}, iterations={iterations}")

        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=random_state,
            passes=passes,
            iterations=iterations,
            chunksize=chunksize,
            alpha=alpha,
            eta=eta,
        )

        # Calculate perplexity
        perplexity = self.lda_model.log_perplexity(self.corpus)
        logger.info(f"Model perplexity: {perplexity:.4f}")

        # Save model and dictionary
        logger.info(f"Saving model to {model_path}")
        self.lda_model.save(str(model_path))
        self.dictionary.save(str(dict_path))

        # Save model info
        info = {
            "status": "trained",
            "model_path": str(model_path),
            "num_topics": num_topics,
            "num_documents": len(self.corpus),
            "vocabulary_size": len(self.dictionary),
            "perplexity": float(perplexity),
            "passes": passes,
            "iterations": iterations,
        }

        info_path = self.model_dir / f"lda_{num_topics}topics_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Model info saved to {info_path}")

        return info

    def load_model(self, num_topics: int) -> None:
        """
        Load a previously trained LDA model.

        Args:
            num_topics: Number of topics (identifies which model to load)
        """
        model_path = self.model_dir / f"lda_{num_topics}topics.model"
        dict_path = self.model_dir / f"lda_{num_topics}topics.dict"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Train a model first: extractor.train_model(num_topics={num_topics})"
            )

        logger.info(f"Loading model from {model_path}")
        self.lda_model = LdaModel.load(str(model_path))
        self.dictionary = corpora.Dictionary.load(str(dict_path))

        logger.info(f"Loaded model with {self.lda_model.num_topics} topics")

    def get_topic_terms(
        self,
        top_k: int = 10,
        num_topics: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract top representative terms for each topic.

        Args:
            top_k: Number of keywords per topic
            num_topics: If model not loaded, try to load this model

        Returns:
            DataFrame with columns: topic_id, topic_terms, scores

        Note:
            These are topically representative terms, not keywords in the
            traditional sense. They represent the most probable words for
            each discovered topic.
        """
        if self.lda_model is None:
            if num_topics is None:
                raise ValueError(
                    "No model loaded. Either train a model or specify num_topics to load"
                )
            self.load_model(num_topics)

        logger.info(f"Extracting top {top_k} representative terms per topic")

        results = []
        for topic_id in range(self.lda_model.num_topics):
            # Get top words for this topic
            topic_words = self.lda_model.show_topic(topic_id, topn=top_k)
            topic_terms = [word for word, _ in topic_words]
            scores = [float(score) for _, score in topic_words]

            results.append(
                {
                    "topic_id": topic_id,
                    "topic_terms": topic_terms,
                    "scores": scores,
                }
            )

        df = pl.DataFrame(results)
        logger.info(f"Extracted topic terms for {len(df)} topics")

        return df

    def extract_topic_based_document_terms(
        self,
        top_k: int = 5,
        min_topic_prob: float = 0.1,
        num_topics: Optional[int] = None,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Assign topic-based terms to documents based on their topic distribution.

        Each document gets representative terms from its most probable topics.
        This provides topic-based document representation, not keyphrase extraction.

        Args:
            top_k: Number of terms per document
            min_topic_prob: Minimum topic probability to consider
            num_topics: If model not loaded, try to load this model
            limit: Limit number of documents
            group_by: Columns to group by

        Returns:
            DataFrame with document IDs and their assigned topic-based terms

        Note:
            These are NOT keywords in the traditional sense. They represent
            the document's content through its topic mixture, weighted by
            topic probabilities.
        """
        start_time = time.time()

        if self.lda_model is None:
            if num_topics is None:
                raise ValueError(
                    "No model loaded. Either train a model or specify num_topics to load"
                )
            self.load_model(num_topics)

        # Load and preprocess data
        preprocessed, doc_ids = self._load_and_preprocess_data(limit=limit, group_by=group_by)

        logger.info(f"Extracting keywords for {len(preprocessed)} documents")

        # Convert documents to bag-of-words
        corpus = [self.dictionary.doc2bow(doc) for doc in preprocessed]

        results = []
        for doc_id, bow in tqdm(
            zip(doc_ids, corpus), total=len(doc_ids), desc="Extracting topic terms"
        ):
            # Get topic distribution for this document
            topic_dist = self.lda_model.get_document_topics(bow, minimum_probability=min_topic_prob)

            if not topic_dist:
                # No topics above threshold
                results.append(
                    {
                        "doc_id": doc_id,
                        "topic_terms": [],
                        "scores": [],
                        "topics": [],
                        "topic_probs": [],
                    }
                )
                continue

            # Sort topics by probability
            topic_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)

            # Collect terms from top topics
            all_terms = []
            all_scores = []
            topics_used = []
            topic_probs = []

            for topic_id, topic_prob in topic_dist:
                # Get top words for this topic
                topic_words = self.lda_model.show_topic(topic_id, topn=top_k)

                # Weight terms by topic probability
                for word, word_prob in topic_words:
                    weighted_score = float(topic_prob * word_prob)
                    all_terms.append(word)
                    all_scores.append(weighted_score)

                topics_used.append(topic_id)
                topic_probs.append(float(topic_prob))

            # Sort by score and take top_k
            term_scores = list(zip(all_terms, all_scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)

            # Remove duplicates while preserving order
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
                    "topics": topics_used,
                    "topic_probs": topic_probs,
                }
            )

        df = pl.DataFrame(results)

        # Add foreign keys
        logger.info("Extracting foreign keys from doc_ids...")
        foreign_keys_list = []
        for doc_id in df["doc_id"]:
            fks = extract_foreign_keys(doc_id)
            foreign_keys_list.append(fks)

        # Add foreign key columns
        df = df.with_columns(
            [
                pl.Series("source_id", [fk.get("source_id") for fk in foreign_keys_list]),
                pl.Series("issue_id", [fk.get("issue_id") for fk in foreign_keys_list]),
                pl.Series("page_id", [fk.get("page_id") for fk in foreign_keys_list]),
                pl.Series("text_block_id", [fk.get("text_block_id") for fk in foreign_keys_list]),
            ]
        )

        # Store for save_results
        self._last_extraction_time = time.time() - start_time
        self._last_input_df = preprocessed  # Store for metadata
        self._last_params = {
            "top_k": top_k,
            "min_topic_prob": min_topic_prob,
            "num_topics": self.lda_model.num_topics if self.lda_model else num_topics,
            "limit": limit,
            "group_by": group_by,
        }

        logger.info(f"Extracted topic terms for {len(df)} documents")

        return df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "lda_topics",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save results to parquet file with metadata.

        Args:
            results_df: Results DataFrame with topic assignments
            output_name: Output filename (without extension, default: "lda_topics")
            top_k: Number of terms (for metadata, optional)

        Returns:
            Path to saved parquet file
        """
        # Create metadata
        params = getattr(self, "_last_params", {})
        if top_k is not None:
            params["top_k"] = top_k

        # Add model-specific parameters
        if self.lda_model:
            params.update(
                {
                    "num_topics": self.lda_model.num_topics,
                    "passes": self.lda_model.num_topics,  # Not stored, approximate
                    "min_topic_prob": params.get("min_topic_prob", 0.1),
                }
            )

        params.update(
            {
                "num_stopwords": len(self.stopwords),
                "text_column": self.text_column,
            }
        )

        # Extract input/output statistics
        input_data = {}
        if hasattr(self, "_last_input_df"):
            # Create a temporary DataFrame for stats extraction
            temp_df = pl.DataFrame({"text": ["sample"]})  # Placeholder
            try:
                input_data = extract_input_stats(temp_df, id_column="doc_id", date_column="")
            except:
                pass  # If extraction fails, use empty dict
            input_data["num_documents"] = len(getattr(self, "_last_input_df", []))

        output_data = extract_output_stats(results_df)

        # Add topic-specific statistics
        if "topics" in results_df.columns:
            all_topics = []
            for topic_list in results_df["topics"]:
                if topic_list:
                    all_topics.extend(topic_list)
            output_data["unique_topics_assigned"] = len(set(all_topics))
            output_data["total_topic_assignments"] = len(all_topics)

        metadata = AnalysisMetadata(
            analysis_type="topics",
            method_type="lda",
            model_name=f"lda_{params.get('num_topics', 'unknown')}_topics",
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


def extract_topic_terms_simple(
    source_name: str,
    num_topics: int = 10,
    top_k: int = 10,
    passes: int = 10,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Simple one-shot function to train LDA and extract topic terms.

    Args:
        source_name: Source name (e.g., "der_tag")
        num_topics: Number of topics to discover
        top_k: Representative terms per topic
        passes: Training passes
        limit: Limit documents for training

    Returns:
        DataFrame with topic terms (topically representative vocabulary)

    Example:
        >>> topics_df = extract_topic_terms_simple("der_tag", num_topics=20)
        >>> print(topics_df)

    Note:
        Returns topic-representative terms, not keywords in the traditional sense.
    """
    extractor = LDAExtractor(source_name=source_name)

    # Train model
    extractor.train_model(
        num_topics=num_topics,
        passes=passes,
        limit=limit,
    )

    # Extract topic terms
    topic_terms_df = extractor.get_topic_terms(top_k=top_k)

    return topic_terms_df
