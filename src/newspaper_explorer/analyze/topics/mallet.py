"""
MALLET-based topic modeling for newspaper texts.

MALLET (MAchine Learning for LanguagE Toolkit) provides highly optimized LDA
via Gibbs sampling. It generally produces higher quality topics than Gensim's
variational Bayes implementation, especially for historical texts.

Requirements:
    - Java Runtime Environment (JRE) installed
    - MALLET binary (auto-downloaded or manually installed)

The wrapper handles:
    - Automatic MALLET installation
    - Data format conversion (text → MALLET format)
    - Model training with optimized hyperparameters
    - Topic extraction and document assignment
    - Integration with newspaper_explorer metadata system

Example:
    >>> from newspaper_explorer.analyze.topics.mallet import MALLETExtractor
    >>> extractor = MALLETExtractor(source_name="der_tag")
    >>>
    >>> # Train MALLET model on the corpus
    >>> model_info = extractor.train_model(num_topics=20, iterations=1000)
    >>>
    >>> # Extract representative terms for each topic
    >>> topic_terms = extractor.get_topic_terms(top_k=10)
    >>>
    >>> # Assign topic-based terms to documents
    >>> doc_topics = extractor.extract_document_topics()
"""

import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve
import zipfile

import numpy as np
import polars as pl
from tqdm import tqdm

from newspaper_explorer.cli.utils.options import resolve_text_column
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.metadata import save_metadata
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.models.data.metadata import AnalysisMetadata

logger = logging.getLogger(__name__)

# MALLET download URL (official release)
MALLET_VERSION = "2.0.8"
MALLET_URL = f"https://mallet.cs.umass.edu/dist/mallet-{MALLET_VERSION}.zip"


class MALLETNotFoundError(Exception):
    """Raised when MALLET installation cannot be found or downloaded."""

    pass


class MALLETExtractor:
    """
    Extract topics from newspaper texts using MALLET's optimized LDA.

    MALLET uses Gibbs sampling for LDA inference, which often produces
    more coherent topics than variational methods. It also supports
    automatic hyperparameter optimization.

    Key features:
        - Optimized Gibbs sampling (faster, better convergence)
        - Hyperparameter optimization (alpha, beta)
        - Parallel training support
        - Better handling of short documents
        - Superior topic coherence for historical texts

    Workflow:
        1. Install/locate MALLET (one-time)
        2. Export texts to MALLET format
        3. Train LDA model
        4. Extract topics and document assignments
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        mallet_path: Optional[Path] = None,
        use_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        model_dir: Optional[Path] = None,
    ):
        """
        Initialize MALLET extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            mallet_path: Path to MALLET installation (auto-detected if None)
            use_stopwords: Whether to use stopwords
            custom_stopwords: Additional stopwords to exclude
            model_dir: Directory to save models (default: results/{source}/topics/models)
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            source_dir = self.config.parsed_dir / source_name
            textblocks_file = source_dir / "textblocks.parquet"
            lines_file = source_dir / "lines.parquet"

            if textblocks_file.exists():
                self.input_file = textblocks_file
                logger.info("Using textblocks.parquet (aggregated text blocks)")
            elif lines_file.exists():
                self.input_file = lines_file
                logger.info("Using lines.parquet (line-level data)")
            else:
                self.input_file = textblocks_file

        # Auto-resolve text column (prefer text_processed if available)
        self.text_column = resolve_text_column(
            self.text_column, file_path=str(self.input_file)
        )

        # Setup model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = self.config.results_dir / source_name / "topics" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Setup stopwords
        self.stopwords = self._get_stopwords(use_stopwords, custom_stopwords)
        self._stopwords_file: Optional[Path] = None

        # Find or install MALLET
        self.mallet_path = self._find_or_install_mallet(mallet_path)

        # Model components (populated after training)
        self._doc_ids: Optional[List[str]] = None
        self._topic_keys_file: Optional[Path] = None
        self._doc_topics_file: Optional[Path] = None
        self._state_file: Optional[Path] = None

        logger.info(f"Initialized MALLET extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"MALLET path: {self.mallet_path}")
        logger.info(f"Using {len(self.stopwords)} stopwords")

    def _get_stopwords(self, use_stopwords: bool, custom_stopwords: Optional[List[str]]) -> set:
        """Get combined stopwords set."""
        stopwords = set()

        if use_stopwords:
            # Try SpaCy German stopwords
            try:
                from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS

                stopwords.update(DE_STOP_WORDS)
                logger.info(f"Added {len(DE_STOP_WORDS)} German stopwords from SpaCy")
            except ImportError:
                logger.warning("SpaCy not installed, using basic German stopwords")
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
                    "es",
                    "er",
                    "aber",
                    "so",
                    "wenn",
                    "was",
                    "wie",
                    "ich",
                    "man",
                    "da",
                    "doch",
                    "dann",
                    "schon",
                    "hier",
                    "mehr",
                    "sehr",
                    "haben",
                    "sein",
                    "seine",
                    "seinen",
                    "seiner",
                    "diese",
                    "dieser",
                    "dieses",
                }
                stopwords.update(basic_german)

            # Add common English stopwords too
            english_basic = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
            }
            stopwords.update(english_basic)

        if custom_stopwords:
            stopwords.update(custom_stopwords)
            logger.info(f"Added {len(custom_stopwords)} custom stopwords")

        return stopwords

    def _find_or_install_mallet(self, mallet_path: Optional[Path] = None) -> Path:
        """
        Find existing MALLET installation or download it.

        Args:
            mallet_path: Optional path to existing MALLET installation

        Returns:
            Path to MALLET bin directory
        """
        # Check if provided path is valid
        if mallet_path:
            mallet_bin = Path(mallet_path) / "bin" / "mallet"
            if mallet_bin.exists() or (Path(mallet_path) / "bin" / "mallet.bat").exists():
                return Path(mallet_path)
            else:
                logger.warning(f"MALLET not found at {mallet_path}, searching elsewhere...")

        # Check common locations
        possible_paths = [
            Path.home() / "mallet" / f"mallet-{MALLET_VERSION}",
            Path.home() / ".mallet" / f"mallet-{MALLET_VERSION}",
            Path("/usr/local/mallet"),
            Path("/opt/mallet"),
            self.config.data_dir.parent / "tools" / f"mallet-{MALLET_VERSION}",
        ]

        # Also check MALLET_HOME environment variable
        mallet_home = os.environ.get("MALLET_HOME")
        if mallet_home:
            possible_paths.insert(0, Path(mallet_home))

        for path in possible_paths:
            mallet_bin = path / "bin" / "mallet"
            if mallet_bin.exists():
                logger.info(f"Found MALLET at {path}")
                return path

        # Not found, attempt installation
        logger.info("MALLET not found, attempting automatic installation...")
        return self._install_mallet()

    def _install_mallet(self) -> Path:
        """
        Download and install MALLET.

        Returns:
            Path to installed MALLET directory
        """
        # Check for Java
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise MALLETNotFoundError(
                    "Java is required but not found. Install Java JRE/JDK first:\n"
                    "  Ubuntu/Debian: sudo apt install default-jre\n"
                    "  macOS: brew install openjdk\n"
                    "  Windows: Download from https://adoptium.net/"
                )
        except FileNotFoundError:
            raise MALLETNotFoundError("Java is required but not found. Install Java JRE/JDK first.")

        # Install to tools directory
        install_dir = self.config.data_dir.parent / "tools"
        install_dir.mkdir(parents=True, exist_ok=True)

        zip_path = install_dir / f"mallet-{MALLET_VERSION}.zip"
        mallet_dir = install_dir / f"mallet-{MALLET_VERSION}"

        # Download if not exists
        if not zip_path.exists():
            logger.info(f"Downloading MALLET from {MALLET_URL}...")
            try:

                def _progress_hook(block_num, block_size, total_size):
                    if total_size > 0:
                        downloaded = block_num * block_size
                        percent = min(100, downloaded * 100 / total_size)
                        print(f"\rDownloading: {percent:.1f}%", end="", flush=True)

                urlretrieve(MALLET_URL, zip_path, reporthook=_progress_hook)
                print()  # Newline after progress
            except Exception as e:
                raise MALLETNotFoundError(
                    f"Failed to download MALLET: {e}\n"
                    f"Please download manually from {MALLET_URL} "
                    f"and extract to {install_dir}"
                )

        # Extract if not exists
        if not mallet_dir.exists():
            logger.info(f"Extracting MALLET to {install_dir}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(install_dir)

        # Verify installation
        mallet_bin = mallet_dir / "bin" / "mallet"
        if not mallet_bin.exists():
            raise MALLETNotFoundError(
                f"MALLET installation failed. Expected binary at {mallet_bin}"
            )

        # Make executable on Unix
        if platform.system() != "Windows":
            os.chmod(mallet_bin, 0o755)

        logger.info(f"MALLET installed successfully at {mallet_dir}")
        return mallet_dir

    def _get_mallet_command(self) -> str:
        """Get the MALLET executable command."""
        if platform.system() == "Windows":
            return str(self.mallet_path / "bin" / "mallet.bat")
        else:
            return str(self.mallet_path / "bin" / "mallet")

    def _get_java_command(self, class_name: str, memory_gb: int = 8) -> List[str]:
        """
        Build Java command with proper memory settings.

        Args:
            class_name: MALLET Java class to run
            memory_gb: Memory in GB

        Returns:
            List of command parts for subprocess
        """
        classpath = f"{self.mallet_path}/class:{self.mallet_path}/lib/mallet-deps.jar"
        return [
            "java",
            f"-Xmx{memory_gb}g",
            "-ea",
            "-Djava.awt.headless=true",
            "-Dfile.encoding=UTF-8",
            "-server",
            "-classpath",
            classpath,
            class_name,
        ]

    def _write_stopwords_file(self) -> Path:
        """Write stopwords to a file for MALLET."""
        if self._stopwords_file and self._stopwords_file.exists():
            return self._stopwords_file

        self._stopwords_file = self.model_dir / "stopwords.txt"
        with open(self._stopwords_file, "w", encoding="utf-8") as f:
            for word in sorted(self.stopwords):
                f.write(f"{word}\n")

        logger.info(f"Wrote {len(self.stopwords)} stopwords to {self._stopwords_file}")
        return self._stopwords_file

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for MALLET (minimal - MALLET does its own tokenization).

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning only - MALLET handles tokenization
        # Remove excessive whitespace
        text = " ".join(text.split())

        return text

    def _export_to_mallet_format(
        self,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
    ) -> Tuple[Path, List[str]]:
        """
        Export data to MALLET input format.

        MALLET expects one document per line:
        doc_id [tab] label [tab] text

        Args:
            limit: Limit number of documents
            group_by: Columns to group by (aggregates text)

        Returns:
            Tuple of (mallet_input_file, doc_ids)
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
            # Create group ID from group columns
            df = df.with_columns(
                pl.concat_str([pl.col(c).cast(str) for c in group_by], separator="_").alias(
                    "_group_id"
                )
            )
            df = df.group_by(group_by + ["_group_id"]).agg(pl.col(self.text_column).str.concat(" "))
            id_column = "_group_id"
        elif "text_block_id" in df.columns:
            id_column = "text_block_id"
        elif "filename" in df.columns:
            id_column = "filename"
        else:
            df = df.with_columns(pl.arange(0, len(df)).cast(str).alias("_doc_id"))
            id_column = "_doc_id"

        # Export to MALLET format
        output_file = self.model_dir / "input.txt"
        doc_ids = []

        logger.info("Exporting to MALLET format...")
        with open(output_file, "w", encoding="utf-8") as f:
            for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Exporting"):
                doc_id = str(row[id_column])
                text = self._preprocess_text(row[self.text_column])

                if text:  # Skip empty documents
                    # Format: doc_id [tab] label [tab] text
                    # Using doc_id as label too since we don't have separate labels
                    f.write(f"{doc_id}\t{doc_id}\t{text}\n")
                    doc_ids.append(doc_id)

        logger.info(f"Exported {len(doc_ids)} documents to {output_file}")
        return output_file, doc_ids

    def train_model(
        self,
        num_topics: int = 20,
        iterations: int = 1000,
        optimize_interval: int = 10,
        num_threads: int = 4,
        alpha: float = 5.0,
        beta: float = 0.01,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
        force_retrain: bool = False,
        random_seed: int = 42,
        memory_gb: int = 8,
    ) -> Dict[str, Any]:
        """
        Train MALLET LDA model on the corpus.

        Args:
            num_topics: Number of topics to discover
            iterations: Number of Gibbs sampling iterations
            optimize_interval: Iterations between hyperparameter optimization (0 to disable)
            num_threads: Number of parallel threads
            alpha: Initial alpha hyperparameter (document-topic prior)
            beta: Beta hyperparameter (topic-word prior)
            limit: Limit number of documents for training
            group_by: Columns to group by before training
            force_retrain: Force retraining even if model exists
            random_seed: Random seed for reproducibility
            memory_gb: Java heap memory in GB (default: 8)

        Returns:
            Dict with model info and file paths
        """
        model_prefix = f"mallet_{num_topics}topics"
        self._topic_keys_file = self.model_dir / f"{model_prefix}_keys.txt"
        self._doc_topics_file = self.model_dir / f"{model_prefix}_doc_topics.txt"
        self._state_file = self.model_dir / f"{model_prefix}_state.gz"
        mallet_file = self.model_dir / f"{model_prefix}.mallet"

        # Check if model already exists
        if self._topic_keys_file.exists() and not force_retrain:
            logger.info(f"Loading existing model from {self.model_dir}")
            return {
                "status": "loaded_existing",
                "model_prefix": model_prefix,
                "topic_keys_file": str(self._topic_keys_file),
                "doc_topics_file": str(self._doc_topics_file),
                "num_topics": num_topics,
            }

        start_time = time.time()

        # Export data to MALLET format
        input_file, doc_ids = self._export_to_mallet_format(limit=limit, group_by=group_by)
        self._doc_ids = doc_ids

        # Write stopwords file
        stopwords_file = self._write_stopwords_file()

        logger.info(f"Using Java heap memory: {memory_gb}GB")

        # Step 1: Import data into MALLET format
        logger.info("Importing data into MALLET format...")
        import_cmd = self._get_java_command(
            "cc.mallet.classify.tui.Csv2Vectors", memory_gb=memory_gb
        ) + [
            "--input",
            str(input_file),
            "--output",
            str(mallet_file),
            "--keep-sequence",
            "--remove-stopwords",
            "--extra-stopwords",
            str(stopwords_file),
            "--token-regex",
            r"[\p{L}\p{N}][\p{L}\p{N}]+",  # Unicode letters/numbers
        ]

        result = subprocess.run(import_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MALLET import failed: {result.stderr}")

        logger.info(f"Created MALLET file: {mallet_file}")

        # Step 2: Train LDA model
        logger.info(f"Training LDA model with {num_topics} topics...")
        logger.info(f"Parameters: iterations={iterations}, threads={num_threads}")

        train_cmd = self._get_java_command(
            "cc.mallet.topics.tui.TopicTrainer", memory_gb=memory_gb
        ) + [
            "--input",
            str(mallet_file),
            "--num-topics",
            str(num_topics),
            "--num-iterations",
            str(iterations),
            "--num-threads",
            str(num_threads),
            "--output-state",
            str(self._state_file),
            "--output-topic-keys",
            str(self._topic_keys_file),
            "--output-doc-topics",
            str(self._doc_topics_file),
            "--alpha",
            str(alpha),
            "--beta",
            str(beta),
            "--random-seed",
            str(random_seed),
        ]

        if optimize_interval > 0:
            train_cmd.extend(["--optimize-interval", str(optimize_interval)])

        # Run training with progress output
        logger.info("Starting MALLET training (this may take a while)...")
        result = subprocess.run(train_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"MALLET training failed: {result.stderr}")

        # Log any output
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-5:]:  # Last 5 lines
                logger.info(f"MALLET: {line}")

        training_time = time.time() - start_time

        # Save doc IDs mapping
        doc_ids_file = self.model_dir / f"{model_prefix}_doc_ids.json"
        with open(doc_ids_file, "w") as f:
            json.dump(doc_ids, f)

        # Save model info
        info = {
            "status": "trained",
            "model_prefix": model_prefix,
            "topic_keys_file": str(self._topic_keys_file),
            "doc_topics_file": str(self._doc_topics_file),
            "state_file": str(self._state_file),
            "num_topics": num_topics,
            "num_documents": len(doc_ids),
            "iterations": iterations,
            "optimize_interval": optimize_interval,
            "alpha": alpha,
            "beta": beta,
            "training_time_seconds": training_time,
        }

        info_file = self.model_dir / f"{model_prefix}_info.json"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Model training complete in {training_time:.1f}s")
        logger.info(f"Model info saved to {info_file}")

        return info

    def load_model(self, num_topics: int) -> Dict[str, Any]:
        """
        Load a previously trained MALLET model.

        Args:
            num_topics: Number of topics (identifies which model to load)

        Returns:
            Dict with model info
        """
        model_prefix = f"mallet_{num_topics}topics"
        self._topic_keys_file = self.model_dir / f"{model_prefix}_keys.txt"
        self._doc_topics_file = self.model_dir / f"{model_prefix}_doc_topics.txt"
        self._state_file = self.model_dir / f"{model_prefix}_state.gz"

        if not self._topic_keys_file.exists():
            raise FileNotFoundError(
                f"Model not found: {self._topic_keys_file}\n"
                f"Train a model first with: extractor.train_model(num_topics={num_topics})"
            )

        # Load doc IDs
        doc_ids_file = self.model_dir / f"{model_prefix}_doc_ids.json"
        if doc_ids_file.exists():
            with open(doc_ids_file) as f:
                self._doc_ids = json.load(f)

        # Load model info
        info_file = self.model_dir / f"{model_prefix}_info.json"
        if info_file.exists():
            with open(info_file) as f:
                return json.load(f)

        return {
            "status": "loaded",
            "num_topics": num_topics,
            "model_prefix": model_prefix,
        }

    def get_topic_terms(
        self,
        top_k: int = 10,
        num_topics: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract top representative terms for each topic.

        Args:
            top_k: Number of keywords per topic
            num_topics: If model not loaded, load this model

        Returns:
            DataFrame with columns: topic_id, topic_terms, scores
        """
        if self._topic_keys_file is None or not self._topic_keys_file.exists():
            if num_topics is None:
                raise ValueError(
                    "No model loaded. Either train a model or specify num_topics to load"
                )
            self.load_model(num_topics)

        logger.info(f"Reading topic terms from {self._topic_keys_file}")

        results = []

        # Parse MALLET topic keys file
        # Format: topic_id [tab] alpha [tab] word1 word2 word3 ...
        with open(self._topic_keys_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    topic_id = int(parts[0])
                    # Handle locale-specific decimal separators (comma vs dot)
                    alpha = float(parts[1].replace(",", "."))
                    words = parts[2].split()[:top_k]

                    # MALLET doesn't provide word probabilities in keys file,
                    # so we'll use decreasing scores based on rank
                    scores = [1.0 / (i + 1) for i in range(len(words))]

                    results.append(
                        {
                            "topic_id": topic_id,
                            "topic_terms": words,
                            "scores": scores,
                            "alpha": alpha,
                        }
                    )

        df = pl.DataFrame(results)
        logger.info(f"Extracted topic terms for {len(df)} topics")

        return df

    def extract_document_topics(
        self,
        top_k: int = 5,
        min_topic_prob: float = 0.1,
        num_topics: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract topic assignments for each document.

        Args:
            top_k: Number of topic terms per document
            min_topic_prob: Minimum topic probability to include
            num_topics: If model not loaded, load this model

        Returns:
            DataFrame with document IDs and their assigned topics
        """
        start_time = time.time()

        if self._doc_topics_file is None or not self._doc_topics_file.exists():
            if num_topics is None:
                raise ValueError(
                    "No model loaded. Either train a model or specify num_topics to load"
                )
            self.load_model(num_topics)

        logger.info(f"Reading document topics from {self._doc_topics_file}")

        # Load topic terms for enriching results
        topic_terms_df = self.get_topic_terms(top_k=top_k, num_topics=num_topics)
        topic_terms_map = {
            row["topic_id"]: row["topic_terms"] for row in topic_terms_df.iter_rows(named=True)
        }

        results = []

        # Parse MALLET doc-topics file
        # Format: doc_idx [tab] doc_id [tab] topic1_prob topic2_prob ...
        with open(self._doc_topics_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Parsing document topics"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    doc_idx = int(parts[0])
                    doc_id = parts[1]
                    probs = [float(p) for p in parts[2:]]

                    # Get top topics above threshold
                    topic_probs = [(i, p) for i, p in enumerate(probs) if p >= min_topic_prob]
                    topic_probs.sort(key=lambda x: x[1], reverse=True)

                    if topic_probs:
                        topics = [t for t, _ in topic_probs]
                        probs_filtered = [p for _, p in topic_probs]

                        # Collect terms from top topics
                        all_terms = []
                        for topic_id, _ in topic_probs[:3]:  # Top 3 topics
                            terms = topic_terms_map.get(topic_id, [])
                            all_terms.extend(terms[:top_k])

                        # Remove duplicates while preserving order
                        seen = set()
                        unique_terms = []
                        for term in all_terms:
                            if term not in seen:
                                seen.add(term)
                                unique_terms.append(term)
                                if len(unique_terms) >= top_k:
                                    break

                        results.append(
                            {
                                "doc_id": doc_id,
                                "topic_terms": unique_terms,
                                "topics": topics,
                                "topic_probs": probs_filtered,
                            }
                        )
                    else:
                        results.append(
                            {
                                "doc_id": doc_id,
                                "topic_terms": [],
                                "topics": [],
                                "topic_probs": [],
                            }
                        )

        df = pl.DataFrame(results)

        # Add foreign keys
        logger.info("Extracting foreign keys from doc_ids...")
        foreign_keys_list = [extract_foreign_keys(doc_id) for doc_id in df["doc_id"]]

        df = df.with_columns(
            [
                pl.Series("source_id", [fk.source_id for fk in foreign_keys_list]),
                pl.Series("issue_id", [fk.issue_id for fk in foreign_keys_list]),
                pl.Series("page_id", [fk.page_id for fk in foreign_keys_list]),
                pl.Series("text_block_id", [fk.text_block_id for fk in foreign_keys_list]),
            ]
        )

        # Store for save_results
        self._last_extraction_time = time.time() - start_time
        self._last_params = {
            "top_k": top_k,
            "min_topic_prob": min_topic_prob,
            "num_topics": num_topics,
        }

        logger.info(f"Extracted topics for {len(df)} documents")

        return df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "mallet_topics",
    ) -> Path:
        """
        Save results to parquet file with metadata.

        Args:
            results_df: Results DataFrame with topic assignments
            output_name: Output filename (without extension)

        Returns:
            Path to saved parquet file
        """
        params = getattr(self, "_last_params", {})
        params.update(
            {
                "num_stopwords": len(self.stopwords),
                "text_column": self.text_column,
                "mallet_version": MALLET_VERSION,
            }
        )

        # Extract statistics
        output_data = extract_output_stats(results_df)

        if "topics" in results_df.columns:
            all_topics = []
            for topic_list in results_df["topics"]:
                if topic_list:
                    all_topics.extend(topic_list)
            output_data["unique_topics_assigned"] = len(set(all_topics))
            output_data["total_topic_assignments"] = len(all_topics)

        metadata = AnalysisMetadata(
            analysis_type="topics",
            method_type="mallet",
            model_name=f"mallet_{params.get('num_topics', 'unknown')}_topics",
            source=self.source_name,
            parameters=params,
            input_data={"num_documents": len(results_df)},
            output_data=output_data,
            granularity="textblock",  # Topic modeling runs on textblock level
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


def extract_topics_mallet(
    source_name: str,
    num_topics: int = 20,
    top_k: int = 10,
    iterations: int = 1000,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Simple one-shot function to train MALLET LDA and extract document topics.

    Args:
        source_name: Source name (e.g., "der_tag")
        num_topics: Number of topics to discover
        top_k: Terms per document
        iterations: Training iterations
        limit: Limit documents for training

    Returns:
        DataFrame with document topic assignments

    Example:
        >>> doc_topics = extract_topics_mallet("der_tag", num_topics=20)
        >>> print(doc_topics)
    """
    extractor = MALLETExtractor(source_name=source_name)

    # Train model
    extractor.train_model(
        num_topics=num_topics,
        iterations=iterations,
        limit=limit,
    )

    # Extract document topics
    return extractor.extract_document_topics(top_k=top_k, num_topics=num_topics)
