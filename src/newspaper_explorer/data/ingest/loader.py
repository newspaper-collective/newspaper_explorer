"""
Main DataIngester class for ALTO XML newspaper files.

High-performance data ingester using multiprocessing and Polars for fast processing.
Configuration-driven: loads source metadata from JSON files.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

from natsort import natsorted
import polars as pl
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.ingest.workers import parse_file_worker, parse_mets_worker
from newspaper_explorer.data.parser.mets import METSParser
from newspaper_explorer.data.utils.sources import get_source_paths, load_source_config
from newspaper_explorer.models.data.sources import SourceConfig

logger = logging.getLogger(__name__)


class DataIngester:
    """
    High-performance data ingester for ALTO XML newspaper files.
    Uses multiprocessing and Polars for fast processing.

    Configuration-driven: loads source metadata from JSON files.

    Output DataFrame Schema:
        The loader produces line-level data with the following columns:

        Primary Key:
            line_id (str): Unique line identifier
                Format: {source}_{date}_{issue}_{daily}_{page}_{block}_{line}
                Example: "3074409-X_1902-09-05_415_2_005_TB_1_TL_1"

        Foreign Keys (for linking and querying):
            source_id (str): Source identifier (e.g., "3074409-X" for ZDB, or "der_tag")
            issue_id (str): Issue identifier
                Format: {source}_{date}_{issue}_{daily}
                Example: "3074409-X_1902-09-05_415_2"
            page_id (str): Page identifier
                Format: {issue_id}_{page}
                Example: "3074409-X_1902-09-05_415_2_005"
            text_block_id (str): Text block identifier
                Format: {page_id}_{block_id}
                Example: "3074409-X_1902-09-05_415_2_005_TB_1"

        Data Columns:
            text (str): OCR text content
            filename (str): Source ALTO XML filename
            date (datetime): Publication date
            x, y, width, height (int): Layout coordinates

        Metadata (denormalized for convenience):
            issue_number (int): Issue number from filename
            edition (int): Daily issue number (2nd issue that day)
            page_number (int): Page number
            year_volume (str): Year volume from METS (e.g., "Jahrgang 1902")
            page_count (int): Total pages in issue from METS
            newspaper_title (str): Newspaper title from METS (e.g., "Der Tag")

    Note: For source configuration utilities (list sources, load config, get paths),
    import directly from newspaper_explorer.utils.sources module.

    Example:
        >>> from newspaper_explorer.data.ingest.loader import DataIngester
        >>> ingester = DataIngester(source_name="der_tag")
        >>> df = ingester.load_source()
        >>> # Filter by issue
        >>> issue_df = df.filter(pl.col("issue_id") == "3074409-X_1902-09-05_415_2")
        >>> # Filter by source
        >>> source_df = df.filter(pl.col("source_id") == "3074409-X")
    """

    def __init__(self, source_name: Optional[str] = None, max_workers: Optional[int] = None):
        """
        Initialize DataLoader.

        Args:
            source_name: Name of the source to load (e.g., 'der_tag').
                        If provided, loads configuration from sources/{source_name}.json
            max_workers: Number of parallel workers (default: CPU count - 1)
        """
        if max_workers is None:
            max_workers = max(1, cpu_count() - 1)
        self.max_workers = max_workers

        # Load configuration if source specified
        self.source_name = source_name
        self.config_data: Optional[SourceConfig] = None

        if source_name:
            self.config_data = load_source_config(source_name)
            logger.debug(f"Loaded configuration for source: {source_name}")

        logger.debug(f"DataLoader initialized with {max_workers} workers")

    def get_loading_status(self) -> Dict[str, Any]:
        """
        Get loading status for the configured source.

        Returns:
            Dictionary with status information:
            - xml_files_count: Number of XML files found
            - parquet_exists: Whether output parquet exists
            - parquet_rows: Number of rows in parquet (if exists)
            - parquet_files: Number of unique files in parquet
            - date_range: Date range in parquet (if exists)

        Raises:
            RuntimeError: If no source is configured
        """
        if not self.source_name or not self.config_data:
            raise RuntimeError("No source configured. Initialize with source_name parameter.")

        paths = get_source_paths(self.config_data)
        raw_dir = paths["raw_dir"]
        output_file = paths["output_file"]

        # Get loading config
        pattern = (
            self.config_data.loading.pattern
            if self.config_data.loading
            else get_config().default_alto_pattern
        )

        status: Dict[str, Any] = {
            "source_name": self.source_name,
            "raw_dir": str(raw_dir),
            "output_file": str(output_file),
        }

        # Count XML files
        if raw_dir.exists():
            xml_files = natsorted(raw_dir.glob(pattern))
            status["xml_files_count"] = len(xml_files)
        else:
            status["xml_files_count"] = 0
            status["raw_dir_exists"] = False

        # Check parquet status
        status["parquet_exists"] = output_file.exists()

        if output_file.exists():
            try:
                df = pl.read_parquet(output_file)
                status["parquet_rows"] = len(df)
                status["parquet_files"] = df["filename"].n_unique()

                if "date" in df.columns and len(df) > 0:
                    min_date = df["date"].min()
                    max_date = df["date"].max()

                    if isinstance(min_date, datetime) and isinstance(max_date, datetime):
                        status["date_range"] = f"{min_date} to {max_date}"

                # Size info
                size_mb = output_file.stat().st_size / (1024 * 1024)
                status["parquet_size_mb"] = round(size_mb, 1)
            except Exception as e:
                logger.warning(f"Error reading parquet: {e}")
                status["parquet_error"] = str(e)

        return status

    def load_source(
        self,
        max_files: Optional[int] = None,
        skip_processed: bool = True,
    ) -> pl.DataFrame:
        """
        Load the configured source.

        Args:
            max_files: Maximum number of files to process (for testing)
            skip_processed: If True, skip already processed files

        Returns:
            Polars DataFrame with line-level data

        Raises:
            RuntimeError: If no source is configured
        """
        if not self.source_name or not self.config_data:
            raise RuntimeError(
                "No source configured. Initialize DataLoader with source_name parameter."
            )

        paths = get_source_paths(self.config_data)
        raw_dir = paths["raw_dir"]
        output_file = paths["output_file"]

        # Get loading config
        pattern = (
            self.config_data.loading.pattern
            if self.config_data.loading
            else get_config().default_alto_pattern
        )

        logger.debug(f"Loading source: {self.source_name}")
        logger.debug(f"Raw directory: {raw_dir}")
        logger.debug(f"Output file: {output_file}")

        return self.load_directory(
            directory=raw_dir,
            pattern=pattern,
            max_files=max_files,
            output_parquet=output_file,
            auto_save=True,
            skip_processed=skip_processed,
        )

    def load_directory(
        self,
        directory: Path,
        pattern: Optional[str] = None,
        max_files: Optional[int] = None,
        output_parquet: Optional[Path] = None,
        auto_save: bool = True,
        skip_processed: bool = True,
    ) -> pl.DataFrame:
        """
        Load all ALTO XML files from directory in parallel.

        Args:
            directory: Directory containing ALTO XML files (e.g., data/raw/der_tag/xml_ocr)
            pattern: Glob pattern for finding files (default: from config)
            max_files: Maximum number of files to process (for testing)
            output_parquet: If provided, save result to this specific path
            auto_save: If True, automatically save to data/raw/{source}/text/{source}_lines.parquet
            skip_processed: If True, load existing parquet and skip already processed files

        Returns:
            Polars DataFrame with line-level data
            (includes existing + new data if skip_processed=True)
        """
        if pattern is None:
            pattern = get_config().default_alto_pattern

        logger.info(f"📂 Scanning for ALTO XML files in {directory}")

        # Find all ALTO XML files (only in fulltext directories)
        xml_files = natsorted(directory.glob(pattern))
        logger.info(f"Found {len(xml_files):,} ALTO XML files")

        if max_files:
            xml_files = xml_files[:max_files]
            logger.info(f"Limiting to {max_files:,} files (test mode)")

        # Determine save path early (needed for resume functionality)
        source_name = self._extract_source_name(directory)
        save_path = None
        if output_parquet:
            save_path = output_parquet
        elif auto_save and source_name:
            # Use fixed filename: {source}_lines.parquet
            save_path = directory.parent / "text" / f"{source_name}_lines.parquet"

        # Load existing parquet and skip already processed files
        existing_df = None
        if skip_processed and save_path and save_path.exists():
            logger.info(f"Loading existing data from {save_path.name}")
            existing_df = pl.read_parquet(save_path)
            existing_files = set(existing_df["filename"].unique().to_list())
            logger.info(f"Found {len(existing_files):,} already processed files")

            # Filter out already processed files
            xml_files = [f for f in xml_files if f.name not in existing_files]
            logger.info(f"{len(xml_files):,} new files to process")

            if len(xml_files) == 0:
                logger.info("All files already processed - returning existing data")
                return existing_df

        # Pre-parse all unique METS files for caching
        logger.info("Pre-parsing METS metadata files...")
        mets_cache = self._build_mets_cache(xml_files)
        logger.info(f"Cached {len(mets_cache):,} METS files")

        all_lines = []

        # Verify we have source_name
        if not self.source_name:
            raise RuntimeError(
                "source_name is required for ID generation. "
                "Initialize DataLoader with source_name parameter."
            )

        # Always use source_name for ID generation (canonical identifier)
        # ZDB ID is kept in config for provenance only, not used in IDs
        source_id = self.source_name
        logger.info(f"Using source name for ID generation: {source_id}")

        # Process files in parallel with METS cache
        logger.info("Processing ALTO XML files in parallel...")
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(parse_file_worker, str(fp), source_id, mets_cache): fp
                for fp in xml_files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing XML files"):
                lines_dicts, success = future.result()
                if success:
                    all_lines.extend(lines_dicts)

        logger.info(f"Extracted {len(all_lines):,} text lines")

        # Create Polars DataFrame from new data
        logger.debug("Creating DataFrame...")
        new_df = pl.DataFrame(all_lines) if all_lines else pl.DataFrame()

        # Combine with existing data if we loaded it
        if existing_df is not None and len(new_df) > 0:
            logger.info(f"Combining {len(existing_df):,} existing + {len(new_df):,} new rows")
            df = pl.concat([existing_df, new_df])
        elif existing_df is not None:
            # No new data, just return existing
            df = existing_df
        else:
            # No existing data, just use new
            df = new_df

        # Sort by filename for consistent ordering
        if len(df) > 0:
            logger.debug("Sorting DataFrame by filename...")
            df = df.sort("filename")

        # Validate: Check that we have data from all newly processed XML files
        if len(new_df) > 0:
            unique_files = new_df["filename"].n_unique()
            expected_files = len(xml_files)

            if unique_files != expected_files:
                logger.warning(
                    f"Expected {expected_files} files, but found data from "
                    f"{unique_files} files. {expected_files - unique_files} files may have failed."
                )
            else:
                logger.info(f"Data from all {expected_files:,} files present")
        elif len(all_lines) > 0:
            logger.warning("DataFrame is empty despite having lines!")

        # Save to Parquet if we have a path and data
        if save_path and len(df) > 0:
            logger.info(f"Saving {len(df):,} rows to {save_path.name}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(save_path, compression="zstd")
            logger.info("Saved successfully")

        return df

    @staticmethod
    def _extract_source_name(directory: Path) -> Optional[str]:
        """
        Extract source name from directory path.

        Examples:
            data/raw/der_tag/xml_ocr -> der_tag
            /path/to/der_tag/xml_ocr -> der_tag
            data/raw/some_newspaper -> some_newspaper

        Args:
            directory: Path to data directory

        Returns:
            Source name or None if not found
        """
        parts = directory.parts
        # Look for pattern: .../raw/{source}/...
        try:
            raw_idx = parts.index("raw")
            if raw_idx + 1 < len(parts):
                return parts[raw_idx + 1]
        except (ValueError, IndexError):
            pass

        # Fallback: use parent directory name
        if directory.parent.name not in ["raw", "data"]:
            return directory.parent.name

        return None

    def _build_mets_cache(self, alto_files: List[Path]) -> Dict[str, Dict]:
        """
        Pre-parse all unique METS files and build cache in parallel.

        This optimization reduces METS parsing from N (ALTO files) to M (issues),
        since multiple ALTO pages share the same METS file.

        Args:
            alto_files: List of ALTO file paths

        Returns:
            Dict mapping METS file paths to metadata dicts
        """
        mets_parser = METSParser()

        # First pass: find all unique METS files
        unique_mets_files = set()
        for alto_file in alto_files:
            mets_file = mets_parser.find_mets_for_alto(alto_file)
            if mets_file and mets_file.exists():
                unique_mets_files.add(mets_file)

        unique_mets_list = list(unique_mets_files)

        # Second pass: parse METS files in parallel with progress bar
        mets_cache = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(parse_mets_worker, str(mets_file)): mets_file
                for mets_file in unique_mets_list
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Caching METS files", unit="files"
            ):
                mets_path_str, metadata = future.result()
                if metadata:
                    mets_cache[mets_path_str] = metadata

        return mets_cache

    def load_files(
        self,
        filepaths: List[Path],
        output_parquet: Optional[Path] = None,
        source_name: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Load specific ALTO XML files in parallel.

        Args:
            filepaths: List of paths to ALTO XML files
            output_parquet: If provided, save result to Parquet file
            source_name: Source identifier (defaults to self.source_name)

        Returns:
            Polars DataFrame with line-level data
        """
        logger.info(f"Loading {len(filepaths)} ALTO XML files")

        # Use provided source_name or fall back to instance source_name
        src = source_name or self.source_name
        if not src:
            raise RuntimeError(
                "source_name is required for ID generation. "
                "Provide source_name parameter or initialize DataLoader with source_name."
            )

        # Always use source_name for ID generation (canonical identifier)
        # ZDB ID is kept in config for provenance only, not used in IDs
        source_id = src
        logger.info(f"Using source name for ID generation: {source_id}")

        all_lines = []

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(parse_file_worker, str(fp), source_id): fp for fp in filepaths
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing XML files"):
                lines_dicts, success = future.result()
                if success:
                    all_lines.extend(lines_dicts)

        logger.info(f"Extracted {len(all_lines)} text lines")

        # Create Polars DataFrame
        df = pl.DataFrame(all_lines)

        # Save to Parquet if requested
        if output_parquet:
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(output_parquet, compression="zstd")

        return df

    @staticmethod
    def load_parquet(filepath: Path) -> pl.DataFrame:
        """
        Load previously saved Parquet file.

        Args:
            filepath: Path to Parquet file

        Returns:
            Polars DataFrame
        """
        return pl.read_parquet(filepath)
