"""
Input path resolution utilities for CLI commands.

Centralizes logic for resolving input file paths based on source name,
input type, and custom paths.
"""

from pathlib import Path
from typing import Optional, Union

import polars as pl

from newspaper_explorer.config.base import get_config


def resolve_input_path(
    source: str,
    input_file: Optional[Union[Path, str]] = None,
    id_column: Optional[str] = None,
) -> tuple[Path, str]:
    """
    Resolve input file path and ID column with auto-detection.

    Auto-detects between textblocks and lines files, preferring textblocks
    for better context in analysis tasks.

    Args:
        source: Source name (e.g., "der_tag")
        input_file: Optional custom path to parquet file
        id_column: Optional explicit ID column name (auto-detected if not provided)

    Returns:
        Tuple of (input_path, id_column)

    Raises:
        FileNotFoundError: If no suitable input file exists
        ValueError: If ID column cannot be detected

    Example:
        >>> # Auto-detect (prefers textblocks)
        >>> path, id_col = resolve_input_path("der_tag")
        >>> print(path)
        /path/to/data/processed/der_tag/text/textblocks.parquet
        >>> print(id_col)
        text_block_id

        >>> # Custom file
        >>> path, id_col = resolve_input_path("der_tag", input_file="custom.parquet")
    """
    config = get_config()

    if input_file:
        # Custom file provided
        input_path = Path(input_file)
        id_col = id_column if id_column else detect_id_column(input_path)
    else:
        # Auto-detect default file (textblocks preferred)
        textblocks_path = config.data_dir / "processed" / source / "text" / "textblocks.parquet"
        lines_path = config.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"

        if textblocks_path.exists():
            input_path = textblocks_path
            id_col = id_column or "text_block_id"
        elif lines_path.exists():
            input_path = lines_path
            id_col = id_column or "line_id"
        else:
            raise FileNotFoundError(
                f"No input file found for source '{source}'. "
                f"Run parsing first: newspaper-explorer data parse --source {source}"
            )

    return input_path, id_col


def detect_id_column(path: Path) -> str:
    """
    Auto-detect ID column from parquet file.

    Args:
        path: Path to parquet file

    Returns:
        ID column name

    Raises:
        ValueError: If ID column cannot be detected

    Example:
        >>> detect_id_column(Path("textblocks.parquet"))
        'text_block_id'
    """
    # Read just the schema (no data)
    df_sample = pl.read_parquet(path, n_rows=1)

    if "text_block_id" in df_sample.columns:
        return "text_block_id"
    if "line_id" in df_sample.columns:
        return "line_id"

    raise ValueError(f"Cannot auto-detect ID column in {path}")


def resolve_output_path(
    source: str,
    analysis_type: str,
    run_id: str,
    output_name: Optional[str] = None,
) -> Path:
    """
    Resolve output path for analysis results.

    Args:
        source: Source name (e.g., "der_tag")
        analysis_type: Type of analysis (e.g., "entities", "topics")
        run_id: Analysis run ID
        output_name: Optional custom output filename (default: results.parquet)

    Returns:
        Path to output file

    Example:
        >>> resolve_output_path("der_tag", "entities", "gliner_2024-01-01")
        /path/to/results/entities/der_tag/gliner_2024-01-01/results.parquet
    """
    config = get_config()
    output_name = output_name or "results.parquet"

    output_dir = config.results_dir / analysis_type / source / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / output_name


def get_source_text_path(source: str, input_type: str = "textblocks") -> Path:
    """
    Get path to source text data without validation.

    Args:
        source: Source name
        input_type: Input data type ("textblocks" or "lines")

    Returns:
        Path to text data file (may not exist)

    Example:
        >>> get_source_text_path("der_tag", "textblocks")
        /path/to/data/processed/der_tag/text/textblocks.parquet
    """
    config = get_config()

    if input_type == "textblocks":
        return config.data_dir / "processed" / source / "text" / "textblocks.parquet"

    return config.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"


def get_source_images_path(source: str) -> Path:
    """
    Get path to source images directory.

    Args:
        source: Source name

    Returns:
        Path to images directory (may not exist)

    Example:
        >>> get_source_images_path("der_tag")
        /path/to/data/raw/der_tag/images
    """
    config = get_config()
    return config.data_dir / "raw" / source / "images"


def get_analysis_results_dir(source: str, analysis_type: str) -> Path:
    """
    Get directory containing all analysis results for a source.

    Args:
        source: Source name
        analysis_type: Type of analysis

    Returns:
        Path to analysis results directory

    Example:
        >>> get_analysis_results_dir("der_tag", "entities")
        /path/to/results/entities/der_tag
    """
    config = get_config()
    return config.results_dir / analysis_type / source
