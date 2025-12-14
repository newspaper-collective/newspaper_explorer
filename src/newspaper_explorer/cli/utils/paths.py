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
    input_type: str = "textblocks",
    custom_path: Optional[Union[Path, str]] = None,
) -> tuple[Path, str]:
    """
    Resolve input file path and detect ID column.

    Args:
        source: Source name (e.g., "der_tag")
        input_type: Input data type ("textblocks" or "lines")
        custom_path: Optional custom path to parquet file

    Returns:
        Tuple of (input_path, id_column)

    Raises:
        FileNotFoundError: If resolved path doesn't exist
        ValueError: If ID column cannot be detected

    Example:
        >>> path, id_col = resolve_input_path("der_tag", "textblocks")
        >>> print(path)
        /path/to/data/processed/der_tag/text/textblocks.parquet
        >>> print(id_col)
        text_block_id
    """
    config = get_config()

    if custom_path:
        input_path = Path(custom_path)
        id_col = detect_id_column(input_path)
    elif input_type == "textblocks":
        input_path = config.data_dir / "processed" / source / "text" / "textblocks.parquet"
        id_col = "text_block_id"
    else:  # lines
        input_path = config.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"
        id_col = "line_id"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

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
