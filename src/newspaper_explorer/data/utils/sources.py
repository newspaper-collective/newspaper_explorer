"""
Source configuration management utilities.

Functions for loading source configurations and computing paths.
Uses models from newspaper_explorer.models.core.sources.
"""

import logging
from pathlib import Path
from typing import Optional

from natsort import natsorted
import polars as pl

from newspaper_explorer.config.base import get_config
from newspaper_explorer.models.data.sources import SourceConfig, SourceStatus

logger = logging.getLogger(__name__)


def list_available_sources() -> list[str]:
    """
    List all available sources from the sources directory.

    Returns:
        list[str]: Naturally sorted list of source names

    Example:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['der_tag']
    """
    config = get_config()

    if config.sources_dir.exists():
        sources = [source_file.stem for source_file in config.sources_dir.glob("*.json")]
    else:
        sources = []

    return natsorted(sources)


def load_source_config(source_name: str) -> SourceConfig:
    """
    Load and validate source configuration.

    Args:
        source_name: Name of the source (e.g., 'der_tag')

    Returns:
        SourceConfig: Validated Pydantic model

    Raises:
        ValueError: If source not found or validation fails

    Example:
        >>> config = load_source_config("der_tag")
        >>> print(config.metadata.newspaper_title)
        'Der Tag'
        >>> print(config.loading.pattern)
        '**/fulltext/*.xml'
        >>> print(len(config.parts))
        7
    """
    config = get_config()
    source_file = config.sources_dir / f"{source_name}.json"

    if not source_file.exists():
        available = list_available_sources()
        raise ValueError(
            f"Source '{source_name}' not found. Available sources: {', '.join(available)}"
        )

    try:
        return SourceConfig.from_json_file(source_file)

    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid source configuration for '{source_name}': {e}") from e


def get_source_paths(source_config: SourceConfig) -> dict[str, Path]:
    """
    Get standard paths for a source's data directories and files.

    Args:
        source_config: Validated SourceConfig object

    Returns:
        Dict[str, Path]: Dictionary with paths:
            - raw_dir: Raw XML/OCR files
            - text_dir: Parsed text data
            - images_dir: Downloaded images
            - output_file: Main parquet output

    Example:
        >>> config = load_source_config("der_tag")
        >>> paths = get_source_paths(config)
        >>> print(paths["raw_dir"])
        PosixPath('data/raw/der_tag/xml_ocr')
    """
    config = get_config()
    dataset_name = source_config.dataset_name
    data_type = source_config.data_type

    raw_dir = config.data_dir / "raw" / dataset_name / data_type
    text_dir = config.data_dir / "raw" / dataset_name / "text"
    images_dir = config.data_dir / "raw" / dataset_name / "images"
    output_file = text_dir / f"{dataset_name}_lines.parquet"

    return {
        "raw_dir": raw_dir,
        "text_dir": text_dir,
        "images_dir": images_dir,
        "output_file": output_file,
    }


def get_source_status(source_name: str) -> SourceStatus:
    """
    Get comprehensive status information for a source.

    Collects all status information including raw XML, parsed data,
    aggregated text blocks, and images. Returns a structured SourceStatus
    model for presentation.

    Args:
        source_name: Name of the source (e.g., 'der_tag')

    Returns:
        SourceStatus: Comprehensive status information

    Raises:
        ValueError: If source not found

    Example:
        >>> status = get_source_status("der_tag")
        >>> print(f"Parsed: {status.has_parsed_data}")
        True
        >>> print(f"Coverage: {status.parsing_coverage_pct:.1f}%")
        98.5
    """
    # Load config
    config = load_source_config(source_name)
    paths = get_source_paths(config)
    app_config = get_config()

    # Raw XML status - single pass through directory tree
    raw_dir = paths["raw_dir"]
    has_raw_xml = raw_dir.exists()
    alto_file_count = 0
    mets_file_count = 0
    total_xml_count = 0

    if has_raw_xml:
        # Single pass: collect all XML files and categorize them
        alto_files: list[Path] = []
        all_xml_count = 0

        for xml_file in raw_dir.glob("**/*.xml"):
            all_xml_count += 1
            # ALTO files are in fulltext/ subdirectories
            if "fulltext" in xml_file.parts:
                alto_files.append(xml_file)

        alto_file_count = len(alto_files)
        total_xml_count = all_xml_count
        mets_file_count = total_xml_count - alto_file_count

    # Download archives status - single pass for both tar.gz and zip
    downloads_dir = app_config.download_dir / config.dataset_name
    has_download_archives = False
    download_archives_count = 0

    if downloads_dir.exists():
        # Single pass: count all archives
        archive_count = 0
        for archive_file in downloads_dir.glob("**/*"):
            if archive_file.is_file() and (
                (archive_file.suffix == ".gz" and archive_file.stem.endswith(".tar"))
                or archive_file.suffix == ".zip"
            ):
                archive_count += 1

        download_archives_count = archive_count
        has_download_archives = archive_count > 0

    # Parsed data status
    output_file = paths["output_file"]
    has_parsed_data = output_file.exists()
    parsed_row_count = 0
    parsed_file_count = 0
    parsing_coverage_pct: Optional[float] = None
    parsed_date_range: Optional[tuple[str, str]] = None
    parsed_size_mb = 0.0

    if has_parsed_data:
        df = pl.read_parquet(output_file)
        parsed_row_count = len(df)
        parsed_file_count = df["filename"].n_unique()

        # Calculate coverage
        if alto_file_count > 0:
            parsing_coverage_pct = (parsed_file_count / alto_file_count) * 100

        # Get date range
        if "date" in df.columns and len(df) > 0:
            min_date = str(df["date"].min())
            max_date = str(df["date"].max())
            parsed_date_range = (min_date, max_date)

        # File size
        parsed_size_mb = output_file.stat().st_size / (1024 * 1024)

    # Aggregated data status
    textblocks_path = (
        app_config.data_dir / "processed" / config.dataset_name / "text" / "textblocks.parquet"
    )
    has_aggregated_data = textblocks_path.exists()
    aggregated_row_count = 0
    aggregated_size_mb = 0.0

    if has_aggregated_data:
        df_agg = pl.read_parquet(textblocks_path)
        aggregated_row_count = len(df_agg)
        aggregated_size_mb = textblocks_path.stat().st_size / (1024 * 1024)

    # Image status - try index first, fallback to ImageDownloader
    images_dir = paths["images_dir"]
    has_images = False
    image_count = 0
    images_expected = 0
    image_coverage_pct: Optional[float] = None
    total_size_gb = 0.0
    image_year_range: Optional[tuple[int, int]] = None
    has_image_index = False

    try:
        from newspaper_explorer.data.indexing.image_index import ImageIndexer # noqa: I001, PLC0415 lazy loading to avoid circular imports

        indexer = ImageIndexer(source_name)
        index = indexer.load_index()

        if index is not None and len(index) > 0:
            has_image_index = True
            has_images = True
            stats = indexer.get_stats()

            # Extract values - all are guaranteed present by ImageStats TypedDict
            image_count = stats["total_images"]
            total_size_gb = stats["total_size_gb"]
            images_expected = stats["total_images_expected"]

            # Year range can be None if no images
            min_year = stats["min_year"]
            max_year = stats["max_year"]
            image_year_range = (min_year, max_year) if min_year and max_year else None

            if images_expected > 0:
                image_coverage_pct = (image_count / images_expected) * 100
            else:
                # Fallback if metadata doesn't have expected count
                images_expected = image_count
                image_coverage_pct = 100.0
        else:
            # No index, use ImageDownloader directly
            try:
                from newspaper_explorer.data.download.images import ImageDownloader # noqa: I001, PLC0415 lazy loading to avoid circular imports

                image_downloader = ImageDownloader(source_name=source_name)
                image_status = image_downloader.get_download_status()

                has_images = image_status["images_dir_exists"]
                image_count = image_status["images_downloaded"]
                images_expected = image_status["total_images_expected"]

                if images_expected > 0:
                    image_coverage_pct = image_status["coverage_pct"]
            except (OSError, ValueError, KeyError) as e:
                logger.debug(f"Could not get image status from downloader: {e}")
                # Fall through to defaults
    except (OSError, ValueError, RuntimeError) as e:
        logger.debug(f"Could not load image index for {source_name}: {e}")
        # If both fail, use defaults
        has_images = images_dir.exists()

    return SourceStatus(
        source_name=source_name,
        # Raw XML
        has_raw_xml=has_raw_xml,
        alto_file_count=alto_file_count,
        mets_file_count=mets_file_count,
        total_xml_count=total_xml_count,
        raw_dir=str(raw_dir),
        # Download archives
        has_download_archives=has_download_archives,
        download_archives_count=download_archives_count,
        downloads_dir=str(downloads_dir),
        # Parsed data
        has_parsed_data=has_parsed_data,
        parsed_row_count=parsed_row_count,
        parsed_file_count=parsed_file_count,
        parsing_coverage_pct=parsing_coverage_pct,
        parsed_date_range=parsed_date_range,
        parsed_size_mb=parsed_size_mb,
        output_file=str(output_file),
        # Aggregated data
        has_aggregated_data=has_aggregated_data,
        aggregated_row_count=aggregated_row_count,
        aggregated_size_mb=aggregated_size_mb,
        textblocks_path=str(textblocks_path),
        # Images
        has_images=has_images,
        image_count=image_count,
        images_expected=images_expected,
        image_coverage_pct=image_coverage_pct,
        total_size_gb=total_size_gb,
        image_year_range=image_year_range,
        images_dir=str(images_dir),
        has_image_index=has_image_index,
    )
