"""Info commands for the data CLI."""

from typing import TYPE_CHECKING

import click

from newspaper_explorer.cli.utils import errors, output
from newspaper_explorer.cli.utils.options import (
    limit_option,
    source_option,
    text_column_option,
)
from newspaper_explorer.data.ingest.loader import DataIngester
from newspaper_explorer.data.utils.sources import (
    get_source_paths,
    get_source_status,
    list_available_sources,
    load_source_config,
)
from newspaper_explorer.data.utils.text import (
    analyze_text_line_character_lengths,
    analyze_token_lengths,
    get_longest_lines_by_tokens,
)

if TYPE_CHECKING:
    from newspaper_explorer.models.data.stats import (
        CharDistribution,
        CharLengthStats,
        TokenDistribution,
        TokenLengthStats,
    )

# Maximum token length for BERT models
MAX_TOKEN_LENGTH = 512
# Maximum text preview length for display
TEXT_PREVIEW_LENGTH = 300
# Speedup thresholds for dynamic padding analysis
SPEEDUP_THRESHOLD_SIGNIFICANT = 2.0
SPEEDUP_THRESHOLD_MODERATE = 1.5
# Full coverage percentage threshold
FULL_COVERAGE_PCT = 100


@click.group(name="info")
def info_group() -> None:
    """Information and analysis commands for data sources."""
    pass


@info_group.command(name="status")
@source_option()
def status(source: str) -> None:
    """
    Show comprehensive status information for a source.

    Displays download status, extraction status, XML file counts,
    parsed data status, and processing coverage for a newspaper source.

    \b
    Examples:
      newspaper-explorer data info --source der_tag
    """

    try:
        # Load config for metadata
        config = load_source_config(source)

        # Main header
        output.header(f"SOURCE INFORMATION: {source.upper()}")
        output.info("Gathering status information...", muted=True)

        # Get comprehensive status
        status = get_source_status(source)

        # Show metadata
        if config.metadata:
            output.key_value("Newspaper", config.metadata.newspaper_title or "N/A")
            output.key_value("Years", config.metadata.years_available or "N/A")
            output.key_value("Language", config.metadata.language or "N/A")
            if config.metadata.location:
                output.key_value("Location", config.metadata.location)

        # Download & Extraction Status
        output.section("DOWNLOAD & EXTRACTION STATUS")

        if status.has_raw_xml:
            if status.xml_file_count > 0:
                output.success("Data extracted and ready")
                output.key_value("Location", status.raw_dir)
                output.key_value("XML files", f"{status.xml_file_count:,}")
            else:
                output.error("Directory exists but no XML files found", symbol=False)
                output.key_value("Location", status.raw_dir)
                output.warning(
                    f"Data may not be properly extracted. "
                    f"Run: newspaper-explorer data unpack --source {source}"
                )
        else:
            output.error("Not extracted", symbol=False)
            output.key_value("Expected location", status.raw_dir)
            output.warning(
                f"No data found. Run:\n"
                f"  newspaper-explorer data download --source {source}\n"
                f"  newspaper-explorer data unpack --source {source}"
            )

        # Raw XML Files
        output.section("RAW XML FILES")
        output.key_value("Directory", status.raw_dir)
        output.key_value(
            "Pattern", config.loading.pattern if config.loading else "**/fulltext/*.xml"
        )

        if status.has_raw_xml and status.xml_file_count > 0:
            output.key_value("XML files found", f"{status.xml_file_count:,}")
        else:
            output.key_value("XML files found", "0 (directory not found)")
            output.warning(
                f"No XML files. Run:\n"
                f"  newspaper-explorer data download --source {source}\n"
                f"  newspaper-explorer data unpack --source {source}"
            )

        # Parsed Data Status
        output.section("PARSED DATA (Parquet)")
        output.key_value("Location", status.output_file)

        if status.has_parsed_data:
            output.success("Exists")
            output.key_value("Total lines", f"{status.parsed_row_count:,}")
            output.key_value("Files parsed", f"{status.parsed_file_count:,}")

            # Coverage
            if status.parsing_coverage_pct is not None:
                output.key_value(
                    "Coverage",
                    f"{status.parsing_coverage_pct:.1f}% "
                    f"({status.parsed_file_count}/{status.xml_file_count})",
                )

                if status.parsed_file_count < status.xml_file_count:
                    remaining = status.xml_file_count - status.parsed_file_count
                    output.warning(
                        f"{remaining:,} XML files not yet parsed. "
                        f"Run: newspaper-explorer data parse --source {source}"
                    )
                else:
                    output.success("All XML files parsed!")

            # Date range
            if status.parsed_date_range:
                min_date, max_date = status.parsed_date_range
                output.key_value("Date range", f"{min_date} to {max_date}")

            # Size
            output.key_value("File size", f"{status.parsed_size_mb:.1f} MB")
        else:
            output.error("Not found", symbol=False)
            output.warning(f"No parsed data. Run: newspaper-explorer data parse --source {source}")

        # Aggregated Data Status
        output.section("AGGREGATED TEXT BLOCKS")
        output.key_value("Location", status.textblocks_path)

        if status.has_aggregated_data:
            output.success("Exists")
            output.key_value("Text blocks", f"{status.aggregated_row_count:,}")
            output.key_value("File size", f"{status.aggregated_size_mb:.1f} MB")
            output.success("Ready for preprocessing!")
        else:
            output.error("Not found", symbol=False)
            if status.has_parsed_data:
                output.warning(
                    f"Parsed data exists but not aggregated. "
                    f"Run: newspaper-explorer data aggregate --source {source}"
                )

        # Image Status
        output.section("PAGE IMAGES")
        output.key_value("Location", status.images_dir)

        if status.has_image_index:
            # Use indexed data
            output.success("Indexed")
            output.key_value("Images indexed", f"{status.image_count:,}")
            output.key_value("Total size", f"{status.total_size_gb:.2f} GB")

            if status.image_year_range:
                min_year, max_year = status.image_year_range
                years = max_year - min_year + 1
                output.key_value("Year range", f"{min_year} - {max_year} ({years} years)")

            if status.images_expected > 0:
                output.key_value("Images expected", f"{status.images_expected:,}")
                if status.image_coverage_pct:
                    output.key_value("Coverage", f"{status.image_coverage_pct:.2f}%")

                    if status.image_coverage_pct < FULL_COVERAGE_PCT:
                        missing = status.images_expected - status.image_count
                        output.warning(
                            f"{missing:,} images missing. "
                            f"Run: newspaper-explorer data download-images --source {source}"
                        )
                    elif status.image_count > status.images_expected:
                        extra = status.image_count - status.images_expected
                        output.success(f"All images downloaded! ({extra:,} extra images found)")
                    else:
                        output.success("All images downloaded!")
            else:
                output.info("Tip: Image index is available for fast queries", muted=True)

        elif status.has_images:
            # No index, basic status
            output.success("Directory exists")
            output.key_value("Images downloaded", f"{status.image_count:,}")

            if status.images_expected > 0:
                output.key_value("Images expected", f"{status.images_expected:,}")
                if status.image_coverage_pct:
                    output.key_value("Coverage", f"{status.image_coverage_pct:.2f}%")

                    if status.image_coverage_pct < FULL_COVERAGE_PCT:
                        missing = status.images_expected - status.image_count
                        output.warning(
                            f"{missing:,} images not yet downloaded. "
                            f"Run: newspaper-explorer data download-images --source {source}"
                        )
                    elif status.image_count > status.images_expected:
                        extra = status.image_count - status.images_expected
                        output.success(f"All images downloaded! ({extra:,} extra images found)")
                    else:
                        output.success("All images downloaded!")

            # Suggest index
            if status.image_count > 0:
                output.info(
                    f"Tip: Create an image index for faster queries:\n"
                    f"     # (Future) newspaper-explorer data index-images --source {source}",
                    muted=True,
                )
        else:
            output.error("Not found", symbol=False)
            if status.images_expected > 0:
                output.key_value("Images expected", f"{status.images_expected:,}")
                output.warning(
                    f"No images downloaded. "
                    f"Run: newspaper-explorer data download-images --source {source}"
                )

        click.echo()  # Final newline

    except FileNotFoundError as e:
        errors.handle_error(e)
    except (ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@info_group.command(name="list-sources")
def list_sources_cmd() -> None:
    """
    List all available data sources.

    Shows all newspaper sources with their metadata and download information.
    This command lists the available source configurations, not the individual
    data parts within a source.

    \b
    Examples:
      newspaper-explorer data info list-sources
    """

    sources = list_available_sources()

    if not sources:
        output.warning("No data sources found in data/sources/")
        return

    output.header("Available Data Sources")

    for source_name in sources:
        config = load_source_config(source_name)
        metadata = config.metadata

        # Get human-readable sizes from parts
        sizes = [part.size for part in config.parts if part.size]
        size_str = ", ".join(sizes) if sizes else "unknown"

        output.section(config.dataset_name)
        output.key_value("Newspaper", metadata.newspaper_title)
        output.key_value("Years", metadata.years_available)
        output.key_value("Language", metadata.language)
        if metadata.location:
            output.key_value("Location", metadata.location)
        output.key_value("Data type", config.data_type)
        output.key_value("Parts", f"{len(config.parts)} (sizes: {size_str})")

    click.echo()
    output.info(
        "Tip: Use 'newspaper-explorer data info status --source <name>' to see details", muted=True
    )


@info_group.command(name="analyze-chars")
@source_option()
@limit_option(default=50000)
@text_column_option()
def analyze_chars(source: str, limit: int, text_column: str) -> None:
    """
    Analyze character lengths in newspaper text data.

    This command samples texts from the parsed Parquet file and computes
    comprehensive statistics about character lengths. Useful for understanding
    text length distribution and comparing with token-based analysis.

    \b
    Examples:
      newspaper-explorer analyze info analyze-chars --source der_tag
      newspaper-explorer analyze info analyze-chars --source der_tag --limit 100000
    """

    try:
        # Load config
        config = load_source_config(source)
        source_name = config.dataset_name

        output.header(f"CHARACTER LENGTH ANALYSIS: {source_name}")

        # Get path to parquet file
        paths = get_source_paths(config)
        parquet_file = paths["output_file"]

        if not parquet_file.exists():
            output.error(f"Parquet file not found: {parquet_file}")
            output.warning(f"Run parsing first:\n  newspaper-explorer data parse --source {source}")
            raise click.Abort()

        # Load data directly from parquet (skip XML scanning)
        output.info(f"Loading data from: {parquet_file.name}")
        df = DataIngester.load_parquet(parquet_file)

        if len(df) == 0:
            output.error("Empty parquet file")
            output.warning(f"Re-run parsing:\n  newspaper-explorer data parse --source {source}")
            raise click.Abort()

        # Analyze character lengths
        output.info(f"Analyzing character lengths (sample size: {limit:,})...")

        stats: CharLengthStats = analyze_text_line_character_lengths(
            df,
            text_column=text_column,
            sample_size=limit,
        )

        # Display results
        output.section("CHARACTER LENGTH STATISTICS")
        output.key_value("Total rows", f"{stats['total_rows']:,}")
        output.key_value("Sample analyzed", f"{stats['sample_size']:,}")

        output.divider()
        output.key_value("Min characters", f"{stats['min_chars']}")
        output.key_value("Max characters", f"{stats['max_chars']}")
        output.key_value("Mean characters", f"{stats['mean_chars']:.1f}")
        output.key_value("Median characters", f"{stats['median_chars']}")

        output.divider()
        output.key_value("90th percentile", f"{stats['p90_chars']} characters")
        output.key_value("95th percentile", f"{stats['p95_chars']} characters")
        output.key_value("99th percentile", f"{stats['p99_chars']} characters")

        dist: CharDistribution = stats["distribution"]
        total = stats["sample_size"]
        output.divider()
        output.key_value(
            "≤   50 chars", f"{dist['under_50']:,} ({100 * dist['under_50'] / total:.1f}%)"
        )
        output.key_value(
            "≤  100 chars", f"{dist['under_100']:,} ({100 * dist['under_100'] / total:.1f}%)"
        )
        output.key_value(
            "≤  200 chars", f"{dist['under_200']:,} ({100 * dist['under_200'] / total:.1f}%)"
        )
        output.key_value(
            "≤  500 chars", f"{dist['under_500']:,} ({100 * dist['under_500'] / total:.1f}%)"
        )
        output.key_value(
            "≤ 1000 chars", f"{dist['under_1000']:,} ({100 * dist['under_1000'] / total:.1f}%)"
        )

        # Show longest examples
        output.section("LONGEST TEXT EXAMPLES")
        for i, (char_count, text) in enumerate(stats["longest_examples"][:3], 1):
            output.divider()
            output.key_value(f"Example {i}", f"{char_count} characters")
            preview = (
                text[:TEXT_PREVIEW_LENGTH] + "..." if len(text) > TEXT_PREVIEW_LENGTH else text
            )
            output.info(preview, muted=True)

        click.echo()
        output.success("Analysis complete")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@info_group.command(name="analyze-tokens")
@source_option()
@limit_option(default=50000)
@click.option(
    "--tokenizer",
    type=str,
    default="deepset/gbert-large",
    help="HuggingFace tokenizer to use",
    show_default=True,
)
@text_column_option()
def analyze_tokens(source: str, limit: int, tokenizer: str, text_column: str) -> None:
    """
    Analyze token lengths in newspaper text data.

    This command samples texts from the parsed Parquet file and computes
    comprehensive statistics about token lengths using a BERT tokenizer.
    Useful for understanding padding requirements and estimating speedup
    potential from dynamic padding.

    \b
    Examples:
      newspaper-explorer analyze info analyze-tokens --source der_tag
      newspaper-explorer analyze info analyze-tokens --source der_tag --limit 100000
      newspaper-explorer analyze info analyze-tokens --source der_tag --tokenizer bert-base-german-cased
    """

    try:
        # Load config
        config = load_source_config(source)
        source_name = config.dataset_name

        output.header(f"TOKEN LENGTH ANALYSIS: {source_name}")

        # Get path to parquet file
        paths = get_source_paths(config)
        parquet_file = paths["output_file"]

        if not parquet_file.exists():
            output.error(f"Parquet file not found: {parquet_file}")
            output.warning(f"Run parsing first:\n  newspaper-explorer data parse --source {source}")
            raise click.Abort()

        # Load data directly from parquet (skip XML scanning)
        output.info(f"Loading data from: {parquet_file.name}")
        df = DataIngester.load_parquet(parquet_file)

        if len(df) == 0:
            output.error("Empty parquet file")
            output.warning(f"Re-run parsing:\n  newspaper-explorer data parse --source {source}")
            raise click.Abort()

        # Analyze token lengths
        output.info(f"Analyzing token lengths (sample size: {limit:,})...")
        output.info(f"Tokenizer: {tokenizer}", muted=True)

        stats: TokenLengthStats = analyze_token_lengths(
            df,
            text_column=text_column,
            tokenizer_name=tokenizer,
            sample_size=limit,
        )

        # Display results
        output.section("TOKEN LENGTH STATISTICS")
        output.key_value("Total rows", f"{stats['total_rows']:,}")
        output.key_value("Sample analyzed", f"{stats['sample_size']:,}")

        output.divider()
        output.key_value("Min tokens", stats["min_tokens"])
        output.key_value("Max tokens", stats["max_tokens"])
        output.key_value("Mean tokens", f"{stats['mean_tokens']:.1f}")
        output.key_value("Median tokens", stats["median_tokens"])

        output.divider()
        output.key_value("90th percentile", f"{stats['p90_tokens']} tokens")
        output.key_value("95th percentile", f"{stats['p95_tokens']} tokens")
        output.key_value("99th percentile", f"{stats['p99_tokens']} tokens")

        dist: TokenDistribution = stats["distribution"]
        total = stats["sample_size"]
        output.divider()
        output.key_value(
            "≤  50 tokens", f"{dist['under_50']:,} ({100 * dist['under_50'] / total:.1f}%)"
        )
        output.key_value(
            "≤ 100 tokens", f"{dist['under_100']:,} ({100 * dist['under_100'] / total:.1f}%)"
        )
        output.key_value(
            "≤ 200 tokens", f"{dist['under_200']:,} ({100 * dist['under_200'] / total:.1f}%)"
        )
        output.key_value(
            "≤ 300 tokens", f"{dist['under_300']:,} ({100 * dist['under_300'] / total:.1f}%)"
        )

        if dist["at_max_length"] > 0:
            output.key_value(
                "= 512 tokens",
                f"{dist['at_max_length']:,} ({stats['truncated_percent']:.2f}%) [truncated]",
            )

        output.section("PADDING ANALYSIS")
        output.key_value("Average tokens (static)", f"{stats['mean_tokens']:.1f} / 512")
        output.key_value("Wasted compute", f"{stats['wasted_padding_percent']:.1f}%")
        output.divider()
        output.key_value("Expected speedup (dynamic)", f"{stats['expected_speedup']:.1f}x")

        click.echo()
        if stats["expected_speedup"] > SPEEDUP_THRESHOLD_SIGNIFICANT:
            output.success("Dynamic padding will provide significant speedup!")
        elif stats["expected_speedup"] > SPEEDUP_THRESHOLD_MODERATE:
            output.success("Dynamic padding will provide moderate speedup.")
        else:
            output.info("Dynamic padding may provide limited benefit.", symbol=True)

        # Show longest examples
        output.section("LONGEST TEXT EXAMPLES")

        for i, (token_count, text) in enumerate(stats["longest_examples"][:3], 1):
            output.divider()
            output.key_value(f"Example {i}", f"{token_count} tokens")
            preview = (
                text[:TEXT_PREVIEW_LENGTH] + "..." if len(text) > TEXT_PREVIEW_LENGTH else text
            )
            output.info(preview, muted=True)

        click.echo()

    except (FileNotFoundError, ValueError, ImportError) as e:
        errors.handle_error(e, show_traceback=True)


@info_group.command(name="longest-tokens")
@source_option()
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of longest lines to retrieve",
    show_default=True,
)
@click.option(
    "--tokenizer",
    type=str,
    default="deepset/gbert-large",
    help="HuggingFace tokenizer to use",
    show_default=True,
)
@text_column_option()
@click.option(
    "--show-metadata/--no-metadata",
    default=True,
    help="Show metadata columns",
)
def longest_tokens(
    source: str, top_n: int, tokenizer: str, text_column: str, *, show_metadata: bool
) -> None:
    """
    Show the longest text lines by token count.

    This command finds texts with the highest token counts using a tokenizer.
    Useful for identifying edge cases, texts that will be truncated, and
    understanding the distribution of long texts.

    \b
    Examples:
      newspaper-explorer analyze info longest-tokens --source der_tag
      newspaper-explorer analyze info longest-tokens --source der_tag --top-n 50
      newspaper-explorer analyze info longest-tokens --source der_tag --tokenizer bert-base-german-cased
      newspaper-explorer analyze info longest-tokens --source der_tag --no-metadata
    """

    try:
        # Load config
        config = load_source_config(source)
        source_name = config.dataset_name

        output.header(f"LONGEST TEXTS BY TOKEN COUNT: {source_name}")

        # Get path to parquet file
        paths = get_source_paths(config)
        parquet_file = paths["output_file"]

        if not parquet_file.exists():
            output.error(f"Parquet file not found: {parquet_file}")
            output.info("Run parsing first:")
            output.info(f"  newspaper-explorer data parse --source {source}", muted=True)
            raise click.Abort()

        # Load data directly from parquet
        output.info(f"Loading data from: {parquet_file.name}")
        df = DataIngester.load_parquet(parquet_file)

        if len(df) == 0:
            output.error("Empty parquet file. Re-run parsing:")
            output.info(f"  newspaper-explorer data parse --source {source}", muted=True)
            raise click.Abort()

        # Get longest lines
        output.info(f"Finding top {top_n} longest texts...")
        output.info(f"Tokenizer: {tokenizer}", muted=True)

        longest_df = get_longest_lines_by_tokens(
            df,
            text_column=text_column,
            tokenizer_name=tokenizer,
            top_n=top_n,
        )

        # Display results
        output.section(f"TOP {top_n} LONGEST TEXTS")

        for i, row in enumerate(longest_df.iter_rows(named=True), 1):
            output.divider()
            output.key_value(f"#{i}", f"{row['token_count']} tokens")

            if show_metadata:
                if "date" in row:
                    output.key_value("Date", row["date"])
                if "newspaper_title" in row:
                    output.key_value("Title", row["newspaper_title"])
                if "page_number" in row:
                    output.key_value("Page", row["page_number"])

            # Show text with preview
            text = row["text"]
            if len(text) > TEXT_PREVIEW_LENGTH:
                output.info(f"{text[:TEXT_PREVIEW_LENGTH]}...", muted=True)
            else:
                output.info(text, muted=True)

        # Summary
        click.echo()  # Blank line
        token_counts = longest_df["token_count"].to_list()
        output.info(f"Token count range: {min(token_counts)} - {max(token_counts)}")

        if max(token_counts) >= MAX_TOKEN_LENGTH:
            output.warning(
                f"{sum(1 for t in token_counts if t >= MAX_TOKEN_LENGTH)} texts will be truncated at {MAX_TOKEN_LENGTH} tokens"
            )
        else:
            output.success(f"All texts fit within {MAX_TOKEN_LENGTH} token limit")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)
