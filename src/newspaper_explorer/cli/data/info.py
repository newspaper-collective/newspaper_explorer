"""Info commands for the data CLI."""

from pathlib import Path

import click
from natsort import natsorted


def register_info_commands(data_group):
    """Register all info-related commands to the data group."""

    @data_group.command()
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    def info(source):
        """
        Show comprehensive status information for a source.

        Displays download status, extraction status, XML file counts,
        parsed data status, and processing coverage for a newspaper source.

        \b
        Examples:
          newspaper-explorer data info --source der_tag
        """
        import polars as pl

        from newspaper_explorer.utils.sources import get_source_paths, load_source_config

        try:
            # Load config
            config = load_source_config(source)
            source_name = config.dataset_name

            click.echo(f"\n{'='*80}")
            click.echo(f"SOURCE INFORMATION: {source_name}")
            click.echo(f"{'='*80}")

            # Show metadata
            if config.metadata:
                click.echo(f"\nNewspaper: {config.metadata.newspaper_title or 'N/A'}")
                click.echo(f"Years: {config.metadata.years_available or 'N/A'}")
                click.echo(f"Language: {config.metadata.language or 'N/A'}")
                if config.metadata.location:
                    click.echo(f"Location: {config.metadata.location}")

            # Get paths
            paths = get_source_paths(config)

            # Download/Extraction Status
            click.echo(f"\n{'='*80}")
            click.echo("DOWNLOAD & EXTRACTION STATUS")
            click.echo(f"{'='*80}")

            raw_dir = paths["raw_dir"]

            # Simple check: if raw_dir exists and has XML files, data is extracted
            if raw_dir.exists():
                xml_pattern = config.loading.pattern if config.loading else "**/fulltext/*.xml"
                xml_files = natsorted(raw_dir.glob(xml_pattern))

                if len(xml_files) > 0:
                    click.echo(f"Status: ✓ Data extracted and ready")
                    click.echo(f"Location: {raw_dir}")
                    click.echo(f"XML files: {len(xml_files):,}")
                else:
                    click.echo(f"Status: ✗ Directory exists but no XML files found")
                    click.echo(f"Location: {raw_dir}")
                    click.echo(f"\n⚠ Data may not be properly extracted. Run:")
                    click.echo(f"  newspaper-explorer data unpack --source {source}")
            else:
                click.echo(f"Status: ✗ Not extracted")
                click.echo(f"Expected location: {raw_dir}")
                click.echo(f"\n⚠ No data found. Run:")
                click.echo(f"  newspaper-explorer data download --source {source}")
                click.echo(f"  newspaper-explorer data unpack --source {source}")

            # XML Files Status
            click.echo(f"\n{'='*80}")
            click.echo("RAW XML FILES")
            click.echo(f"{'='*80}")

            raw_dir = paths["raw_dir"]
            xml_pattern = config.loading.pattern if config.loading else "**/fulltext/*.xml"

            click.echo(f"Directory: {raw_dir}")
            click.echo(f"Pattern: {xml_pattern}")

            if raw_dir.exists():
                xml_files = natsorted(raw_dir.glob(xml_pattern))
                click.echo(f"XML files found: {len(xml_files):,}")
            else:
                click.echo(f"XML files found: 0 (directory not found)")
                click.echo(f"\n⚠ No XML files. Run:")
                click.echo(f"  newspaper-explorer data download --source {source}")
                click.echo(f"  newspaper-explorer data unpack --source {source}")

            # Parsed Data Status
            click.echo(f"\n{'='*80}")
            click.echo("PARSED DATA (Parquet)")
            click.echo(f"{'='*80}")

            output_file = paths["output_file"]
            click.echo(f"Location: {output_file}")

            if output_file.exists():
                df = pl.read_parquet(output_file)
                unique_files = df["filename"].n_unique()
                total_lines = len(df)

                click.echo(f"Status: ✓ Exists")
                click.echo(f"Total lines: {total_lines:,}")
                click.echo(f"Files parsed: {unique_files:,}")

                # Calculate coverage if we have XML files
                if raw_dir.exists():
                    xml_files = natsorted(raw_dir.glob(xml_pattern))
                    if len(xml_files) > 0:
                        coverage_pct = (unique_files / len(xml_files)) * 100
                        click.echo(
                            f"Coverage: {coverage_pct:.1f}% ({unique_files}/{len(xml_files)})"
                        )

                        if unique_files < len(xml_files):
                            remaining = len(xml_files) - unique_files
                            click.echo(f"\n⚠ {remaining:,} XML files not yet parsed. Run:")
                            click.echo(f"  newspaper-explorer data parse --source {source}")
                        else:
                            click.echo(f"\n✓ All XML files parsed!")

                # Show date range
                if "date" in df.columns and len(df) > 0:
                    min_date = df["date"].min()
                    max_date = df["date"].max()
                    click.echo(f"Date range: {min_date} to {max_date}")

                # Size info
                size_mb = output_file.stat().st_size / (1024 * 1024)
                click.echo(f"File size: {size_mb:.1f} MB")
            else:
                click.echo(f"Status: ✗ Not found")
                click.echo(f"\n⚠ No parsed data. Run:")
                click.echo(f"  newspaper-explorer data parse --source {source}")

            # Aggregated Data Status
            click.echo(f"\n{'='*80}")
            click.echo("AGGREGATED TEXT BLOCKS")
            click.echo(f"{'='*80}")

            textblocks_path = (
                Path("data") / "processed" / source_name / "text" / "textblocks.parquet"
            )
            click.echo(f"Location: {textblocks_path}")

            if textblocks_path.exists():
                df = pl.read_parquet(textblocks_path)
                click.echo(f"Status: ✓ Exists")
                click.echo(f"Text blocks: {len(df):,}")

                size_mb = textblocks_path.stat().st_size / (1024 * 1024)
                click.echo(f"File size: {size_mb:.1f} MB")

                click.echo(f"\n✓ Ready for preprocessing!")
            else:
                click.echo(f"Status: ✗ Not found")
                if output_file.exists():
                    click.echo(f"\n⚠ Parsed data exists but not aggregated. Run:")
                    click.echo(f"  newspaper-explorer data aggregate --source {source}")

            # Image Download Status
            click.echo(f"\n{'='*80}")
            click.echo("PAGE IMAGES")
            click.echo(f"{'='*80}")

            try:
                from newspaper_explorer.data.download.images import ImageDownloader

                image_downloader = ImageDownloader(source_name=source)
                image_status = image_downloader.get_download_status()

                click.echo(f"Location: {image_status['images_dir']}")

                if image_status["images_dir_exists"]:
                    click.echo(f"Status: ✓ Directory exists")
                    click.echo(f"Images downloaded: {image_status['images_downloaded']:,}")
                    click.echo(
                        f"Images expected: {image_status['total_images_expected']:,} (from {image_status['mets_files']} METS files)"
                    )

                    if image_status["total_images_expected"] > 0:
                        coverage = image_status["coverage_pct"]
                        click.echo(f"Coverage: {coverage:.1f}%")

                        if coverage < 100:
                            missing = (
                                image_status["total_images_expected"]
                                - image_status["images_downloaded"]
                            )
                            click.echo(f"\n⚠ {missing:,} images not yet downloaded. Run:")
                            click.echo(
                                f"  newspaper-explorer data download-images --source {source}"
                            )
                        else:
                            click.echo(f"\n✓ All images downloaded!")
                else:
                    click.echo(f"Status: ✗ Not found")
                    if image_status["total_images_expected"] > 0:
                        click.echo(
                            f"Expected images: {image_status['total_images_expected']:,} (from {image_status['mets_files']} METS files)"
                        )
                        click.echo(f"\n⚠ No images downloaded. Run:")
                        click.echo(f"  newspaper-explorer data download-images --source {source}")
            except Exception as e:
                click.echo(f"Status: ⚠ Could not determine image status")
                click.echo(f"Error: {e}")

            click.echo(f"\n{'='*80}\n")

        except FileNotFoundError as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()
        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()

    def list_sources():
        """
        List all available data sources.

        Shows all newspaper sources with their metadata and download information.
        This command lists the available source configurations, not the individual
        data parts within a source.

        \b
        Examples:
          newspaper-explorer data list-sources
        """
        from newspaper_explorer.utils.sources import list_available_sources, load_source_config

        sources = list_available_sources()

        if not sources:
            click.echo("\nNo data sources found in data/sources/")
            return

        click.echo(f"\n{'Available Data Sources'}")
        click.echo("=" * 90)
        click.echo()

        for source_name in sources:
            config = load_source_config(source_name)
            dataset_name = config.dataset_name
            data_type = config.data_type
            metadata = config.metadata

            # Get human-readable sizes from parts
            sizes = [part.size for part in config.parts if part.size]
            size_str = ", ".join(sizes) if sizes else "unknown"

            click.echo(f"{dataset_name}")
            click.echo(
                f"  Description: {metadata.newspaper_title} newspaper collection from Zenodo ({data_type} data)"
            )
            click.echo(f"  Newspaper: {metadata.newspaper_title}")
            click.echo(f"  Years: {metadata.years_available}")
            click.echo(f"  Language: {metadata.language}")
            if metadata.location:
                click.echo(f"  Location: {metadata.location}")
            click.echo(f"  Parts: {len(config.parts)} (sizes: {size_str})")
            click.echo()

        click.echo("=" * 90)
        click.echo()
        click.echo()
        click.echo("Tip: Use 'newspaper-explorer data info --source <name>' to see status")

    @data_group.command("check-completeness")
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--save-report",
        type=click.Path(),
        default=None,
        help="Save list of missing files to a report file",
    )
    def check_completeness(source, save_report):
        """
        Check completeness of downloaded files against METS references.

        Checks that all images and ALTO XML files referenced in METS files
        have been successfully downloaded. Reports missing files.

        \b
        Examples:
          newspaper-explorer data check-completeness --source der_tag
          newspaper-explorer data check-completeness --source der_tag --save-report missing_files.txt
        """
        import logging

        from newspaper_explorer.data.utils.validation import verify_mets_completeness

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        try:
            click.echo(f"Verifying completeness for source: {source}\n")

            result = verify_mets_completeness(source)

            click.echo("\n" + "=" * 60)
            click.echo("COMPLETENESS VERIFICATION SUMMARY")
            click.echo("=" * 60)
            click.echo(f"METS files checked: {result['mets_files_checked']}\n")

            click.echo("IMAGES:")
            click.echo(f"  Expected:  {result['images_expected']}")
            click.echo(f"  Found:     {result['images_found']}")
            click.echo(f"  Missing:   {result['images_missing']}")
            if result["images_expected"] > 0:
                coverage = result["images_found"] / result["images_expected"] * 100
                click.echo(f"  Coverage:  {coverage:.1f}%")

            click.echo("\nALTO XML FILES:")
            click.echo(f"  Expected:  {result['alto_expected']}")
            click.echo(f"  Found:     {result['alto_found']}")
            click.echo(f"  Missing:   {result['alto_missing']}")
            if result["alto_expected"] > 0:
                coverage = result["alto_found"] / result["alto_expected"] * 100
                click.echo(f"  Coverage:  {coverage:.1f}%")

            click.echo("=" * 60)

            # Show sample of missing files
            if result["images_missing"] > 0:
                click.echo("\nSample of missing images (first 10):")
                for path in result["missing_images_list"][:10]:
                    click.echo(f"  - {path}")
                if len(result["missing_images_list"]) > 10:
                    remaining = len(result["missing_images_list"]) - 10
                    click.echo(f"  ... and {remaining} more")

            if result["alto_missing"] > 0:
                click.echo("\nSample of missing ALTO files (first 10):")
                for path in result["missing_alto_list"][:10]:
                    click.echo(f"  - {path}")
                if len(result["missing_alto_list"]) > 10:
                    remaining = len(result["missing_alto_list"]) - 10
                    click.echo(f"  ... and {remaining} more")

            # Save report if requested
            if save_report:
                with open(save_report, "w") as f:
                    f.write("# File Completeness Report\n")
                    f.write(f"# Source: {source}\n\n")

                    f.write("## Missing Images\n")
                    f.write(f"# Total: {result['images_missing']}\n")
                    for path in result["missing_images_list"]:
                        f.write(f"{path}\n")

                    f.write("\n## Missing ALTO Files\n")
                    f.write(f"# Total: {result['alto_missing']}\n")
                    for path in result["missing_alto_list"]:
                        f.write(f"{path}\n")

                click.echo(f"\nMissing files report saved to: {save_report}")

            # Summary message
            total_missing = result["images_missing"] + result["alto_missing"]
            if total_missing > 0:
                click.echo(
                    f"\n⚠ Warning: {total_missing} files are missing!",
                    err=True,
                )
            else:
                click.echo("\n✓ All referenced files are present!")

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()

    @data_group.command()
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of rows to sample for analysis (default: 50000)",
    )
    @click.option(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column (default: text)",
    )
    def analyze_chars(source, sample_size, text_column):
        """
        Analyze character lengths in newspaper text data.

        This command samples texts from the parsed Parquet file and computes
        comprehensive statistics about character lengths. Useful for understanding
        text length distribution and comparing with token-based analysis.

        \b
        Examples:
          # Analyze character lengths for der_tag
          newspaper-explorer data analyze-chars --source der_tag

          # Custom sample size
          newspaper-explorer data analyze-chars --source der_tag --sample-size 100000
        """
        from newspaper_explorer.data.loading.loader import DataLoader
        from newspaper_explorer.data.utils.text import analyze_character_lengths
        from newspaper_explorer.utils.sources import load_source_config, get_source_paths

        try:
            # Load config
            config = load_source_config(source)
            source_name = config.dataset_name

            click.echo(f"\n{'='*80}")
            click.echo(f"CHARACTER LENGTH ANALYSIS: {source_name}")
            click.echo(f"{'='*80}")

            # Get path to parquet file
            paths = get_source_paths(config)
            parquet_file = paths["output_file"]

            if not parquet_file.exists():
                click.echo(f"\n✗ Parquet file not found: {parquet_file}", err=True)
                click.echo(f"Run parsing first:", err=True)
                click.echo(f"  newspaper-explorer data parse --source {source}")
                raise click.Abort()

            # Load data directly from parquet (skip XML scanning)
            click.echo(f"\nLoading data from: {parquet_file.name}")
            df = DataLoader.load_parquet(parquet_file)

            if len(df) == 0:
                click.echo("\n✗ Empty parquet file. Re-run parsing:", err=True)
                click.echo(f"  newspaper-explorer data parse --source {source}")
                raise click.Abort()

            # Analyze character lengths
            click.echo(f"\nAnalyzing character lengths (sample size: {sample_size:,})...")

            stats = analyze_character_lengths(
                df,
                text_column=text_column,
                sample_size=sample_size,
            )

            # Display results
            click.echo(f"\n{'='*80}")
            click.echo("CHARACTER LENGTH STATISTICS")
            click.echo(f"{'='*80}")

            click.echo(f"\nDataset:")
            click.echo(f"  Total rows:       {stats['total_rows']:,}")
            click.echo(f"  Sample analyzed:  {stats['sample_size']:,}")

            click.echo(f"\nCharacter counts:")
            click.echo(f"  Min:              {stats['min_chars']}")
            click.echo(f"  Max:              {stats['max_chars']}")
            click.echo(f"  Mean:             {stats['mean_chars']:.1f}")
            click.echo(f"  Median:           {stats['median_chars']}")

            click.echo(f"\nPercentiles:")
            click.echo(f"  90th:             {stats['p90_chars']} characters")
            click.echo(f"  95th:             {stats['p95_chars']} characters")
            click.echo(f"  99th:             {stats['p99_chars']} characters")

            dist = stats["distribution"]
            total = stats["sample_size"]
            click.echo(f"\nDistribution:")
            click.echo(
                f"  ≤   50 chars:     {dist['under_50']:,} ({100*dist['under_50']/total:.1f}%)"
            )
            click.echo(
                f"  ≤  100 chars:     {dist['under_100']:,} ({100*dist['under_100']/total:.1f}%)"
            )
            click.echo(
                f"  ≤  200 chars:     {dist['under_200']:,} ({100*dist['under_200']/total:.1f}%)"
            )
            click.echo(
                f"  ≤  500 chars:     {dist['under_500']:,} ({100*dist['under_500']/total:.1f}%)"
            )
            click.echo(
                f"  ≤ 1000 chars:     {dist['under_1000']:,} ({100*dist['under_1000']/total:.1f}%)"
            )

            # Show longest examples
            click.echo(f"\n{'='*80}")
            click.echo("LONGEST TEXT EXAMPLES")
            click.echo(f"{'='*80}")

            for i, (char_count, text) in enumerate(stats["longest_examples"][:3], 1):
                click.echo(f"\n{i}. {char_count} characters:")
                preview = text[:200] + "..." if len(text) > 200 else text
                click.echo(f"   {preview}")

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()

    @data_group.command()
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of rows to sample for analysis (default: 50000)",
    )
    @click.option(
        "--tokenizer",
        type=str,
        default="deepset/gbert-large",
        help="HuggingFace tokenizer to use (default: deepset/gbert-large)",
    )
    @click.option(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column (default: text)",
    )
    def analyze_tokens(source, sample_size, tokenizer, text_column):
        """
        Analyze token lengths in newspaper text data.

        This command samples texts from the parsed Parquet file and computes
        comprehensive statistics about token lengths using a BERT tokenizer.
        Useful for understanding padding requirements and estimating speedup
        potential from dynamic padding.

        \b
        Examples:
          # Analyze token lengths for der_tag
          newspaper-explorer data analyze-tokens --source der_tag

          # Custom sample size
          newspaper-explorer data analyze-tokens --source der_tag --sample-size 100000

          # Use different tokenizer
          newspaper-explorer data analyze-tokens --source der_tag --tokenizer bert-base-german-cased
        """
        from newspaper_explorer.data.loading.loader import DataLoader
        from newspaper_explorer.data.utils.text import analyze_token_lengths
        from newspaper_explorer.utils.sources import load_source_config, get_source_paths

        try:
            # Load config
            config = load_source_config(source)
            source_name = config.dataset_name

            click.echo(f"\n{'='*80}")
            click.echo(f"TOKEN LENGTH ANALYSIS: {source_name}")
            click.echo(f"{'='*80}")

            # Get path to parquet file
            paths = get_source_paths(config)
            parquet_file = paths["output_file"]

            if not parquet_file.exists():
                click.echo(f"\n✗ Parquet file not found: {parquet_file}", err=True)
                click.echo(f"Run parsing first:", err=True)
                click.echo(f"  newspaper-explorer data parse --source {source}")
                raise click.Abort()

            # Load data directly from parquet (skip XML scanning)
            click.echo(f"\nLoading data from: {parquet_file.name}")
            df = DataLoader.load_parquet(parquet_file)

            if len(df) == 0:
                click.echo("\n✗ Empty parquet file. Re-run parsing:", err=True)
                click.echo(f"  newspaper-explorer data parse --source {source}")
                raise click.Abort()

            # Analyze token lengths
            click.echo(f"\nAnalyzing token lengths (sample size: {sample_size:,})...")
            click.echo(f"Tokenizer: {tokenizer}")

            stats = analyze_token_lengths(
                df,
                text_column=text_column,
                tokenizer_name=tokenizer,
                sample_size=sample_size,
            )

            # Display results
            click.echo(f"\n{'='*80}")
            click.echo("TOKEN LENGTH STATISTICS")
            click.echo(f"{'='*80}")

            click.echo(f"\nDataset:")
            click.echo(f"  Total rows:       {stats['total_rows']:,}")
            click.echo(f"  Sample analyzed:  {stats['sample_size']:,}")

            click.echo(f"\nToken counts:")
            click.echo(f"  Min:              {stats['min_tokens']}")
            click.echo(f"  Max:              {stats['max_tokens']}")
            click.echo(f"  Mean:             {stats['mean_tokens']:.1f}")
            click.echo(f"  Median:           {stats['median_tokens']}")

            click.echo(f"\nPercentiles:")
            click.echo(f"  90th:             {stats['p90_tokens']} tokens")
            click.echo(f"  95th:             {stats['p95_tokens']} tokens")
            click.echo(f"  99th:             {stats['p99_tokens']} tokens")

            dist = stats["distribution"]
            total = stats["sample_size"]
            click.echo(f"\nDistribution:")
            click.echo(
                f"  ≤  50 tokens:     {dist['under_50']:,} ({100*dist['under_50']/total:.1f}%)"
            )
            click.echo(
                f"  ≤ 100 tokens:     {dist['under_100']:,} ({100*dist['under_100']/total:.1f}%)"
            )
            click.echo(
                f"  ≤ 200 tokens:     {dist['under_200']:,} ({100*dist['under_200']/total:.1f}%)"
            )
            click.echo(
                f"  ≤ 300 tokens:     {dist['under_300']:,} ({100*dist['under_300']/total:.1f}%)"
            )

            if dist["at_max_length"] > 0:
                click.echo(
                    f"  = 512 tokens:     {dist['at_max_length']:,} "
                    f"({stats['truncated_percent']:.2f}%) [truncated]"
                )

            click.echo(f"\n{'='*80}")
            click.echo("PADDING ANALYSIS")
            click.echo(f"{'='*80}")

            click.echo(f"\nWith static padding to 512 tokens:")
            click.echo(f"  Average tokens used:    {stats['mean_tokens']:.1f}")
            click.echo(f"  Wasted compute:         {stats['wasted_padding_percent']:.1f}%")
            click.echo(f"\nWith dynamic padding:")
            click.echo(f"  Expected speedup:       {stats['expected_speedup']:.1f}x")

            if stats["expected_speedup"] > 2.0:
                click.echo("\n✓ Dynamic padding will provide significant speedup!")
            elif stats["expected_speedup"] > 1.5:
                click.echo("\n✓ Dynamic padding will provide moderate speedup.")
            else:
                click.echo("\nℹ Dynamic padding may provide limited benefit.")

            # Show longest examples
            click.echo(f"\n{'='*80}")
            click.echo("LONGEST TEXT EXAMPLES")
            click.echo(f"{'='*80}")

            for i, (token_count, text) in enumerate(stats["longest_examples"][:3], 1):
                click.echo(f"\n{i}. {token_count} tokens:")
                preview = text[:200] + "..." if len(text) > 200 else text
                click.echo(f"   {preview}")

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()

    @data_group.command()
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--top-n",
        type=int,
        default=10,
        help="Number of longest lines to retrieve (default: 10)",
    )
    @click.option(
        "--tokenizer",
        type=str,
        default="deepset/gbert-large",
        help="HuggingFace tokenizer to use (default: deepset/gbert-large)",
    )
    @click.option(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column (default: text)",
    )
    @click.option(
        "--show-metadata/--no-metadata",
        default=True,
        help="Show metadata columns (default: True)",
    )
    def longest_tokens(source, top_n, tokenizer, text_column, show_metadata):
        """
        Show the longest text lines by token count.

        This command finds texts with the highest token counts using a tokenizer.
        Useful for identifying edge cases, texts that will be truncated, and
        understanding the distribution of long texts.

        \b
        Examples:
          # Show top 10 longest texts
          newspaper-explorer data longest-tokens --source der_tag

          # Show more results
          newspaper-explorer data longest-tokens --source der_tag --top-n 50

          # Use different tokenizer
          newspaper-explorer data longest-tokens --source der_tag --tokenizer bert-base-german-cased

          # Hide metadata columns
          newspaper-explorer data longest-tokens --source der_tag --no-metadata
        """
        from newspaper_explorer.data.loading.loader import DataLoader
        from newspaper_explorer.data.utils.text import get_longest_lines_by_tokens
        from newspaper_explorer.utils.sources import load_source_config, get_source_paths

        try:
            # Load config
            config = load_source_config(source)
            source_name = config.dataset_name

            click.echo(f"\n{'='*80}")
            click.echo(f"LONGEST TEXTS BY TOKEN COUNT: {source_name}")
            click.echo(f"{'='*80}")

            # Get path to parquet file
            paths = get_source_paths(config)
            parquet_file = paths["output_file"]

            if not parquet_file.exists():
                click.echo(f"\n✗ Parquet file not found: {parquet_file}", err=True)
                click.echo(f"Run parsing first:", err=True)
                click.echo(f"  newspaper-explorer data parse --source {source}")
                raise click.Abort()

            # Load data directly from parquet
            click.echo(f"\nLoading data from: {parquet_file.name}")
            df = DataLoader.load_parquet(parquet_file)

            if len(df) == 0:
                click.echo("\n✗ Empty parquet file. Re-run parsing:", err=True)
                click.echo(f"  newspaper-explorer data parse --source {source}")
                raise click.Abort()

            # Get longest lines
            click.echo(f"\nFinding top {top_n} longest texts...")
            click.echo(f"Tokenizer: {tokenizer}")

            longest_df = get_longest_lines_by_tokens(
                df,
                text_column=text_column,
                tokenizer_name=tokenizer,
                top_n=top_n,
            )

            # Display results
            click.echo(f"\n{'='*80}")
            click.echo(f"TOP {top_n} LONGEST TEXTS")
            click.echo(f"{'='*80}")

            # Select columns to display
            if show_metadata:
                display_cols = ["token_count", "text", "date", "newspaper_title", "page_number"]
                # Only show columns that exist
                display_cols = [col for col in display_cols if col in longest_df.columns]
            else:
                display_cols = ["token_count", "text"]

            for i, row in enumerate(longest_df.iter_rows(named=True), 1):
                click.echo(f"\n{i}. {row['token_count']} tokens")

                if show_metadata:
                    if "date" in row:
                        click.echo(f"   Date: {row['date']}")
                    if "newspaper_title" in row:
                        click.echo(f"   Title: {row['newspaper_title']}")
                    if "page_number" in row:
                        click.echo(f"   Page: {row['page_number']}")

                # Show text with preview
                text = row["text"]
                if len(text) > 300:
                    click.echo(f"   Text: {text[:300]}...")
                else:
                    click.echo(f"   Text: {text}")

            # Summary
            click.echo(f"\n{'='*80}")
            token_counts = longest_df["token_count"].to_list()
            click.echo(f"Token count range: {min(token_counts)} - {max(token_counts)}")

            if max(token_counts) >= 512:
                click.echo(
                    f"\n⚠ Warning: {sum(1 for t in token_counts if t >= 512)} texts will be truncated at 512 tokens"
                )
            else:
                click.echo(f"\n✓ All texts fit within 512 token limit")

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()

    # Register the command
    data_group.command("list-sources")(list_sources)
