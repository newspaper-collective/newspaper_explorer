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
            source_name = config.get("dataset_name", source)

            click.echo(f"\n{'='*80}")
            click.echo(f"SOURCE INFORMATION: {source_name}")
            click.echo(f"{'='*80}")

            # Show metadata
            metadata = config.get("metadata", {})
            if metadata:
                click.echo(f"\nNewspaper: {metadata.get('newspaper_title', 'N/A')}")
                click.echo(f"Years: {metadata.get('years_available', 'N/A')}")
                click.echo(f"Language: {metadata.get('language', 'N/A')}")
                if metadata.get("location"):
                    click.echo(f"Location: {metadata.get('location')}")

            # Get paths
            paths = get_source_paths(config)

            # Download/Extraction Status
            click.echo(f"\n{'='*80}")
            click.echo("DOWNLOAD & EXTRACTION STATUS")
            click.echo(f"{'='*80}")

            raw_dir = paths["raw_dir"]

            # Simple check: if raw_dir exists and has XML files, data is extracted
            if raw_dir.exists():
                xml_pattern = config.get("loading", {}).get("pattern", "**/fulltext/*.xml")
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
            xml_pattern = config.get("loading", {}).get("pattern", "**/fulltext/*.xml")

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
            dataset_name = config.get("dataset_name", source_name)
            data_type = config.get("data_type", "unknown")
            metadata = config.get("metadata", {})

            # Calculate total size
            parts = config.get("parts", [])
            total_bytes = sum(part.get("bytes", 0) for part in parts)
            total_gb = total_bytes / (1024**3)
            total_mb = total_bytes / (1024**2)

            if total_gb >= 1:
                size_str = f"~{total_gb:.1f} GB (compressed)"
            else:
                size_str = f"~{total_mb:.0f} MB (compressed)"

            click.echo(f"{dataset_name}")
            click.echo(
                f"  Description: {metadata.get('newspaper_title', 'N/A')} newspaper collection from Zenodo ({data_type} data)"
            )
            click.echo(f"  Newspaper: {metadata.get('newspaper_title', 'N/A')}")
            click.echo(f"  Years: {metadata.get('years_available', 'N/A')}")
            click.echo(f"  Language: {metadata.get('language', 'N/A')}")
            if metadata.get("location"):
                click.echo(f"  Location: {metadata.get('location')}")
            click.echo(f"  Parts: {len(parts)} (total size: {size_str})")
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

    # Register the command
    data_group.command("list-sources")(list_sources)
