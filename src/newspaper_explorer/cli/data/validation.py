"""CLI commands for data validation."""

import logging
from pathlib import Path
from typing import Optional

import click

from newspaper_explorer.cli.utils import errors, output
from newspaper_explorer.cli.utils.options import min_image_size_option, source_option
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.download.images import ImageDownloader
from newspaper_explorer.data.processing.validation import (
    validate_alto_mets_relationship,
    verify_mets_completeness,
)
from newspaper_explorer.data.utils.validation import validate_images_in_directory

# Display limits for validation issue lists
MAX_ISSUES_TO_DISPLAY = 10
MAX_MISSING_FILES_DISPLAY = 10

# Full coverage percentage threshold
FULL_COVERAGE_PCT = 100


@click.group(name="validation")
def validation_group() -> None:
    """Data validation commands."""
    pass


@validation_group.command(name="alto-mets")
@source_option()
@click.option(
    "--save-report",
    type=click.Path(),
    help="Save orphaned/unlisted ALTO files to a report file",
)
def alto_mets_cmd(source: str, save_report: Optional[str]) -> None:
    """
    Validate ALTO-METS relationships for a source.

    Checks that each ALTO file:
    1. Has a parent METS file
    2. Is properly referenced in that METS file

    This helps identify orphaned ALTO files or data integrity issues.

    \b
    Examples:
      newspaper-explorer data validation alto-mets --source der_tag
      newspaper-explorer data validation alto-mets --source der_tag --save-report issues.txt
    """
    try:
        output.header(f"VALIDATE ALTO-METS: {source.upper()}")

        output.section("VALIDATING")
        output.info("Checking ALTO-METS relationships...")
        result = validate_alto_mets_relationship(source)

        # Display summary
        output.section("RESULTS")
        output.key_value("Total ALTO files", f"{result['total_alto_files']:,}")
        output.key_value("ALTO with valid METS", f"{result['alto_with_mets']:,}")
        output.key_value("ALTO without parent METS", f"{result['alto_without_mets']:,}")
        output.key_value("ALTO not listed in METS", f"{result['alto_not_in_mets']:,}")

        # Show issues if any
        if result["alto_without_mets"] > 0 or result["alto_not_in_mets"] > 0:
            if result["alto_without_mets"] > 0:
                output.section(f"ALTO WITHOUT PARENT METS ({result['alto_without_mets']} files)")
                for i, path in enumerate(result["orphaned_alto_list"][:MAX_ISSUES_TO_DISPLAY], 1):
                    output.info(f"{i}. {path}", muted=True)

                if len(result["orphaned_alto_list"]) > MAX_ISSUES_TO_DISPLAY:
                    remaining = len(result["orphaned_alto_list"]) - MAX_ISSUES_TO_DISPLAY
                    output.info(f"... and {remaining:,} more", muted=True)

            if result["alto_not_in_mets"] > 0:
                output.section(f"ALTO NOT REFERENCED IN METS ({result['alto_not_in_mets']} files)")
                for i, path in enumerate(result["unlisted_alto_list"][:MAX_ISSUES_TO_DISPLAY], 1):
                    output.info(f"{i}. {path}", muted=True)

                if len(result["unlisted_alto_list"]) > MAX_ISSUES_TO_DISPLAY:
                    remaining = len(result["unlisted_alto_list"]) - MAX_ISSUES_TO_DISPLAY
                    output.info(f"... and {remaining:,} more", muted=True)

            # Save report if requested
            if save_report:
                report_path = Path(save_report)
                with report_path.open("w") as f:
                    f.write("ALTO-METS RELATIONSHIP VALIDATION REPORT\n")
                    f.write("=" * 60 + "\n\n")

                    if result["orphaned_alto_list"]:
                        f.write(
                            f"ALTO files without parent METS ({len(result['orphaned_alto_list'])}):\n"
                        )
                        for path in result["orphaned_alto_list"]:
                            f.write(f"  {path}\n")
                        f.write("\n")

                    if result["unlisted_alto_list"]:
                        f.write(
                            f"ALTO files not referenced in METS ({len(result['unlisted_alto_list'])}):\n"
                        )
                        for path in result["unlisted_alto_list"]:
                            f.write(f"  {path}\n")

                click.echo()
                output.success(f"Report saved to: {report_path}")

            click.echo()
            output.warning("Validation found issues - see details above")
        else:
            click.echo()
            output.success("All ALTO files have valid METS relationships!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@validation_group.command(name="images")
@source_option()
@min_image_size_option(default=1024)
@click.option(
    "--save-report",
    type=click.Path(),
    default=None,
    help="Save list of invalid images to file",
)
def images_cmd(source: str, min_size: int, save_report: Optional[str]) -> None:
    """
    Validate downloaded images for a source.

    Checks all downloaded images in data/raw/{source}/images/ to ensure:
    - Files are valid image formats that can be opened
    - Files meet minimum size requirements (not corrupted/truncated)

    Invalid images are reported and can be saved to a file for review.

    \b
    Examples:
      newspaper-explorer data validation images --source der_tag
      newspaper-explorer data validation images --source der_tag --min-size 5000
      newspaper-explorer data validation images --source der_tag --save-report invalid_images.txt
    """

    # Configure logging
    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    try:
        output.header(f"VALIDATE IMAGES: {source.upper()}")

        # Show configuration
        output.section("CONFIGURATION")
        output.key_value("Minimum image size", f"{min_size} bytes")

        # Get images directory for this source
        downloader = ImageDownloader(source_name=source)
        images_dir = downloader.images_dir

        if not images_dir.exists():
            output.error(f"Images directory not found: {images_dir}")
            output.warning("Run 'newspaper-explorer data images download' first")
            return

        output.section("VALIDATING")
        output.info(f"Checking images in {images_dir}...")
        result = validate_images_in_directory(images_dir, min_size_bytes=min_size)

        output.section("RESULTS")
        output.key_value("Total images checked", f"{result['total']:,}")
        output.key_value("Valid images", f"{result['valid']:,}")
        output.key_value("Invalid images", f"{result['invalid']:,}")

        if result["invalid"] > 0:
            output.section("INVALID IMAGES")
            for i, (path, error) in enumerate(result["invalid_list"][:MAX_ISSUES_TO_DISPLAY], 1):
                output.info(f"{i}. {path}: {error}", muted=True)

            if len(result["invalid_list"]) > MAX_ISSUES_TO_DISPLAY:
                remaining = len(result["invalid_list"]) - MAX_ISSUES_TO_DISPLAY
                output.info(f"... and {remaining} more", muted=True)

            # Save report if requested
            if save_report:
                report_path = Path(save_report)
                with report_path.open("w") as f:
                    f.write("INVALID IMAGES REPORT\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Source: {source}\n")
                    f.write(f"Total invalid: {result['invalid']}\n\n")
                    for path, error in result["invalid_list"]:
                        f.write(f"{path}\t{error}\n")

                click.echo()
                output.success(f"Invalid images list saved to: {report_path}")

            click.echo()
            output.warning("Some images are invalid. Consider re-downloading them.")
        else:
            click.echo()
            output.success("All images are valid!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@validation_group.command(name="completeness")
@source_option()
@click.option(
    "--save-report",
    type=click.Path(),
    default=None,
    help="Save list of missing files to a report file",
)
def completeness_cmd(source: str, save_report: Optional[str]) -> None:
    """
    Validate completeness of downloaded files against METS references.

    Checks that all images and ALTO XML files referenced in METS files
    have been successfully downloaded. Reports missing files.

    \b
    Examples:
      newspaper-explorer data validation completeness --source der_tag
      newspaper-explorer data validation completeness --source der_tag --save-report missing_files.txt
    """

    try:
        output.header(f"CHECK COMPLETENESS: {source.upper()}")

        output.section("VALIDATING")
        output.info("Checking file completeness against METS references...")
        result = verify_mets_completeness(source)

        output.section("RESULTS")
        output.key_value("METS files checked", f"{result['mets_files_checked']:,}")

        # Images section
        output.divider()
        output.key_value("Images expected", f"{result['images_expected']:,}")
        output.key_value("Images found", f"{result['images_found']:,}")
        output.key_value("Images missing", f"{result['images_missing']:,}")
        if result["images_expected"] > 0:
            coverage = result["images_found"] / result["images_expected"] * FULL_COVERAGE_PCT
            output.key_value("Images coverage", f"{coverage:.1f}%")

        # ALTO section
        output.divider()
        output.key_value("ALTO files expected", f"{result['alto_expected']:,}")
        output.key_value("ALTO files found", f"{result['alto_found']:,}")
        output.key_value("ALTO files missing", f"{result['alto_missing']:,}")
        if result["alto_expected"] > 0:
            coverage = result["alto_found"] / result["alto_expected"] * FULL_COVERAGE_PCT
            output.key_value("ALTO coverage", f"{coverage:.1f}%")

        # Show sample of missing files
        if result["images_missing"] > 0:
            output.section(
                f"MISSING IMAGES (showing {min(MAX_MISSING_FILES_DISPLAY, len(result['missing_images_list']))} of {len(result['missing_images_list'])})"
            )
            for i, path in enumerate(result["missing_images_list"][:MAX_MISSING_FILES_DISPLAY], 1):
                output.info(f"{i}. {path}", muted=True)

            if len(result["missing_images_list"]) > MAX_MISSING_FILES_DISPLAY:
                remaining = len(result["missing_images_list"]) - MAX_MISSING_FILES_DISPLAY
                output.info(f"... and {remaining:,} more", muted=True)

        if result["alto_missing"] > 0:
            output.section(
                f"MISSING ALTO FILES (showing {min(MAX_MISSING_FILES_DISPLAY, len(result['missing_alto_list']))} of {len(result['missing_alto_list'])})"
            )
            for i, path in enumerate(result["missing_alto_list"][:MAX_MISSING_FILES_DISPLAY], 1):
                output.info(f"{i}. {path}", muted=True)

            if len(result["missing_alto_list"]) > MAX_MISSING_FILES_DISPLAY:
                remaining = len(result["missing_alto_list"]) - MAX_MISSING_FILES_DISPLAY
                output.info(f"... and {remaining:,} more", muted=True)

        # Save report if requested
        if save_report:
            report_path = Path(save_report)
            with report_path.open("w") as f:
                f.write("FILE COMPLETENESS REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Source: {source}\n\n")
                f.write("MISSING IMAGES\n")
                f.write(f"Total: {result['images_missing']}\n\n")
                for path in result["missing_images_list"]:
                    f.write(f"{path}\n")
                f.write("\nMISSING ALTO FILES\n")
                f.write(f"Total: {result['alto_missing']}\n\n")
                for path in result["missing_alto_list"]:
                    f.write(f"{path}\n")

            click.echo()
            output.success(f"Report saved to: {report_path}")

        # Summary
        click.echo()
        total_missing = result["images_missing"] + result["alto_missing"]
        if total_missing > 0:
            output.warning(f"{total_missing:,} files are missing!")
        else:
            output.success("All referenced files are present!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)
