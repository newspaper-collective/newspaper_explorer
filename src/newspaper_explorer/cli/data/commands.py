"""Data management CLI commands.

This module provides commands for downloading, loading, and processing
historical newspaper data.
"""

import click

from newspaper_explorer.cli.data.download import download_group
from newspaper_explorer.cli.data.images import images_group
from newspaper_explorer.cli.data.info import info_group
from newspaper_explorer.cli.data.loading import loading_group
from newspaper_explorer.cli.data.preprocessing import preprocessing_group
from newspaper_explorer.cli.data.validation import validation_group


@click.group()
def data() -> None:
    """
    Manage newspaper data (download, load, preprocess).

    This command group provides tools for the full data pipeline:
    - Download archives from Zenodo
    - Load and parse XML files to Parquet
    - Aggregate lines into text blocks
    - Preprocess text with various normalization methods

    Use --help on any subcommand for more details.
    """
    pass


# Register all command modules
data.add_command(info_group)
data.add_command(download_group)
data.add_command(images_group)
data.add_command(loading_group)
data.add_command(validation_group)
data.add_command(preprocessing_group)
