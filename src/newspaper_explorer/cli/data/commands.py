"""Data management CLI commands.

This module provides commands for downloading, loading, and processing
historical newspaper data from various sources.
"""

import click

from .download import register_download_commands
from .images import register_image_commands
from .info import register_info_commands
from .loading import register_loading_commands
from .preprocessing import register_preprocessing_commands


@click.group()
def data():
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
register_info_commands(data)
register_download_commands(data)
register_image_commands(data)
register_loading_commands(data)
register_preprocessing_commands(data)
