"""Data management CLI commands.

This module provides commands for downloading, loading, and processing
historical newspaper data.

Command structure:
    data text       - Text data pipeline (download, parse, aggregate)
    data images     - Image download and indexing
    data validation - Data quality checks
    data info       - Source status information (flat)
    data preprocess - Text preprocessing (flat)
    data list-sources, list-pipelines, etc. (flat)
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

    \b
    Text Pipeline:
        newspaper-explorer data text download --source <name> --all
        newspaper-explorer data text parse --source <name>
        newspaper-explorer data text aggregate --source <name>

    \b
    Image Pipeline:
        newspaper-explorer data images download --source <name>
        newspaper-explorer data images index --source <name>

    \b
    Preprocessing:
        newspaper-explorer data preprocess --source <name> --pipeline standard

    \b
    Information:
        newspaper-explorer data list-sources
        newspaper-explorer data info --source <name>

    \b
    Validation:
        newspaper-explorer data validation all --source <name>

    Use --help on any subcommand for more details.
    """
    pass


# --- Grouped commands ---

# Text pipeline: combines download/unpack/verify + parse/aggregate
@click.group(name="text")
def text_group() -> None:
    """Text data pipeline (download, parse, aggregate)."""
    pass


# Pull individual commands from their source modules into the text group
for _name, _cmd in download_group.commands.items():
    text_group.add_command(_cmd, _name)
for _name, _cmd in loading_group.commands.items():
    text_group.add_command(_cmd, _name)

data.add_command(text_group)

# Images: keep as-is
data.add_command(images_group)

# Validation: keep as-is
data.add_command(validation_group)

# --- Flat commands ---

# Info: register "status" as "info" at top level, plus list-sources and analysis commands
data.add_command(info_group.commands["status"], "info")
data.add_command(info_group.commands["list-sources"], "list-sources")
data.add_command(info_group.commands["analyze-chars"], "analyze-chars")
data.add_command(info_group.commands["analyze-tokens"], "analyze-tokens")
data.add_command(info_group.commands["longest-tokens"], "longest-tokens")

# Preprocessing: register commands flat
data.add_command(preprocessing_group.commands["preprocess"], "preprocess")
data.add_command(preprocessing_group.commands["preprocess-all"], "preprocess-all")
data.add_command(preprocessing_group.commands["list-pipelines"], "list-pipelines")
