"""
Newspaper Explorer - Main CLI Entry Point

Command-line interface for loading, processing, and analyzing historical newspaper
data from ALTO XML archives.

This module serves as the entry point for the newspaper-explorer CLI tool, which
provides two main command groups:

- data: Download, extract, parse, and preprocess newspaper archives
- analyze: Run analysis tasks on processed data (entities, topics, emotions, etc.)

Usage:
    newspaper-explorer data --help
    newspaper-explorer analyze --help
"""

import click

from newspaper_explorer.cli.analyze import analyze
from newspaper_explorer.cli.data.commands import data


@click.group()
def cli() -> None:
    """
    Newspaper Explorer - Explore and analyze historical newspaper data.

    This tool provides commands for managing newspaper data and running analyses:

    \b
    Data Management:
        newspaper-explorer data download --source <name>
        newspaper-explorer data parse --source <name>
        newspaper-explorer data info --source <name>

    \b
    Analysis:
        newspaper-explorer analyze <command> [options]

    For detailed help on any command group, use:
        newspaper-explorer data --help
        newspaper-explorer analyze --help
    """
    pass


# Register command groups
cli.add_command(data)
cli.add_command(analyze)


if __name__ == "__main__":
    cli(prog_name="newspaper-explorer")
