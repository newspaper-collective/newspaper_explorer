"""
Newspaper Explorer - Main CLI Entry Point

A tool for exploring and analyzing historical newspaper data.
"""

import click

from newspaper_explorer.cli.analyze import analyze
from newspaper_explorer.cli.data import data


@click.group()
def cli() -> None:
    """
    Newspaper Explorer - Explore and analyze historical newspaper data.
    """
    pass


# Register command groups
cli.add_command(data)
cli.add_command(analyze)


def main() -> None:
    """Main entry point for the CLI."""
    cli(prog_name="newspaper-explorer")


if __name__ == "__main__":
    cli(prog_name="newspaper-explorer")
