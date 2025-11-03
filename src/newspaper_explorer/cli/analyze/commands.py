"""
CLI commands for analysis group.
Organizes all analysis commands (layout, entities, topics, etc.).
"""

import click

from newspaper_explorer.cli.analyze.layout.commands import layout_group
from newspaper_explorer.cli.analyze.entities.commands import entities_group
from newspaper_explorer.cli.analyze.keywords.commands import keywords_group


@click.group()
def analyze():
    """Run analysis on newspaper data (entities, topics, layout, etc.)."""
    pass


# Register command groups
analyze.add_command(layout_group)
analyze.add_command(entities_group)
analyze.add_command(keywords_group)
