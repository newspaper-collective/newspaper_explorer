"""
CLI commands for analysis group.
Organizes all analysis commands (layout, entities, keywords, topics, emotions, etc.).
"""

import click

from newspaper_explorer.cli.analyze.captions.commands import captions_group
from newspaper_explorer.cli.analyze.emotions.commands import emotions_group
from newspaper_explorer.cli.analyze.entities.commands import entities_group
from newspaper_explorer.cli.analyze.keywords.commands import keywords_group
from newspaper_explorer.cli.analyze.layout.commands import layout_group
from newspaper_explorer.cli.analyze.topics.commands import topics_group


@click.group()
def analyze() -> None:
    """Run analysis on newspaper data (entities, keywords, topics, layout, etc.)."""
    pass


# Register command groups
analyze.add_command(layout_group)
analyze.add_command(entities_group)
analyze.add_command(keywords_group)
analyze.add_command(topics_group)
analyze.add_command(emotions_group)
analyze.add_command(captions_group)
